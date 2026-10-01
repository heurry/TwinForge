package taskplan

import (
	"encoding/json"
	"errors"
	"fmt"
	"path"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"sync"
)

// VerificationAssertion is a language-neutral assertion over a Tool result.
// Path uses dotted object traversal, for example "exit_code" or
// "response.status". Providers may define additional operators.
type VerificationAssertion struct {
	Path     string          `json:"path"`
	Operator string          `json:"operator"`
	Value    json.RawMessage `json:"value,omitempty"`
}

// EvidenceReceipt is the immutable, successful Tool observation presented to
// a VerificationProvider. Persistence and provenance checks happen before a
// provider receives it.
type EvidenceReceipt struct {
	Tool      string
	Arguments json.RawMessage
	Result    json.RawMessage
}

// VerificationFailure classifies recovery without parsing human-readable
// error text. The wrapped error remains available for audit and UI detail.
type VerificationFailure struct {
	Reason string
	Err    error
}

func (e *VerificationFailure) Error() string {
	if e == nil || e.Err == nil {
		return "verification failed"
	}
	return e.Err.Error()
}

func (e *VerificationFailure) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Err
}

func NewVerificationFailure(reason string, err error) error {
	return &VerificationFailure{Reason: reason, Err: err}
}

func VerificationFailureReason(err error) string {
	var failure *VerificationFailure
	if errors.As(err, &failure) && strings.TrimSpace(failure.Reason) != "" {
		return failure.Reason
	}
	return VerificationReasonAssertionFailed
}

// VerificationProvider makes completion semantics extensible without adding
// another language-specific switch to the Plan runtime. Skills, MCP adapters,
// or deployment packages can register providers during process startup.
type VerificationProvider interface {
	Kind() string
	Validate(VerificationSpec) error
	EvidenceTools(VerificationSpec) []string
	Verify(VerificationSpec, []EvidenceReceipt) bool
	ActionHint(VerificationSpec) string
}

// verificationCandidateMatcher is an optional provider capability used before
// a receipt becomes a formal verification attempt. Candidate matching checks
// only whether the receipt addresses the declared subject/arguments; it must
// not inspect the verdict-bearing result fields.
type verificationCandidateMatcher interface {
	CandidateMatches(VerificationSpec, EvidenceReceipt) bool
}

var providers = struct {
	sync.RWMutex
	values map[string]VerificationProvider
}{values: make(map[string]VerificationProvider)}

// RegisterVerificationProvider installs one process-wide provider. Duplicate
// kinds are rejected so startup order cannot silently change evidence rules.
func RegisterVerificationProvider(provider VerificationProvider) error {
	if provider == nil || strings.TrimSpace(provider.Kind()) == "" {
		return errors.New("verification provider and kind are required")
	}
	kind := strings.TrimSpace(provider.Kind())
	providers.Lock()
	defer providers.Unlock()
	if _, exists := providers.values[kind]; exists {
		return fmt.Errorf("verification provider %q is already registered", kind)
	}
	providers.values[kind] = provider
	return nil
}

func verificationProvider(kind string) (VerificationProvider, bool) {
	providers.RLock()
	defer providers.RUnlock()
	provider, ok := providers.values[strings.TrimSpace(kind)]
	return provider, ok
}

// VerificationKinds returns the currently registered contract kinds for Tool
// descriptions and diagnostics. JSON Schema deliberately accepts a string;
// the registry remains the authoritative, extensible validator.
func VerificationKinds() []string {
	providers.RLock()
	defer providers.RUnlock()
	result := make([]string, 0, len(providers.values))
	for kind := range providers.values {
		result = append(result, kind)
	}
	sort.Strings(result)
	return result
}

func ValidateVerification(spec VerificationSpec) error {
	if strings.TrimSpace(spec.Kind) == "" {
		// Compatibility for plans written before typed verification.
		return nil
	}
	provider, ok := verificationProvider(spec.Kind)
	if !ok {
		return fmt.Errorf("verification provider %q is not registered", spec.Kind)
	}
	return provider.Validate(spec)
}

func VerificationEvidenceTools(spec VerificationSpec) []string {
	provider, ok := verificationProvider(spec.Kind)
	if !ok {
		return nil
	}
	return append([]string(nil), provider.EvidenceTools(spec)...)
}

func VerifyEvidence(spec VerificationSpec, receipts []EvidenceReceipt) error {
	provider, ok := verificationProvider(spec.Kind)
	if !ok {
		return fmt.Errorf("verification provider %q is not registered", spec.Kind)
	}
	if !provider.Verify(spec, receipts) {
		return fmt.Errorf("successful Tool receipts do not satisfy verification kind=%s target=%q", spec.Kind, spec.Target)
	}
	return nil
}

// VerificationCandidateMatches reports whether one receipt can prove the
// declared contract. A non-matching receipt is merely filtered evidence, not a
// failed verification attempt. Providers that do not expose a structural
// matcher retain compatibility by accepting only receipts they already verify.
func VerificationCandidateMatches(spec VerificationSpec, receipt EvidenceReceipt) bool {
	provider, ok := verificationProvider(spec.Kind)
	if !ok {
		return false
	}
	if matcher, ok := provider.(verificationCandidateMatcher); ok {
		return matcher.CandidateMatches(spec, receipt)
	}
	return provider.Verify(spec, []EvidenceReceipt{receipt})
}

func VerificationActionHint(spec VerificationSpec) string {
	provider, ok := verificationProvider(spec.Kind)
	if !ok {
		return "obtain a successful Tool receipt from a registered verification provider"
	}
	return provider.ActionHint(spec)
}

// NormalizeUpdate converts unambiguous legacy contracts before they become
// durable. It never turns a syntax-only observation into proof that a program
// actually starts or behaves correctly.
func NormalizeUpdate(update Update) (Update, error) {
	return normalizeUpdate(update, true)
}

func normalizeUpdate(update Update, sanitizePlatformState bool) (Update, error) {
	for stepIndex := range update.Steps {
		if sanitizePlatformState {
			// State is a platform-owned projection. The model may provide a
			// human-readable Result, but it cannot submit fake tests, tokens,
			// Artifact IDs, or retry pointers.
			update.Steps[stepIndex].State = NodeState{Output: update.Steps[stepIndex].Result}
		}
		NormalizeNodeState(&update.Steps[stepIndex])
		for criterionIndex := range update.Steps[stepIndex].AcceptanceCriteria {
			criterion := &update.Steps[stepIndex].AcceptanceCriteria[criterionIndex]
			criterion.Enforcement = effectiveEnforcement(criterion.Enforcement)
			criterion.Origin = effectiveOrigin(criterion.Origin)
			// A skipped advisory is a durable policy decision, not a verification
			// attempt. Re-validating its retired/unsupported provider would turn it
			// back into unsupported during every Plan normalization and erase the
			// recovery decision that revise_verification just committed.
			if criterion.Status == CriterionSkipped && !criterion.BlocksCompletion() {
				continue
			}
			normalized, err := NormalizeCriterionVerification(criterion.Description, criterion.Verification)
			if err != nil {
				if criterion.BlocksCompletion() {
					return Update{}, fmt.Errorf("step %q criterion %q: %w", update.Steps[stepIndex].ID, criterion.ID, err)
				}
				criterion.Status = CriterionInvalid
				criterion.VerificationReason = VerificationReasonSpecInvalid
				criterion.VerificationMessage = err.Error()
				continue
			}
			criterion.Verification = normalized
			if err := ValidateVerification(normalized); err != nil {
				if criterion.BlocksCompletion() {
					return Update{}, fmt.Errorf("step %q criterion %q: %w", update.Steps[stepIndex].ID, criterion.ID, err)
				}
				criterion.Status = CriterionInvalid
				criterion.VerificationReason = VerificationReasonSpecInvalid
				if _, exists := verificationProvider(normalized.Kind); strings.TrimSpace(normalized.Kind) != "" && !exists {
					criterion.Status = CriterionUnsupported
					criterion.VerificationReason = VerificationReasonProviderUnavailable
				}
				criterion.VerificationMessage = err.Error()
				continue
			}
			if criterion.Status == CriterionInvalid || criterion.Status == CriterionUnsupported {
				criterion.Status = CriterionPending
				criterion.VerificationReason = ""
				criterion.VerificationMessage = ""
			}
		}
	}
	return update, nil
}

// NormalizePlan provides read compatibility for already persisted JSONB Plans.
// It also repairs the impossible state where a completed downstream step
// depends on a step that has been reopened.
func NormalizePlan(plan Plan) (Plan, error) {
	update, err := normalizeUpdate(Update{Goal: plan.Goal, Explanation: plan.Explanation, Steps: plan.Steps}, false)
	if err != nil {
		return Plan{}, err
	}
	plan.Steps = repairDependencyProgress(update.Steps)
	plan.ExecutionOutcome, plan.VerificationOutcome = plan.DeriveOutcomes()
	plan.GraphState = plan.DeriveGraphState()
	return plan, nil
}

func NormalizeCriterionVerification(description string, spec VerificationSpec) (VerificationSpec, error) {
	spec.Kind = strings.TrimSpace(spec.Kind)
	spec.Target = strings.TrimSpace(spec.Target)
	if spec.Kind == "command_exit_zero" || spec.Kind == "test_pass" {
		spec = normalizeLegacyCommandSpec(spec)
	}
	spec = normalizeKnownAssertionTypes(spec)
	if spec.Kind != "command_exit_zero" {
		return spec, nil
	}
	target, ok := structuredPythonCompileTarget(spec.Arguments)
	if !ok {
		target, ok = legacyPythonCompileTarget(spec.Target)
	}
	if !ok {
		return spec, nil
	}
	lower := strings.ToLower(description)
	syntaxClaim := strings.Contains(lower, "syntax") || strings.Contains(lower, "compile") || strings.Contains(lower, "语法") || strings.Contains(lower, "编译")
	runtimeClaim := strings.Contains(lower, "runnable") || strings.Contains(lower, "run successfully") || strings.Contains(lower, "可运行") || strings.Contains(lower, "可以运行") || strings.Contains(lower, "直接运行") || strings.Contains(lower, "启动")
	if runtimeClaim && !syntaxClaim {
		return VerificationSpec{}, errors.New("py_compile proves syntax only and cannot prove that a program starts or runs successfully")
	}
	spec.Kind = "python_syntax"
	spec.Target = target
	spec.Arguments = nil
	return spec, nil
}

func normalizeLegacyCommandSpec(spec VerificationSpec) VerificationSpec {
	arguments := map[string]any{}
	if len(spec.Arguments) != 0 && json.Unmarshal(spec.Arguments, &arguments) != nil {
		return spec
	}
	delete(arguments, "exit_code")
	delete(arguments, "reading")
	delete(arguments, "working_directory")
	delete(arguments, "on_timeout")
	command := strings.TrimSpace(stringValue(arguments["command"]))
	args := stringSlice(arguments["args"])
	if len(args) == 0 && command != "" && !strings.Contains(command, " -c ") {
		fields := strings.Fields(command)
		if len(fields) > 1 {
			arguments["command"] = fields[0]
			arguments["args"] = fields[1:]
			args = fields[1:]
		}
	}
	if (command == "" || len(args) == 0) && !strings.Contains(spec.Target, " -c ") {
		fields := strings.Fields(spec.Target)
		if len(fields) > 1 && (fields[0] == "python" || fields[0] == "python3") {
			arguments["command"] = fields[0]
			arguments["args"] = fields[1:]
		}
	}
	if strings.TrimSpace(stringValue(arguments["command"])) != "" && len(stringSlice(arguments["args"])) != 0 {
		spec.Arguments, _ = json.Marshal(arguments)
	}
	return spec
}

// structuredPythonCompileTarget is the first built-in Verification Compiler:
// it derives the syntax subject from the canonical invocation instead of
// trusting a second, independently model-authored target string.
func structuredPythonCompileTarget(raw json.RawMessage) (string, bool) {
	if len(raw) == 0 {
		return "", false
	}
	var arguments map[string]any
	if json.Unmarshal(raw, &arguments) != nil {
		return "", false
	}
	command := strings.TrimSpace(stringValue(arguments["command"]))
	args := stringSlice(arguments["args"])
	if (command != "python" && command != "python3") || len(args) != 3 || args[0] != "-m" || (args[1] != "py_compile" && args[1] != "compileall") {
		return "", false
	}
	target := canonicalVerificationPath(args[2])
	return target, target != ""
}

func normalizeKnownAssertionTypes(spec VerificationSpec) VerificationSpec {
	if spec.Kind != "tool_receipt" || strings.TrimSpace(spec.Tool) != "run_command" {
		return spec
	}
	for index := range spec.Assertions {
		assertion := &spec.Assertions[index]
		var text string
		if json.Unmarshal(assertion.Value, &text) != nil {
			continue
		}
		switch assertion.Path {
		case "exit_code", "duration_ms":
			if value, err := strconv.ParseInt(strings.TrimSpace(text), 10, 64); err == nil {
				assertion.Value, _ = json.Marshal(value)
			}
		case "timed_out":
			if value, err := strconv.ParseBool(strings.TrimSpace(text)); err == nil {
				assertion.Value, _ = json.Marshal(value)
			}
		}
	}
	return spec
}

func legacyPythonCompileTarget(value string) (string, bool) {
	fields := strings.Fields(value)
	if len(fields) != 4 || (fields[0] != "python" && fields[0] != "python3") || fields[1] != "-m" || fields[2] != "py_compile" {
		return "", false
	}
	target := canonicalVerificationPath(fields[3])
	return target, target != ""
}

func repairDependencyProgress(steps []Step) []Step {
	result := append([]Step(nil), steps...)
	byID := make(map[string]int, len(result))
	for index := range result {
		byID[result[index].ID] = index
	}
	for changed := true; changed; {
		changed = false
		for index := range result {
			if result[index].Status != StatusCompleted && result[index].Status != StatusSkipped {
				continue
			}
			for _, dependency := range result[index].DependsOn {
				dependencyIndex, exists := byID[dependency]
				if exists && result[dependencyIndex].Status != StatusCompleted && result[dependencyIndex].Status != StatusSkipped {
					resetStepProgress(&result[index])
					changed = true
					break
				}
			}
		}
	}
	return result
}

func resetStepProgress(step *Step) {
	step.Status = StatusPending
	step.Result = ""
	for index := range step.AcceptanceCriteria {
		step.AcceptanceCriteria[index].Status = CriterionPending
		step.AcceptanceCriteria[index].Evidence = ""
		step.AcceptanceCriteria[index].EvidenceCallIDs = nil
	}
}

type builtinProvider struct {
	kind      string
	tools     []string
	validate  func(VerificationSpec) error
	candidate func(VerificationSpec, EvidenceReceipt) bool
	verify    func(VerificationSpec, []EvidenceReceipt) bool
	hint      func(VerificationSpec) string
}

func (p builtinProvider) Kind() string { return p.kind }
func (p builtinProvider) Validate(spec VerificationSpec) error {
	if p.validate != nil {
		return p.validate(spec)
	}
	return nil
}
func (p builtinProvider) EvidenceTools(VerificationSpec) []string { return p.tools }
func (p builtinProvider) CandidateMatches(spec VerificationSpec, receipt EvidenceReceipt) bool {
	if p.candidate != nil {
		return p.candidate(spec, receipt)
	}
	if len(p.tools) == 0 {
		return true
	}
	for _, name := range p.tools {
		if receipt.Tool == name {
			return true
		}
	}
	return false
}
func (p builtinProvider) Verify(spec VerificationSpec, receipts []EvidenceReceipt) bool {
	return p.verify != nil && p.verify(spec, receipts)
}
func (p builtinProvider) ActionHint(spec VerificationSpec) string {
	if p.hint != nil {
		return p.hint(spec)
	}
	return "obtain a successful Tool receipt that directly proves this criterion"
}

func init() {
	registerBuiltinProviders()
}

func registerBuiltinProviders() {
	mustRegister(toolSuccessProvider{})
	mustRegister(pathProvider("file_exists", []string{"read_file", "write_file", "append_file", "edit_file", "promote_file"}, func(receipt EvidenceReceipt, _ map[string]any, _ map[string]any) bool {
		return isFileTool(receipt.Tool)
	}))
	mustRegister(builtinProvider{kind: "file_contains", tools: []string{"read_file"}, validate: requireTargetAndMatch, candidate: func(spec VerificationSpec, receipt EvidenceReceipt) bool {
		arguments, _ := decodeReceipt(receipt)
		return receipt.Tool == "read_file" && receiptPathMatches(arguments, spec.Target)
	}, verify: func(spec VerificationSpec, values []EvidenceReceipt) bool {
		for _, receipt := range values {
			arguments, result := decodeReceipt(receipt)
			if receipt.Tool == "read_file" && receiptPathMatches(arguments, spec.Target) && strings.Contains(stringValue(result["content"]), spec.Match) {
				return true
			}
		}
		return false
	}, hint: func(spec VerificationSpec) string {
		return fmt.Sprintf("call read_file for %q and verify %q in its result", spec.Target, spec.Match)
	}})
	mustRegister(builtinProvider{kind: "python_syntax", tools: []string{"run_command"}, validate: requireTarget, candidate: func(spec VerificationSpec, receipt EvidenceReceipt) bool {
		arguments, _ := decodeReceipt(receipt)
		return receipt.Tool == "run_command" && commandTargets(arguments, spec.Target) && isPythonCompile(arguments)
	}, verify: func(spec VerificationSpec, values []EvidenceReceipt) bool {
		for _, receipt := range values {
			arguments, _ := decodeReceipt(receipt)
			if receipt.Tool == "run_command" && commandTargets(arguments, spec.Target) && isPythonCompile(arguments) {
				return true
			}
		}
		return false
	}, hint: func(spec VerificationSpec) string {
		return fmt.Sprintf("call run_command with command=python3 and args=[\"-m\",\"py_compile\",%q], requiring exit_code=0", spec.Target)
	}})
	mustRegister(builtinProvider{kind: "command_exit_zero", tools: []string{"run_command"}, validate: requireCommandVerification, candidate: func(spec VerificationSpec, receipt EvidenceReceipt) bool {
		arguments, _ := decodeReceipt(receipt)
		return receipt.Tool == "run_command" && commandVerificationMatches(arguments, spec) && !isPythonCompile(arguments)
	}, verify: func(spec VerificationSpec, values []EvidenceReceipt) bool {
		for _, receipt := range values {
			arguments, result := decodeReceipt(receipt)
			if receipt.Tool == "run_command" && commandVerificationMatches(arguments, spec) && !isPythonCompile(arguments) && numberValue(result["exit_code"]) == 0 {
				return true
			}
		}
		return false
	}, hint: func(spec VerificationSpec) string {
		if len(spec.Arguments) != 0 {
			return fmt.Sprintf("call run_command with arguments=%s and require exit_code=0", compactJSON(spec.Arguments))
		}
		return fmt.Sprintf("call run_command for target %q and require exit_code=0", spec.Target)
	}})
	mustRegister(builtinProvider{kind: "test_pass", tools: []string{"run_command"}, validate: requireCommandVerification, candidate: func(spec VerificationSpec, receipt EvidenceReceipt) bool {
		arguments, _ := decodeReceipt(receipt)
		return receipt.Tool == "run_command" && commandVerificationMatches(arguments, spec) && isTestCommand(arguments)
	}, verify: func(spec VerificationSpec, values []EvidenceReceipt) bool {
		for _, receipt := range values {
			arguments, result := decodeReceipt(receipt)
			if receipt.Tool == "run_command" && commandVerificationMatches(arguments, spec) && isTestCommand(arguments) && numberValue(result["exit_code"]) == 0 {
				return true
			}
		}
		return false
	}, hint: func(spec VerificationSpec) string {
		return fmt.Sprintf("run the registered test provider for %q and require a passing result", spec.Target)
	}})
	mustRegister(pathProvider("list_nonempty", []string{"list_files"}, func(receipt EvidenceReceipt, _ map[string]any, result map[string]any) bool {
		return receipt.Tool == "list_files" && sliceLength(result["entries"]) > 0
	}))
	mustRegister(pathProvider("search_nonempty", []string{"search_files"}, func(receipt EvidenceReceipt, _ map[string]any, result map[string]any) bool {
		return receipt.Tool == "search_files" && sliceLength(result["results"]) > 0
	}))
	mustRegister(genericToolReceiptProvider{})
}

func mustRegister(provider VerificationProvider) {
	if err := RegisterVerificationProvider(provider); err != nil {
		panic(err)
	}
}

func pathProvider(kind string, tools []string, predicate func(EvidenceReceipt, map[string]any, map[string]any) bool) VerificationProvider {
	return builtinProvider{kind: kind, tools: tools, validate: requireTarget, candidate: func(spec VerificationSpec, receipt EvidenceReceipt) bool {
		arguments, _ := decodeReceipt(receipt)
		if !receiptPathMatches(arguments, spec.Target) {
			return false
		}
		for _, name := range tools {
			if receipt.Tool == name {
				return true
			}
		}
		return false
	}, verify: func(spec VerificationSpec, values []EvidenceReceipt) bool {
		for _, receipt := range values {
			arguments, result := decodeReceipt(receipt)
			if receiptPathMatches(arguments, spec.Target) && predicate(receipt, arguments, result) {
				return true
			}
		}
		return false
	}, hint: func(spec VerificationSpec) string {
		return fmt.Sprintf("obtain a current %s Tool receipt for %q", kind, spec.Target)
	}}
}

type genericToolReceiptProvider struct{}

// toolSuccessProvider is the least-specific supported verification contract.
// When a tool or argument subset is declared it still binds to that exact
// action, preventing an unrelated successful receipt from advancing a legacy
// advisory criterion during migration.
type toolSuccessProvider struct{}

func (toolSuccessProvider) Kind() string { return "tool_success" }
func (toolSuccessProvider) Validate(spec VerificationSpec) error {
	if len(spec.Arguments) == 0 {
		return nil
	}
	var object map[string]any
	if json.Unmarshal(spec.Arguments, &object) != nil || object == nil {
		return errors.New("tool_success arguments must be a JSON object")
	}
	return nil
}
func (toolSuccessProvider) EvidenceTools(spec VerificationSpec) []string {
	if strings.TrimSpace(spec.Tool) == "" {
		return nil
	}
	return []string{strings.TrimSpace(spec.Tool)}
}
func (toolSuccessProvider) CandidateMatches(spec VerificationSpec, receipt EvidenceReceipt) bool {
	if strings.TrimSpace(spec.Tool) != "" && receipt.Tool != strings.TrimSpace(spec.Tool) {
		return false
	}
	var expected map[string]any
	_ = json.Unmarshal(spec.Arguments, &expected)
	arguments, _ := decodeReceipt(receipt)
	return objectContains(arguments, expected)
}
func (provider toolSuccessProvider) Verify(spec VerificationSpec, receipts []EvidenceReceipt) bool {
	for _, receipt := range receipts {
		if provider.CandidateMatches(spec, receipt) {
			return true
		}
	}
	return false
}
func (toolSuccessProvider) ActionHint(spec VerificationSpec) string {
	if strings.TrimSpace(spec.Tool) != "" {
		return fmt.Sprintf("obtain one successful %s Tool receipt matching the declared arguments", strings.TrimSpace(spec.Tool))
	}
	return "obtain one successful Tool receipt directly related to this criterion"
}

func (genericToolReceiptProvider) Kind() string { return "tool_receipt" }
func (genericToolReceiptProvider) Validate(spec VerificationSpec) error {
	if strings.TrimSpace(spec.Tool) == "" {
		return errors.New("tool_receipt verification requires tool")
	}
	switch strings.TrimSpace(spec.Tool) {
	case "write_file", "append_file", "edit_file", "promote_file":
		return errors.New("workspace mutation receipts are not delivery outcomes; use file_exists, file_contains, python_syntax, command_exit_zero, or test_pass")
	}
	if len(spec.Assertions) == 0 {
		return errors.New("tool_receipt verification requires at least one assertion")
	}
	if len(spec.Arguments) != 0 {
		var object map[string]any
		if json.Unmarshal(spec.Arguments, &object) != nil || object == nil {
			return errors.New("tool_receipt arguments must be a JSON object")
		}
	}
	for _, assertion := range spec.Assertions {
		if strings.TrimSpace(assertion.Path) == "" {
			return errors.New("tool_receipt assertion path is required")
		}
		switch assertion.Operator {
		case "equals", "contains", "nonempty":
		default:
			return fmt.Errorf("unsupported tool_receipt assertion operator %q", assertion.Operator)
		}
		if assertion.Operator != "nonempty" && len(assertion.Value) == 0 {
			return errors.New("tool_receipt equals/contains assertion requires value")
		}
	}
	return nil
}
func (genericToolReceiptProvider) EvidenceTools(spec VerificationSpec) []string {
	return []string{spec.Tool}
}
func (genericToolReceiptProvider) CandidateMatches(spec VerificationSpec, receipt EvidenceReceipt) bool {
	if receipt.Tool != spec.Tool {
		return false
	}
	var expected map[string]any
	_ = json.Unmarshal(spec.Arguments, &expected)
	arguments, _ := decodeReceipt(receipt)
	return objectContains(arguments, expected)
}
func (genericToolReceiptProvider) Verify(spec VerificationSpec, receipts []EvidenceReceipt) bool {
	var expected map[string]any
	_ = json.Unmarshal(spec.Arguments, &expected)
	for _, receipt := range receipts {
		if receipt.Tool != spec.Tool {
			continue
		}
		arguments, result := decodeReceipt(receipt)
		if !objectContains(arguments, expected) {
			continue
		}
		matched := true
		for _, assertion := range spec.Assertions {
			if !assertionMatches(result, assertion) {
				matched = false
				break
			}
		}
		if matched {
			return true
		}
	}
	return false
}
func (genericToolReceiptProvider) ActionHint(spec VerificationSpec) string {
	return fmt.Sprintf("call %s with the declared arguments and satisfy its structured result assertions", spec.Tool)
}

func requireTarget(spec VerificationSpec) error {
	if strings.TrimSpace(spec.Target) == "" {
		return errors.New("typed acceptance verification requires a target")
	}
	return nil
}

func requireCommandVerification(spec VerificationSpec) error {
	if len(spec.Arguments) == 0 {
		return requireTarget(spec)
	}
	var arguments map[string]any
	if json.Unmarshal(spec.Arguments, &arguments) != nil || arguments == nil {
		return errors.New("command verification arguments must be a JSON object")
	}
	if strings.TrimSpace(stringValue(arguments["command"])) == "" || len(stringSlice(arguments["args"])) == 0 {
		return errors.New("command verification arguments require command and non-empty args")
	}
	return nil
}
func requireTargetAndMatch(spec VerificationSpec) error {
	if err := requireTarget(spec); err != nil {
		return err
	}
	if strings.TrimSpace(spec.Match) == "" {
		return errors.New("file_contains verification requires match text")
	}
	return nil
}

func decodeReceipt(receipt EvidenceReceipt) (map[string]any, map[string]any) {
	arguments, result := map[string]any{}, map[string]any{}
	_ = json.Unmarshal(receipt.Arguments, &arguments)
	_ = json.Unmarshal(receipt.Result, &result)
	return arguments, result
}
func receiptPathMatches(arguments map[string]any, target string) bool {
	actual := stringValue(arguments["path"])
	if strings.TrimSpace(actual) == "" {
		actual = stringValue(arguments["target_path"])
	}
	return canonicalVerificationPath(actual) == canonicalVerificationPath(target)
}
func isFileTool(name string) bool {
	switch name {
	case "read_file", "write_file", "append_file", "edit_file", "promote_file":
		return true
	default:
		return false
	}
}
func commandTargets(arguments map[string]any, target string) bool {
	target = canonicalVerificationPath(target)
	if target == "" {
		return false
	}
	working := canonicalVerificationPath(stringValue(arguments["working_directory"]))
	for _, argument := range stringSlice(arguments["args"]) {
		candidate := canonicalVerificationPath(argument)
		if working != "" && working != "." {
			candidate = canonicalVerificationPath(working + "/" + candidate)
		}
		if candidate == target {
			return true
		}
	}
	return false
}
func commandVerificationMatches(arguments map[string]any, spec VerificationSpec) bool {
	if len(spec.Arguments) == 0 {
		return commandTargets(arguments, spec.Target)
	}
	var expected map[string]any
	if json.Unmarshal(spec.Arguments, &expected) != nil {
		return false
	}
	return objectContains(arguments, expected)
}
func isPythonCompile(arguments map[string]any) bool {
	args := stringSlice(arguments["args"])
	return len(args) >= 3 && args[0] == "-m" && (args[1] == "py_compile" || args[1] == "compileall")
}
func isTestCommand(arguments map[string]any) bool {
	command := strings.ToLower(path.Base(strings.TrimSpace(stringValue(arguments["command"]))))
	args := stringSlice(arguments["args"])
	if len(args) == 0 {
		return false
	}
	if (command == "python" || command == "python3") && len(args) >= 2 && args[0] == "-m" && (args[1] == "unittest" || args[1] == "pytest") {
		return true
	}
	if command == "pytest" || (command == "go" && args[0] == "test") ||
		(command == "cargo" && args[0] == "test") ||
		((command == "npm" || command == "pnpm" || command == "yarn") && args[0] == "test") {
		return true
	}
	if command != "python" && command != "python3" {
		return false
	}
	// A repository may intentionally use a standalone test/smoke/verify script
	// instead of a framework module. The test_pass criterion is already an
	// explicit test contract, so accept only recognizably test-shaped .py entry
	// points rather than treating every successful Python script as a test.
	entry := strings.ToLower(path.Base(strings.TrimSpace(args[0])))
	return strings.HasSuffix(entry, ".py") &&
		(strings.Contains(entry, "test") || strings.Contains(entry, "smoke") || strings.Contains(entry, "verify"))
}
func canonicalVerificationPath(value string) string {
	value = strings.TrimSpace(strings.ReplaceAll(value, "\\", "/"))
	value = strings.TrimPrefix(value, "/workspace/")
	value = strings.TrimPrefix(value, "./")
	value = strings.TrimPrefix(value, "/")
	if value == "" {
		return ""
	}
	cleaned := path.Clean(value)
	if cleaned == "." || cleaned == ".." || strings.HasPrefix(cleaned, "../") {
		return ""
	}
	return cleaned
}
func stringValue(value any) string  { text, _ := value.(string); return text }
func numberValue(value any) float64 { number, _ := value.(float64); return number }
func compactJSON(value json.RawMessage) string {
	return strings.TrimSpace(string(value))
}
func stringSlice(value any) []string {
	items, _ := value.([]any)
	result := make([]string, 0, len(items))
	for _, item := range items {
		if text, ok := item.(string); ok {
			result = append(result, text)
		}
	}
	return result
}
func sliceLength(value any) int { items, _ := value.([]any); return len(items) }

func objectContains(actual, expected map[string]any) bool {
	for key, expectedValue := range expected {
		actualValue, exists := actual[key]
		if !exists {
			return false
		}
		expectedObject, nested := expectedValue.(map[string]any)
		if nested {
			actualObject, ok := actualValue.(map[string]any)
			if !ok || !objectContains(actualObject, expectedObject) {
				return false
			}
			continue
		}
		if !reflect.DeepEqual(actualValue, expectedValue) {
			return false
		}
	}
	return true
}

func assertionMatches(result map[string]any, assertion VerificationAssertion) bool {
	var value any = result
	for _, segment := range strings.Split(assertion.Path, ".") {
		object, ok := value.(map[string]any)
		if !ok {
			return false
		}
		value, ok = object[segment]
		if !ok {
			return false
		}
	}
	switch assertion.Operator {
	case "nonempty":
		switch typed := value.(type) {
		case string:
			return strings.TrimSpace(typed) != ""
		case []any:
			return len(typed) != 0
		case map[string]any:
			return len(typed) != 0
		default:
			return value != nil
		}
	case "contains":
		var expected any
		if json.Unmarshal(assertion.Value, &expected) != nil {
			return false
		}
		return strings.Contains(fmt.Sprint(value), fmt.Sprint(expected))
	case "equals":
		var expected any
		if json.Unmarshal(assertion.Value, &expected) != nil {
			return false
		}
		return reflect.DeepEqual(value, expected)
	default:
		return false
	}
}

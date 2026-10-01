package harness

import (
	"fmt"
	"sort"
	"strings"
)

// Descriptor is the public, runtime-backed capability contract for a Harness.
// Agent versions may only reference names present in this registry.
type Descriptor struct {
	Name               string `json:"name"`
	DisplayName        string `json:"display_name"`
	SupportsCheckpoint bool   `json:"supports_checkpoint"`
	SupportsTools      bool   `json:"supports_tools"`
	SupportsMultiTurn  bool   `json:"supports_multi_turn"`
}

var builtins = map[string]Descriptor{
	"react-v1": {
		Name: "react-v1", DisplayName: "ReAct v1",
		SupportsCheckpoint: true, SupportsTools: true, SupportsMultiTurn: true,
	},
}

// Lookup returns a registered Harness descriptor.
func Lookup(name string) (Descriptor, bool) {
	descriptor, ok := builtins[strings.TrimSpace(name)]
	return descriptor, ok
}

// ValidateName rejects a declarative Harness that this runtime cannot execute.
func ValidateName(name string) error {
	if _, ok := Lookup(name); !ok {
		return fmt.Errorf("unsupported harness %q", name)
	}
	return nil
}

// Descriptors returns a stable snapshot for capability discovery and UIs.
func Descriptors() []Descriptor {
	result := make([]Descriptor, 0, len(builtins))
	for _, descriptor := range builtins {
		result = append(result, descriptor)
	}
	sort.Slice(result, func(i, j int) bool { return result[i].Name < result[j].Name })
	return result
}

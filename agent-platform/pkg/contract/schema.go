// Package contract compiles and enforces JSON Schema contracts at runtime.
package contract

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"

	jsonschema "github.com/santhosh-tekuri/jsonschema/v5"
)

// Schema is an immutable compiled JSON Schema.
type Schema struct{ compiled *jsonschema.Schema }

// Compile validates the schema document itself, including referenced keywords.
func Compile(raw json.RawMessage) (*Schema, error) {
	if len(raw) == 0 || !json.Valid(raw) {
		return nil, errors.New("schema must be valid JSON")
	}
	compiler := jsonschema.NewCompiler()
	compiler.Draft = jsonschema.Draft2020
	if err := compiler.AddResource("memory://schema.json", bytes.NewReader(raw)); err != nil {
		return nil, fmt.Errorf("load schema: %w", err)
	}
	compiled, err := compiler.Compile("memory://schema.json")
	if err != nil {
		return nil, fmt.Errorf("compile schema: %w", err)
	}
	return &Schema{compiled: compiled}, nil
}

// ValidateJSON validates a JSON instance against a compiled schema.
func (s *Schema) ValidateJSON(raw json.RawMessage) error {
	if s == nil || s.compiled == nil {
		return errors.New("compiled schema is required")
	}
	var value any
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	if err := decoder.Decode(&value); err != nil {
		return fmt.Errorf("decode instance: %w", err)
	}
	if err := s.compiled.Validate(value); err != nil {
		return fmt.Errorf("schema validation failed: %w", err)
	}
	return nil
}

// Validate is the convenience path for immutable Agent input/output schemas.
func Validate(schema, instance json.RawMessage) error {
	compiled, err := Compile(schema)
	if err != nil {
		return err
	}
	return compiled.ValidateJSON(instance)
}

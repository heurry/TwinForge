// Package interaction defines durable pauses where an Agent needs user input.
package interaction

import (
	"errors"
	"time"
)

var (
	ErrInputRequired = errors.New("user input is required")
	ErrNotFound      = errors.New("user question not found")
)

type Request struct {
	Question string   `json:"question"`
	Options  []string `json:"options,omitempty"`
	Context  string   `json:"context,omitempty"`
}

type Question struct {
	ID         string     `json:"id"`
	RunID      string     `json:"run_id"`
	CallID     string     `json:"call_id"`
	Turn       int        `json:"turn"`
	Step       int        `json:"step"`
	Question   string     `json:"question"`
	Options    []string   `json:"options,omitempty"`
	Context    string     `json:"context,omitempty"`
	Answer     string     `json:"answer,omitempty"`
	Status     string     `json:"status"`
	CreatedAt  time.Time  `json:"created_at"`
	AnsweredBy *string    `json:"answered_by,omitempty"`
	AnsweredAt *time.Time `json:"answered_at,omitempty"`
}

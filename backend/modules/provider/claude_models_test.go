package provider

import (
	"reflect"
	"testing"
)

func TestParseClaudeModelID(t *testing.T) {
	tests := map[string]struct {
		want ClaudeModelRelease
		ok   bool
	}{
		"claude-opus-5-5":                {want: ClaudeModelRelease{ID: "claude-opus-5-5", Family: "opus", Version: []int{5, 5}}, ok: true},
		"cc/claude-fable-5-1":            {want: ClaudeModelRelease{ID: "claude-fable-5-1", Family: "fable", Version: []int{5, 1}}, ok: true},
		"cli-cc/claude-opus-4-6:low":     {want: ClaudeModelRelease{ID: "claude-opus-4-6", Family: "opus", Version: []int{4, 6}}, ok: true},
		"anthropic/claude-sonnet-5":      {want: ClaudeModelRelease{ID: "claude-sonnet-5", Family: "sonnet", Version: []int{5}}, ok: true},
		"claude-haiku-4-5-20251001":      {want: ClaudeModelRelease{ID: "claude-haiku-4-5-20251001", Family: "haiku", Version: []int{4, 5}, Dated: true}, ok: true},
		"claude-mythos-preview":          {want: ClaudeModelRelease{ID: "claude-mythos-preview", Family: "mythos-preview"}, ok: true},
		"claude-3-5-sonnet-20241022":     {ok: false},
		"gpt-5":                          {ok: false},
		"claude-":                        {ok: false},
		"claude-opus-4-6-20260101-extra": {ok: false},
	}
	for id, test := range tests {
		got, ok := ParseClaudeModelID(id)
		if ok != test.ok {
			t.Errorf("ParseClaudeModelID(%q) ok = %v, want %v", id, ok, test.ok)
			continue
		}
		if ok && !reflect.DeepEqual(got, test.want) {
			t.Errorf("ParseClaudeModelID(%q) = %#v, want %#v", id, got, test.want)
		}
	}
}

func TestClaudeCapabilitiesFollowTheRelease(t *testing.T) {
	tests := map[string]struct {
		adaptive bool
		effort   bool
		input    int
	}{
		"claude-fable-5-1":           {adaptive: true, effort: true, input: 1_000_000},
		"claude-fable-6":             {adaptive: true, effort: true, input: 1_000_000},
		"claude-mythos-preview":      {adaptive: true, effort: true, input: 1_000_000},
		"claude-opus-5-5":            {adaptive: true, effort: true, input: 1_000_000},
		"claude-opus-6":              {adaptive: true, effort: true, input: 1_000_000},
		"claude-opus-4-8":            {adaptive: true, effort: true, input: 1_000_000},
		"claude-opus-4-6":            {adaptive: true, effort: true, input: 1_000_000},
		"claude-opus-4-5":            {adaptive: false, effort: true, input: 200_000},
		"claude-sonnet-5":            {adaptive: true, effort: true, input: 1_000_000},
		"claude-sonnet-4-6":          {adaptive: true, effort: true, input: 1_000_000},
		"claude-sonnet-4-5":          {adaptive: false, effort: false, input: 200_000},
		"claude-haiku-5":             {adaptive: true, effort: true, input: 1_000_000},
		"claude-haiku-4-5":           {adaptive: false, effort: false, input: 200_000},
		"claude-haiku-4-5-20251001":  {adaptive: false, effort: false, input: 200_000},
		"claude-3-5-sonnet-20241022": {adaptive: false, effort: false, input: 200_000},
	}
	for id, test := range tests {
		if got := ClaudeSupportsAdaptiveThinking(id); got != test.adaptive {
			t.Errorf("ClaudeSupportsAdaptiveThinking(%q) = %v, want %v", id, got, test.adaptive)
		}
		if got := ClaudeSupportsEffort(id); got != test.effort {
			t.Errorf("ClaudeSupportsEffort(%q) = %v, want %v", id, got, test.effort)
		}
		if input, _ := ClaudeModelLimits(id); input != test.input {
			t.Errorf("ClaudeModelLimits(%q) input = %d, want %d", id, input, test.input)
		}
	}
}

func TestNewestClaudeModelIDPrefersTheLatestUndatedRelease(t *testing.T) {
	catalog := []Model{
		{ID: "cc/claude-opus-5"},
		{ID: "cc/claude-opus-5-5-20260801"},
		{ID: "cc/claude-opus-5-5"},
		{ID: "cc/claude-opus-4-8"},
		{ID: "cc/claude-sonnet-5"},
		{ID: "cc/claude-3-5-sonnet-20241022"},
	}
	if got, ok := NewestClaudeModelID("opus", catalog); !ok || got != "claude-opus-5-5" {
		t.Errorf("newest opus = %q (ok=%v), want claude-opus-5-5", got, ok)
	}
	if got, ok := NewestClaudeModelID("sonnet", catalog); !ok || got != "claude-sonnet-5" {
		t.Errorf("newest sonnet = %q (ok=%v), want claude-sonnet-5", got, ok)
	}
	if _, ok := NewestClaudeModelID("haiku", catalog); ok {
		t.Error("expected no haiku release in the catalog")
	}
}

package provider

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestParseClaudeCodeVersion(t *testing.T) {
	tests := map[string]string{
		"2.1.280 (Claude Code)":    "2.1.280",
		"2.1.280\n":                "2.1.280",
		"  v2.1.281 (Claude Code)": "2.1.281",
		"2.1":                      "",
		"":                         "",
	}
	for input, want := range tests {
		if got := parseClaudeCodeVersion(input); got != want {
			t.Errorf("parseClaudeCodeVersion(%q) = %q, want %q", input, got, want)
		}
	}
}

func TestCompareDottedVersions(t *testing.T) {
	tests := []struct {
		a, b string
		want int
	}{
		{"2.1.280", "2.1.219", 1},
		{"2.1.219", "2.1.280", -1},
		{"2.1.280", "2.1.280", 0},
		{"2.2.0", "2.1.999", 1},
		{"3.0.0", "2.9.9", 1},
		{"2.1.280", "", 1},
		{"", "", 0},
		{"2.1", "2.1.0", 0},
	}
	for _, test := range tests {
		if got := compareDottedVersions(test.a, test.b); got != test.want {
			t.Errorf("compareDottedVersions(%q, %q) = %d, want %d", test.a, test.b, got, test.want)
		}
	}
}

func TestFetchLatestClaudeCodeVersionReadsTheNpmManifest(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"name":"@anthropic-ai/claude-code","version":"2.1.300"}`))
	}))
	defer server.Close()

	if got := fetchLatestClaudeCodeVersion(context.Background(), server.URL); got != "2.1.300" {
		t.Fatalf("fetchLatestClaudeCodeVersion = %q, want 2.1.300", got)
	}

	failing := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer failing.Close()
	if got := fetchLatestClaudeCodeVersion(context.Background(), failing.URL); got != "" {
		t.Fatalf("a failing registry should yield no version, got %q", got)
	}
}

func TestClaudeCodeVersionStateLearnsUpstreamRequirementsAndNeverLowers(t *testing.T) {
	state := &claudeCodeVersionState{version: "2.1.219"}

	rejection := `{"type":"error","error":{"type":"invalid_request_error","message":"Claude Code 2.1.219 does not support this model; version 2.1.280 or newer is required. Run 'claude update'."}}`
	match := claudeCodeRequirementPattern.FindStringSubmatch(rejection)
	if match == nil {
		t.Fatal("requirement pattern did not match the upstream rejection")
	}
	if !state.raise(match[1], "test") || state.version != "2.1.280" {
		t.Fatalf("expected the state to adopt 2.1.280, got %q", state.version)
	}
	if state.raise("2.1.200", "test") {
		t.Fatal("an older version must never replace a newer one")
	}
	if state.raise("2.1.280", "test") {
		t.Fatal("the same version is not a raise")
	}
	if state.raise("", "test") {
		t.Fatal("an empty version is not a raise")
	}

	if learnClaudeCodeVersionRequirement(`{"error":{"message":"Overloaded"}}`) {
		t.Fatal("unrelated errors carry no requirement")
	}
}

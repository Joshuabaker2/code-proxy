package auth

import "testing"

func TestProviderNameForType(t *testing.T) {
	if got := ProviderNameForType("anthropic-api"); got != "claude" {
		t.Fatalf("anthropic OAuth mapped to %q", got)
	}
	if got := ProviderNameForType("custom"); got != "custom" {
		t.Fatalf("unknown provider changed to %q", got)
	}
}

func TestClaudeUsesCurrentClaudeCodeTokenService(t *testing.T) {
	config, ok := GetConfig("claude")
	if !ok {
		t.Fatal("Claude OAuth config is missing")
	}
	if config.TokenURL != "https://platform.claude.com/v1/oauth/token" {
		t.Fatalf("Claude token URL = %q", config.TokenURL)
	}
}

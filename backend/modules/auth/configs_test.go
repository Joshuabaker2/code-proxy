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

package api

import (
	"strings"
	"testing"
	"time"
)

func TestClaudeOAuthTestHeadersUseBearerAuth(t *testing.T) {
	headers := claudeOAuthTestHeaders("oauth-token")
	if headers["Authorization"] != "Bearer oauth-token" {
		t.Fatalf("OAuth bearer token was not used: %#v", headers)
	}
	if _, ok := headers["x-api-key"]; ok {
		t.Fatalf("OAuth token was incorrectly sent as an API key: %#v", headers)
	}
	if !strings.Contains(headers["anthropic-beta"], "oauth-2025-04-20") {
		t.Fatalf("OAuth beta header is missing: %#v", headers)
	}
}

func TestOAuthTokenNeedsRefreshBeforeExpiry(t *testing.T) {
	soon := time.Now().Add(2 * time.Minute)
	later := time.Now().Add(10 * time.Minute)
	if !oauthTokenNeedsRefresh(&soon) {
		t.Fatal("token expiring soon was not selected for refresh")
	}
	if oauthTokenNeedsRefresh(&later) {
		t.Fatal("token with sufficient lifetime was selected for refresh")
	}
	if oauthTokenNeedsRefresh(nil) {
		t.Fatal("account without an expiration was selected for refresh")
	}
}

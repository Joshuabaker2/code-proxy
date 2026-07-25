package api

import (
	"testing"

	"code-proxy/modules/database"
)

func TestSanitizedAccountRemovesSecrets(t *testing.T) {
	original := database.Account{
		AccessToken:  "access-secret",
		RefreshToken: "refresh-secret",
		APIKey:       "api-secret",
		Metadata: map[string]string{
			"id_token": "identity-secret",
			"base_url": "http://127.0.0.1",
		},
	}

	got := sanitizedAccount(original)
	if got.AccessToken != "" || got.RefreshToken != "" || got.APIKey != "" {
		t.Fatalf("account secrets were returned: %#v", got)
	}
	if _, ok := got.Metadata["id_token"]; ok {
		t.Fatal("ID token was returned in account metadata")
	}
	if got.Metadata["base_url"] != "http://127.0.0.1" {
		t.Fatal("non-secret metadata was removed")
	}
	if original.Metadata["id_token"] != "identity-secret" {
		t.Fatal("sanitization mutated the stored account")
	}
}

package api

import (
	"testing"
	"time"
)

// The desktop app drives its "Reconnect Claude" prompt off this flag. A
// cooling-down account is still signed in, so reporting it as unauthenticated
// would send the user to re-run OAuth to fix a ten-second backoff.
func TestEmbeddedAuthStatusStaysAuthenticatedDuringCooldown(t *testing.T) {
	server, db := newEmbeddedTestServer(t)

	expiresAt := time.Now().Add(8 * time.Hour)
	created, err := db.CreateAccountFull(
		"anthropic-api", "Claude", "oauth", "access-token", "refresh-token", "", &expiresAt, nil,
	)
	if err != nil {
		t.Fatalf("create account: %v", err)
	}

	status, err := server.embeddedAuthStatus()
	if err != nil {
		t.Fatalf("auth status: %v", err)
	}
	if !status.Authenticated {
		t.Fatal("account was not reported as authenticated before cooldown")
	}

	if err := db.SetAccountCooldown(created.ID, time.Now().Add(2*time.Minute), 1, "rate limited"); err != nil {
		t.Fatalf("set cooldown: %v", err)
	}

	status, err = server.embeddedAuthStatus()
	if err != nil {
		t.Fatalf("auth status during cooldown: %v", err)
	}
	if !status.Authenticated {
		t.Fatal("a cooling-down account was reported as signed out")
	}
	if status.AccountID != created.ID {
		t.Errorf("AccountID = %q, want %q", status.AccountID, created.ID)
	}
}

// With no account at all the status must still report signed out.
func TestEmbeddedAuthStatusReportsSignedOutWithoutAccount(t *testing.T) {
	server, _ := newEmbeddedTestServer(t)

	status, err := server.embeddedAuthStatus()
	if err != nil {
		t.Fatalf("auth status: %v", err)
	}
	if status.Authenticated {
		t.Fatal("status reported authenticated with no account configured")
	}
}

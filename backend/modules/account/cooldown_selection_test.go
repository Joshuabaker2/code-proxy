package account

import (
	"errors"
	"path/filepath"
	"testing"
	"time"

	"code-proxy/modules/database"
)

func newCooldownTestDB(t *testing.T) *database.DB {
	t.Helper()
	db, err := database.Open(filepath.Join(t.TempDir(), "data.db"))
	if err != nil {
		t.Fatalf("open database: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })
	return db
}

func createAPIKeyAccount(t *testing.T, db *database.DB, label string) *database.Account {
	t.Helper()
	created, err := db.CreateAccountFull("anthropic-api", label, "api_key", "", "", "sk-test", nil, nil)
	if err != nil {
		t.Fatalf("create account: %v", err)
	}
	return created
}

// The regression this whole change exists for: with the only account in
// cooldown, Select used to return (nil, nil), which the provider reported as
// "Anthropic API requires a configured account" — a bogus authentication
// failure for what is really a transient, self-healing condition.
func TestSelectReportsCooldownRatherThanNoAccount(t *testing.T) {
	db := newCooldownTestDB(t)
	created := createAPIKeyAccount(t, db, "Claude")

	until := time.Now().Add(30 * time.Second)
	if err := db.SetAccountCooldown(created.ID, until, 1, "upstream HTTP 529"); err != nil {
		t.Fatalf("set cooldown: %v", err)
	}

	manager := NewManager(db)
	selected, err := manager.Select("anthropic-api", "claude-sonnet-5")
	if selected != nil {
		t.Fatalf("selected = %v, want nil", selected)
	}

	var coolingDown *CoolingDownError
	if !errors.As(err, &coolingDown) {
		t.Fatalf("err = %v, want *CoolingDownError", err)
	}
	if coolingDown.ProviderType != "anthropic-api" {
		t.Errorf("ProviderType = %q, want anthropic-api", coolingDown.ProviderType)
	}
	if wait := coolingDown.RetryAfter(); wait <= 0 || wait > 30*time.Second {
		t.Errorf("RetryAfter() = %s, want a positive value within 30s", wait)
	}
}

// The genuine "not configured" case must stay distinguishable: no accounts at
// all still means the provider is allowed to run unauthenticated.
func TestSelectReturnsNilWhenNoAccountsConfigured(t *testing.T) {
	db := newCooldownTestDB(t)

	manager := NewManager(db)
	selected, err := manager.Select("anthropic-api", "claude-sonnet-5")
	if err != nil {
		t.Fatalf("err = %v, want nil", err)
	}
	if selected != nil {
		t.Fatalf("selected = %v, want nil", selected)
	}
}

// An inactive account is not a cooldown, so it must not be reported as one.
func TestSelectReturnsNilWhenOnlyAccountIsInactive(t *testing.T) {
	db := newCooldownTestDB(t)
	created := createAPIKeyAccount(t, db, "Claude")
	if err := db.UpdateAccount(created.ID, "Claude", false, 0); err != nil {
		t.Fatalf("deactivate account: %v", err)
	}

	manager := NewManager(db)
	selected, err := manager.Select("anthropic-api", "claude-sonnet-5")
	if err != nil {
		t.Fatalf("err = %v, want nil", err)
	}
	if selected != nil {
		t.Fatalf("selected = %v, want nil", selected)
	}
}

// Upstream health problems say nothing about the account, so they must not
// make it unselectable — that is what stranded single-account setups.
func TestReportErrorDoesNotCooldownForUpstreamFailures(t *testing.T) {
	for _, status := range []int{500, 502, 503, 504, 529, 400, 404} {
		db := newCooldownTestDB(t)
		created := createAPIKeyAccount(t, db, "Claude")
		manager := NewManager(db)

		manager.ReportError(created.ID, "claude-sonnet-5", status, 0, "upstream boom")

		selected, err := manager.Select("anthropic-api", "claude-sonnet-5")
		if err != nil {
			t.Fatalf("status %d: Select err = %v, want nil", status, err)
		}
		if selected == nil {
			t.Fatalf("status %d: account was made unselectable", status)
		}

		stored, err := db.GetAccount(created.ID)
		if err != nil {
			t.Fatalf("status %d: reload account: %v", status, err)
		}
		if stored.LastError != "upstream boom" {
			t.Errorf("status %d: LastError = %q, want the failure to still be recorded", status, stored.LastError)
		}
	}
}

// Account-scoped failures still take the account out of rotation.
func TestReportErrorCooldownsForAccountScopedFailures(t *testing.T) {
	for _, status := range []int{401, 402, 403, 429} {
		db := newCooldownTestDB(t)
		created := createAPIKeyAccount(t, db, "Claude")
		manager := NewManager(db)

		manager.ReportError(created.ID, "claude-sonnet-5", status, 0, "nope")

		var coolingDown *CoolingDownError
		if _, err := manager.Select("anthropic-api", "claude-sonnet-5"); !errors.As(err, &coolingDown) {
			t.Fatalf("status %d: err = %v, want *CoolingDownError", status, err)
		}
	}
}

// A Retry-After longer than our own guess wins, so we stop hammering a
// provider that told us exactly how long to wait.
func TestReportErrorHonoursRetryAfter(t *testing.T) {
	db := newCooldownTestDB(t)
	created := createAPIKeyAccount(t, db, "Claude")
	manager := NewManager(db)

	manager.ReportError(created.ID, "claude-sonnet-5", 429, 90*time.Second, "rate limited")

	stored, err := db.GetAccount(created.ID)
	if err != nil {
		t.Fatalf("reload account: %v", err)
	}
	if stored.CooldownUntil == nil {
		t.Fatal("CooldownUntil = nil, want a cooldown")
	}
	if wait := time.Until(*stored.CooldownUntil); wait < 80*time.Second {
		t.Errorf("cooldown = %s, want the 90s Retry-After to be honoured", wait)
	}
}

// Without decay the backoff level only ever grows between successes, so a
// long-lived row ends up permanently pinned at the exponential cap.
func TestBackoffLevelDecaysAfterQuietPeriod(t *testing.T) {
	recent := time.Now().Add(-time.Minute)
	if level := decayBackoffLevel(5, &recent, time.Now()); level != 5 {
		t.Errorf("level = %d, want 5 retained within the decay window", level)
	}

	stale := time.Now().Add(-2 * backoffDecayWindow)
	if level := decayBackoffLevel(5, &stale, time.Now()); level != 0 {
		t.Errorf("level = %d, want 0 after the decay window", level)
	}

	if level := decayBackoffLevel(5, nil, time.Now()); level != 0 {
		t.Errorf("level = %d, want 0 when there is no previous cooldown", level)
	}
}

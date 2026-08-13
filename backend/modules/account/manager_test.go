package account

import (
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"code-proxy/modules/database"
	"code-proxy/modules/provider"
)

func TestSelectSerializesProactiveOAuthRefresh(t *testing.T) {
	db, err := database.Open(filepath.Join(t.TempDir(), "data.db"))
	if err != nil {
		t.Fatalf("open database: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })

	expiresAt := time.Now().Add(-time.Minute)
	created, err := db.CreateAccountFull(
		"anthropic-api",
		"Claude",
		"oauth",
		"access-old",
		"refresh-old",
		"",
		&expiresAt,
		nil,
	)
	if err != nil {
		t.Fatalf("create account: %v", err)
	}

	manager := NewManager(db)
	var refreshCalls atomic.Int32
	manager.SetTokenRefresher(func(account *provider.Account) (RefreshedTokens, error) {
		refreshCalls.Add(1)
		if account.RefreshToken != "refresh-old" {
			t.Fatalf("refresh token = %q, want refresh-old", account.RefreshToken)
		}
		return RefreshedTokens{
			AccessToken:  "access-new",
			RefreshToken: "refresh-new",
			ExpiresAt:    time.Now().Add(8 * time.Hour),
		}, nil
	})

	const callers = 8
	start := make(chan struct{})
	results := make(chan *provider.Account, callers)
	errors := make(chan error, callers)
	var wait sync.WaitGroup
	for range callers {
		wait.Add(1)
		go func() {
			defer wait.Done()
			<-start
			selected, selectErr := manager.Select("anthropic-api", "claude-sonnet-5")
			results <- selected
			errors <- selectErr
		}()
	}
	close(start)
	wait.Wait()
	close(results)
	close(errors)

	for selectErr := range errors {
		if selectErr != nil {
			t.Errorf("select account: %v", selectErr)
		}
	}
	for selected := range results {
		if selected == nil || selected.ID != created.ID || selected.AccessToken != "access-new" || selected.RefreshToken != "refresh-new" {
			t.Errorf("selected stale account: %#v", selected)
		}
	}
	if got := refreshCalls.Load(); got != 1 {
		t.Fatalf("refresh calls = %d, want 1", got)
	}
}

package account

import (
	"context"
	"fmt"
	"log"
	"sync"
	"time"

	"code-proxy/modules/database"
	"code-proxy/modules/provider"
)

// Manager manages account selection, cooldown and refresh
type Manager struct {
	db             *database.DB
	mu             sync.RWMutex
	strategy       string         // "fill-first" or "round-robin"
	cursors        map[string]int // providerType -> current index (round-robin)
	tokenRefresher TokenRefresher
	refreshMu      sync.Mutex
}

// RefreshedTokens is the complete rotated credential set returned by an OAuth
// refresh. Refresh tokens are single-use, so access and refresh tokens must be
// persisted together before another request is allowed to refresh the account.
type RefreshedTokens struct {
	AccessToken  string
	RefreshToken string
	ExpiresAt    time.Time
}

type TokenRefresher func(account *provider.Account) (RefreshedTokens, error)

// NewManager creates an AccountManager
func NewManager(db *database.DB) *Manager {
	return &Manager{
		db:       db,
		strategy: "fill-first",
		cursors:  make(map[string]int),
	}
}

// SetStrategy sets the selection strategy
func (m *Manager) SetStrategy(strategy string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.strategy = strategy
}

// SetTokenRefresher installs the provider-specific OAuth refresh operation.
// Manager serializes calls to it and owns persistence of the rotated tokens.
func (m *Manager) SetTokenRefresher(refresher TokenRefresher) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.tokenRefresher = refresher
}

// Select picks the next available account for the provider+model
func (m *Manager) Select(providerType, model string) (*provider.Account, error) {
	if m.db == nil {
		return nil, nil // No database, no accounts
	}

	accounts, err := m.db.GetAvailableAccounts(providerType)
	if err != nil {
		return nil, fmt.Errorf("fetch accounts: %w", err)
	}

	if len(accounts) == 0 {
		// No accounts configured = provider works without auth
		return nil, nil
	}

	m.mu.Lock()

	var selected *database.Account

	switch m.strategy {
	case "round-robin":
		key := providerType
		idx := m.cursors[key] % len(accounts)
		selected = &accounts[idx]
		m.cursors[key] = idx + 1

	default: // fill-first
		selected = &accounts[0]
	}
	m.mu.Unlock()

	account := dbAccountToProvider(selected)
	refreshed, err := m.refreshAccount(account, false)
	if err != nil {
		// A token inside the refresh buffer is still usable until its actual
		// expiry. Keep serving with it, but never send a known-expired token.
		if !account.ExpiresAt.IsZero() && account.ExpiresAt.After(time.Now()) {
			log.Printf("[ACCOUNT] Proactive refresh failed for %s; using token until %s: %v",
				account.ID[:8], account.ExpiresAt.Format(time.RFC3339), err)
			return account, nil
		}
		return nil, fmt.Errorf("refresh expired OAuth account: %w", err)
	}
	return refreshed, nil
}

// RefreshAfterAuthFailure refreshes an account rejected by the upstream API.
// If another request already rotated the observed access token, the fresh
// database credential is returned without consuming the refresh token again.
func (m *Manager) RefreshAfterAuthFailure(account *provider.Account) (*provider.Account, error) {
	if account == nil {
		return nil, fmt.Errorf("cannot refresh a missing account")
	}
	return m.refreshAccount(account, true)
}

func (m *Manager) refreshAccount(observed *provider.Account, force bool) (*provider.Account, error) {
	if m.db == nil || observed == nil {
		return observed, nil
	}

	m.refreshMu.Lock()
	defer m.refreshMu.Unlock()

	stored, err := m.db.GetAccount(observed.ID)
	if err != nil {
		return nil, fmt.Errorf("reload account before refresh: %w", err)
	}
	current := dbAccountToProvider(stored)
	if current.AccessToken != observed.AccessToken {
		return current, nil
	}
	if !force && (current.ExpiresAt.IsZero() || current.ExpiresAt.After(time.Now().Add(10*time.Minute))) {
		return current, nil
	}
	if current.AuthMode != "oauth" || current.RefreshToken == "" {
		if force {
			return nil, fmt.Errorf("account has no OAuth refresh token")
		}
		return current, nil
	}

	m.mu.RLock()
	refresher := m.tokenRefresher
	m.mu.RUnlock()
	if refresher == nil {
		return nil, fmt.Errorf("OAuth token refresher is not configured")
	}
	tokens, err := refresher(current)
	if err != nil {
		return nil, err
	}
	if tokens.AccessToken == "" {
		return nil, fmt.Errorf("OAuth refresh returned no access token")
	}
	if tokens.RefreshToken == "" {
		tokens.RefreshToken = current.RefreshToken
	}
	if tokens.ExpiresAt.IsZero() {
		return nil, fmt.Errorf("OAuth refresh returned no expiration")
	}
	if err := m.db.UpdateAccountTokens(
		current.ID,
		tokens.AccessToken,
		tokens.RefreshToken,
		&tokens.ExpiresAt,
	); err != nil {
		return nil, fmt.Errorf("persist refreshed OAuth tokens: %w", err)
	}
	_ = m.db.ClearAccountCooldown(current.ID)
	updated, err := m.db.GetAccount(current.ID)
	if err != nil {
		return nil, fmt.Errorf("reload refreshed account: %w", err)
	}
	log.Printf("[ACCOUNT] Refresh OK: %s (%s)", updated.ID[:8], updated.Label)
	return dbAccountToProvider(updated), nil
}

// ReportSuccess clears cooldown for an account
func (m *Manager) ReportSuccess(accountID, model string) {
	if m.db == nil {
		return
	}
	m.db.ClearAccountCooldown(accountID)
}

// ReportError applies cooldown with exponential backoff
func (m *Manager) ReportError(accountID, model string, httpStatus int, errText string) {
	if m.db == nil {
		return
	}

	acct, err := m.db.GetAccount(accountID)
	if err != nil {
		return
	}

	duration := CooldownForStatus(httpStatus, acct.BackoffLevel)
	until := time.Now().Add(duration)
	newLevel := acct.BackoffLevel + 1

	log.Printf("[ACCOUNT] Cooldown %s: status=%d, duration=%s, level=%d",
		accountID[:8], httpStatus, duration, newLevel)

	m.db.SetAccountCooldown(accountID, until, newLevel, errText)
}

// RefreshLoop checks for expiring tokens and refreshes them in background
func (m *Manager) RefreshLoop(ctx context.Context, interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	// Refresh immediately on startup so a persisted, nearly expired token does
	// not fail requests during the first interval.
	m.refreshExpiring()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			m.refreshExpiring()
		}
	}
}

func (m *Manager) refreshExpiring() {
	if m.db == nil {
		return
	}

	// Fetch accounts expiring in the next 10 minutes
	accounts, err := m.db.GetExpiringAccounts(10 * time.Minute)
	if err != nil {
		log.Printf("[ACCOUNT] Error fetching expiring accounts: %v", err)
		return
	}

	for _, a := range accounts {
		acct := dbAccountToProvider(&a)
		if _, err := m.refreshAccount(acct, false); err != nil {
			log.Printf("[ACCOUNT] Refresh error %s (%s): %v", a.ID[:8], a.Label, err)
		}
	}
}

// dbAccountToProvider converts database.Account to provider.Account
func dbAccountToProvider(a *database.Account) *provider.Account {
	var expiresAt time.Time
	if a.ExpiresAt != nil {
		expiresAt = *a.ExpiresAt
	}

	return &provider.Account{
		ID:           a.ID,
		ProviderType: a.ProviderType,
		Label:        a.Label,
		AuthMode:     a.AuthMode,
		AccessToken:  a.AccessToken,
		RefreshToken: a.RefreshToken,
		APIKey:       a.APIKey,
		ExpiresAt:    expiresAt,
		Metadata:     a.Metadata,
		IsActive:     a.IsActive,
		Priority:     a.Priority,
	}
}

package api

import (
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"code-proxy/modules/account"
	"code-proxy/modules/database"
	"code-proxy/modules/provider"
)

func newEmbeddedTestServer(t *testing.T) (*EmbeddedServer, *database.DB) {
	t.Helper()

	db, err := database.Open(filepath.Join(t.TempDir(), "data.db"))
	if err != nil {
		t.Fatalf("open test database: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })

	registry := provider.NewRegistry()
	registry.Register("anthropic-api", provider.NewAnthropicAPI())
	server, err := NewEmbeddedServer(EmbeddedServerOptions{
		Registry:     registry,
		AccountMgr:   account.NewManager(db),
		DB:           db,
		DefaultModel: "sonnet",
		ControlToken: "test-control-token",
	})
	if err != nil {
		t.Fatalf("create embedded server: %v", err)
	}
	return server, db
}

func embeddedRequest(t *testing.T, server *EmbeddedServer, method, path, token string) *httptest.ResponseRecorder {
	t.Helper()

	request := httptest.NewRequest(method, path, nil)
	if token != "" {
		request.Header.Set("Authorization", "Bearer "+token)
	}
	response := httptest.NewRecorder()
	server.Handler().ServeHTTP(response, request)
	return response
}

func TestEmbeddedServerRequiresTokenExceptForHealth(t *testing.T) {
	server, _ := newEmbeddedTestServer(t)

	health := embeddedRequest(t, server, http.MethodGet, "/health", "")
	if health.Code != http.StatusOK {
		t.Fatalf("health status = %d, want %d", health.Code, http.StatusOK)
	}

	unauthorized := embeddedRequest(t, server, http.MethodGet, "/v1/models", "")
	if unauthorized.Code != http.StatusUnauthorized {
		t.Fatalf("unauthorized status = %d, want %d", unauthorized.Code, http.StatusUnauthorized)
	}

	authorized := embeddedRequest(t, server, http.MethodGet, "/v1/models", "test-control-token")
	if authorized.Code != http.StatusOK {
		t.Fatalf("authorized status = %d, want %d: %s", authorized.Code, http.StatusOK, authorized.Body.String())
	}
}

func TestEmbeddedModelsUseTheClaudeOAuthCatalog(t *testing.T) {
	server, _ := newEmbeddedTestServer(t)
	response := embeddedRequest(t, server, http.MethodGet, "/v1/models", "test-control-token")
	if response.Code != http.StatusOK {
		t.Fatalf("models status = %d, want %d: %s", response.Code, http.StatusOK, response.Body.String())
	}

	var models ModelsResponse
	if err := json.Unmarshal(response.Body.Bytes(), &models); err != nil {
		t.Fatalf("decode models: %v", err)
	}
	want := []string{"claude-opus-5", "claude-sonnet-5", "claude-fable-5"}
	if len(models.Data) != len(want) {
		t.Fatalf("models = %#v, want %v", models.Data, want)
	}
	for index, modelID := range want {
		if models.Data[index].ID != modelID {
			t.Errorf("models[%d].ID = %q, want %q", index, models.Data[index].ID, modelID)
		}
	}
}

func TestMergeClaudeModelCatalogsKeepsStableAliasesFirst(t *testing.T) {
	stable := provider.ClaudeOAuthModels()
	discovered := []provider.Model{
		{ID: "cc/claude-opus-4-6", Name: "Claude Opus 4.6"},
		{ID: "cc/claude-sonnet-4-6", Name: "Claude Sonnet 4.6"},
		{ID: "cc/claude-opus-5", Name: "duplicate"},
	}

	got := mergeClaudeModelCatalogs(stable, discovered)
	want := []string{
		"cc/claude-opus-5",
		"cc/claude-sonnet-5",
		"cc/claude-fable-5",
		"cc/claude-opus-4-6",
		"cc/claude-sonnet-4-6",
	}
	if len(got) != len(want) {
		t.Fatalf("merged models = %#v, want IDs %v", got, want)
	}
	for index, id := range want {
		if got[index].ID != id {
			t.Errorf("models[%d].ID = %q, want %q", index, got[index].ID, id)
		}
	}
}

func TestEmbeddedServerDoesNotExposeDashboardRoutes(t *testing.T) {
	server, _ := newEmbeddedTestServer(t)

	for _, path := range []string{"/", "/api/accounts", "/api/tunnel/status", "/dashboard"} {
		response := embeddedRequest(t, server, http.MethodGet, path, "test-control-token")
		if response.Code != http.StatusNotFound {
			t.Errorf("%s status = %d, want %d", path, response.Code, http.StatusNotFound)
		}
	}
}

func TestEmbeddedAuthStatusIsSanitizedAndLogoutDeletesAccount(t *testing.T) {
	server, db := newEmbeddedTestServer(t)

	created, err := db.CreateAccountFull(
		"anthropic-api",
		"josh@example.com",
		"oauth",
		"access-secret",
		"refresh-secret",
		"",
		nil,
		map[string]string{"id_token": "identity-secret"},
	)
	if err != nil {
		t.Fatalf("create account: %v", err)
	}

	statusResponse := embeddedRequest(t, server, http.MethodGet, "/sidecar/auth/status", "test-control-token")
	if statusResponse.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d: %s", statusResponse.Code, http.StatusOK, statusResponse.Body.String())
	}
	var status EmbeddedAuthStatus
	if err := json.Unmarshal(statusResponse.Body.Bytes(), &status); err != nil {
		t.Fatalf("decode status: %v", err)
	}
	if !status.Authenticated || status.AccountID != created.ID || status.AccountLabel != "josh@example.com" {
		t.Fatalf("unexpected auth status: %#v", status)
	}
	for _, secret := range []string{"access-secret", "refresh-secret", "identity-secret"} {
		if body := statusResponse.Body.String(); strings.Contains(body, secret) {
			t.Fatalf("status leaked %q: %s", secret, body)
		}
	}

	logout := embeddedRequest(t, server, http.MethodDelete, "/sidecar/auth", "test-control-token")
	if logout.Code != http.StatusOK {
		t.Fatalf("logout status = %d, want %d: %s", logout.Code, http.StatusOK, logout.Body.String())
	}
	accounts, err := db.ListAccounts("anthropic-api")
	if err != nil {
		t.Fatalf("list accounts: %v", err)
	}
	if len(accounts) != 0 {
		t.Fatalf("accounts remained after logout: %#v", accounts)
	}
}

func TestEmbeddedAuthStatusRequiresARefreshableSession(t *testing.T) {
	server, db := newEmbeddedTestServer(t)
	expiredAt := time.Now().Add(-time.Hour)
	if _, err := db.CreateAccountFull(
		"anthropic-api",
		"josh@example.com",
		"oauth",
		"expired-access",
		"revoked-refresh",
		"",
		&expiredAt,
		nil,
	); err != nil {
		t.Fatalf("create expired account: %v", err)
	}
	server.accountMgr.SetTokenRefresher(func(*provider.Account) (account.RefreshedTokens, error) {
		return account.RefreshedTokens{}, errors.New("invalid_grant: refresh token revoked")
	})

	response := embeddedRequest(t, server, http.MethodGet, "/sidecar/auth/status", "test-control-token")
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d: %s", response.Code, http.StatusOK, response.Body.String())
	}
	var status EmbeddedAuthStatus
	if err := json.Unmarshal(response.Body.Bytes(), &status); err != nil {
		t.Fatalf("decode status: %v", err)
	}
	if status.Authenticated {
		t.Fatalf("expired, unrefreshable account reported authenticated: %#v", status)
	}
}

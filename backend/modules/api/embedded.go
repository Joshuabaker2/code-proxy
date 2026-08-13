package api

import (
	"context"
	"crypto/subtle"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"code-proxy/modules/account"
	"code-proxy/modules/auth"
	"code-proxy/modules/database"
	"code-proxy/modules/provider"
)

// EmbeddedServer is the deliberately small HTTP surface exposed by the
// Tauri-managed code-proxy sidecar. It shares the provider and OAuth
// implementation with the full dashboard server without exposing dashboard,
// tunnel, Zed, or CLI-provider routes.
type EmbeddedServer struct {
	mux          *http.ServeMux
	registry     *provider.Registry
	accountMgr   *account.Manager
	db           *database.DB
	defaultModel string
	controlToken string
}

type EmbeddedServerOptions struct {
	Registry     *provider.Registry
	AccountMgr   *account.Manager
	DB           *database.DB
	DefaultModel string
	ControlToken string
}

type EmbeddedAuthStatus struct {
	Authenticated bool       `json:"authenticated"`
	AccountID     string     `json:"account_id,omitempty"`
	AccountLabel  string     `json:"account_label,omitempty"`
	ExpiresAt     *time.Time `json:"expires_at,omitempty"`
}

func NewEmbeddedServer(opts EmbeddedServerOptions) (*EmbeddedServer, error) {
	if opts.Registry == nil {
		return nil, fmt.Errorf("embedded server requires a provider registry")
	}
	if opts.AccountMgr == nil {
		return nil, fmt.Errorf("embedded server requires an account manager")
	}
	if opts.DB == nil {
		return nil, fmt.Errorf("embedded server requires a database")
	}
	if opts.ControlToken == "" {
		return nil, fmt.Errorf("embedded server requires a control token")
	}

	server := &EmbeddedServer{
		mux:          http.NewServeMux(),
		registry:     opts.Registry,
		accountMgr:   opts.AccountMgr,
		db:           opts.DB,
		defaultModel: opts.DefaultModel,
		controlToken: opts.ControlToken,
	}
	server.registerRoutes()
	return server, nil
}

func (s *EmbeddedServer) Handler() http.Handler {
	return embeddedTokenAuth(s.controlToken, s.mux)
}

func (s *EmbeddedServer) registerRoutes() {
	s.mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			writeError(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	})

	s.mux.HandleFunc("/sidecar/auth/status", s.handleEmbeddedAuthStatus)
	s.mux.HandleFunc("/sidecar/auth/start", s.handleEmbeddedAuthStart)
	s.mux.HandleFunc("/sidecar/auth/complete", s.handleEmbeddedAuthComplete)
	s.mux.HandleFunc("/sidecar/auth", s.handleEmbeddedAuthDelete)
	s.mux.HandleFunc("/v1/models", s.handleEmbeddedModels)
	s.mux.HandleFunc("/v1/chat/completions", handleChat(s.registry, s.accountMgr, s.defaultModel, s.db))
}

func embeddedTokenAuth(controlToken string, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			next.ServeHTTP(w, r)
			return
		}

		token := extractBearerToken(r)
		valid := len(token) == len(controlToken) &&
			subtle.ConstantTimeCompare([]byte(token), []byte(controlToken)) == 1
		if !valid {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusUnauthorized)
			json.NewEncoder(w).Encode(map[string]any{
				"error": map[string]string{
					"message": "Missing or invalid sidecar token",
					"type":    "authentication_error",
				},
			})
			return
		}

		next.ServeHTTP(w, r)
	})
}

func (s *EmbeddedServer) handleEmbeddedAuthStatus(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	status, err := s.embeddedAuthStatus()
	if err != nil {
		writeError(w, err.Error(), http.StatusInternalServerError)
		return
	}
	writeEmbeddedJSON(w, http.StatusOK, status)
}

func (s *EmbeddedServer) handleEmbeddedAuthStart(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	flowID, authURL, err := oauthFlows.StartFlow("claude")
	if err != nil {
		writeError(w, err.Error(), http.StatusInternalServerError)
		return
	}

	writeEmbeddedJSON(w, http.StatusOK, map[string]any{
		"flow_id":                  flowID,
		"auth_url":                 authURL,
		"supports_manual_callback": true,
	})
}

func (s *EmbeddedServer) handleEmbeddedAuthComplete(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var request struct {
		FlowID      string `json:"flow_id"`
		CallbackURL string `json:"callback_url"`
	}
	if err := readJSON(r, &request); err != nil {
		writeError(w, "Invalid JSON", http.StatusBadRequest)
		return
	}
	if request.FlowID == "" {
		writeError(w, "flow_id is required", http.StatusBadRequest)
		return
	}

	var (
		tokens *auth.OAuthTokens
		err    error
	)
	if request.CallbackURL != "" {
		tokens, err = oauthFlows.SubmitCallback(request.FlowID, request.CallbackURL)
	} else {
		tokens, err = oauthFlows.WaitForCallback(request.FlowID, 5*time.Minute)
	}
	if err != nil {
		writeError(w, err.Error(), http.StatusBadRequest)
		return
	}

	label := "Claude"
	if email, ok := tokens.RawResponse["email"].(string); ok && email != "" {
		label = email
	}
	metadata := map[string]string{}
	if tokens.IDToken != "" {
		metadata["id_token"] = tokens.IDToken
	}

	existing, err := s.db.ListAccounts("anthropic-api")
	if err != nil {
		writeError(w, "Failed to list existing accounts", http.StatusInternalServerError)
		return
	}

	expiresAt := tokens.ExpiresAt
	created, err := s.db.CreateAccountFull(
		"anthropic-api",
		label,
		"oauth",
		tokens.AccessToken,
		tokens.RefreshToken,
		"",
		&expiresAt,
		metadata,
	)
	if err != nil {
		writeError(w, "Failed to save account", http.StatusInternalServerError)
		return
	}

	for _, oldAccount := range existing {
		if oldAccount.ID != created.ID {
			_ = s.db.DeleteAccount(oldAccount.ID)
		}
	}

	writeEmbeddedJSON(w, http.StatusCreated, EmbeddedAuthStatus{
		Authenticated: true,
		AccountID:     created.ID,
		AccountLabel:  created.Label,
		ExpiresAt:     created.ExpiresAt,
	})
}

func (s *EmbeddedServer) handleEmbeddedAuthDelete(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodDelete {
		writeError(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	accounts, err := s.db.ListAccounts("anthropic-api")
	if err != nil {
		writeError(w, err.Error(), http.StatusInternalServerError)
		return
	}
	for _, storedAccount := range accounts {
		if err := s.db.DeleteAccount(storedAccount.ID); err != nil {
			writeError(w, err.Error(), http.StatusInternalServerError)
			return
		}
	}

	writeEmbeddedJSON(w, http.StatusOK, EmbeddedAuthStatus{})
}

func (s *EmbeddedServer) handleEmbeddedModels(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Keep the stable Claude 5 aliases first. Live discovery contains useful
	// dated/historical IDs too, but it must extend this product-facing catalog
	// rather than replace it; Zed and the embedded Tauri client both rely on
	// these aliases remaining selectable.
	catalog := provider.ClaudeOAuthModels()
	if accounts, err := s.db.GetAvailableAccounts("anthropic-api"); err == nil {
		for _, storedAccount := range accounts {
			if storedAccount.AuthMode != "oauth" || storedAccount.AccessToken == "" {
				continue
			}
			discoveryContext, cancelDiscovery := context.WithTimeout(r.Context(), 15*time.Second)
			discovered, discoverErr := provider.DiscoverClaudeOAuthModels(
				discoveryContext,
				storedAccount.AccessToken,
			)
			cancelDiscovery()
			if discoverErr == nil {
				catalog = mergeClaudeModelCatalogs(catalog, discovered)
				break
			}
		}
	}

	models := make([]ModelItem, 0, len(catalog))
	for _, model := range catalog {
		models = append(models, ModelItem{
			ID:      strings.TrimPrefix(model.ID, "cc/"),
			Object:  "model",
			OwnedBy: model.OwnedBy,
		})
	}
	writeEmbeddedJSON(w, http.StatusOK, ModelsResponse{Object: "list", Data: models})
}

func mergeClaudeModelCatalogs(catalogs ...[]provider.Model) []provider.Model {
	seen := make(map[string]struct{})
	var merged []provider.Model
	for _, catalog := range catalogs {
		for _, model := range catalog {
			id := strings.TrimPrefix(model.ID, "cc/")
			if _, exists := seen[id]; exists {
				continue
			}
			seen[id] = struct{}{}
			merged = append(merged, model)
		}
	}
	return merged
}

func (s *EmbeddedServer) embeddedAuthStatus() (EmbeddedAuthStatus, error) {
	selected, err := s.accountMgr.Select("anthropic-api", "")
	if err != nil {
		// A stored row is not an authenticated session if its expired token can
		// no longer be refreshed. Report signed-out so the client can reconnect.
		return EmbeddedAuthStatus{}, nil
	}
	if selected == nil || !selected.IsActive || selected.AuthMode != "oauth" || selected.AccessToken == "" {
		return EmbeddedAuthStatus{}, nil
	}
	var expiresAt *time.Time
	if !selected.ExpiresAt.IsZero() {
		expiresAt = &selected.ExpiresAt
	}
	return EmbeddedAuthStatus{
		Authenticated: true,
		AccountID:     selected.ID,
		AccountLabel:  selected.Label,
		ExpiresAt:     expiresAt,
	}, nil
}

func writeEmbeddedJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(value)
}

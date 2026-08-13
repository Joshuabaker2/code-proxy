package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"code-proxy/embed"
	"code-proxy/modules/account"
	"code-proxy/modules/api"
	"code-proxy/modules/auth"
	"code-proxy/modules/config"
	"code-proxy/modules/database"
	"code-proxy/modules/provider"
	"code-proxy/modules/tunnel"
	"code-proxy/modules/zed"
)

func main() {
	cfg := config.Load()

	// API keys from env (comma-separated)
	var envApiKeys []string
	if keys := os.Getenv("PROXY_API_KEY"); keys != "" {
		for _, k := range strings.Split(keys, ",") {
			k = strings.TrimSpace(k)
			if k != "" {
				envApiKeys = append(envApiKeys, k)
			}
		}
	}

	// Open SQLite database
	db, err := database.Open(cfg.DBPath)
	if err != nil {
		log.Printf("[MAIN] WARNING: Failed to open database: %v (running without persistence)", err)
	} else {
		defer db.Close()
		log.Printf("[MAIN] Database: %s", cfg.DBPath)
	}

	// Provider registry
	registry := provider.NewRegistry()

	// Register Claude CLI provider
	if cfg.UseACP {
		log.Printf("[MAIN] Mode: ACP (command: %s)", cfg.ACPCommand)
		registry.Register("claude-cli", provider.NewClaudeACP(cfg.WorkDir, cfg.ACPCommand, cfg.ACPArgs))
	} else {
		log.Println("[MAIN] Mode: CLI (exec claude)")
		registry.Register("claude-cli", provider.NewClaude(cfg.WorkDir))
	}

	// Register API providers (accounts added via dashboard)
	registry.Register("anthropic-api", provider.NewAnthropicAPI())
	registry.Register("openai-api", provider.NewOpenAIAPI())
	registry.Register("gemini-api", provider.NewGeminiAPI())
	registry.Register("generic-openai", provider.NewGenericOpenAI())

	// Register additional CLI providers
	registry.Register("codex-cli", provider.NewCodex(cfg.WorkDir))
	registry.Register("gemini-cli", provider.NewGeminiCLI(cfg.WorkDir))

	// Account manager
	acctMgr := account.NewManager(db)
	acctMgr.SetTokenRefresher(func(acct *provider.Account) (account.RefreshedTokens, error) {
		if acct.RefreshToken == "" {
			return account.RefreshedTokens{}, fmt.Errorf("OAuth account has no refresh token")
		}
		oauthProvider := auth.ProviderNameForType(acct.ProviderType)
		oauthConfig, ok := auth.GetConfig(oauthProvider)
		if !ok {
			return account.RefreshedTokens{}, fmt.Errorf("OAuth config not found for %s", acct.ProviderType)
		}
		tokens, err := auth.RefreshTokens(oauthConfig, acct.RefreshToken)
		if err != nil {
			return account.RefreshedTokens{}, err
		}
		return account.RefreshedTokens{
			AccessToken:  tokens.AccessToken,
			RefreshToken: tokens.RefreshToken,
			ExpiresAt:    tokens.ExpiresAt,
		}, nil
	})

	// Background token refresh (every 5 minutes)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go acctMgr.RefreshLoop(ctx, 5*time.Minute)

	if os.Getenv("ZED_SYNC_MODELS") == "true" {
		settingsPath := os.Getenv("ZED_SETTINGS_PATH")
		if settingsPath == "" {
			settingsPath, err = zed.DefaultSettingsPath()
		}
		if err != nil {
			log.Printf("[ZED] Could not resolve settings path: %v", err)
		} else {
			syncZedModels(ctx, settingsPath, db)
			go zedModelSyncLoop(ctx, settingsPath, db, 6*time.Hour)
		}
	}

	// Tunnel manager
	tunnelMgr := tunnel.NewManager(cfg.Port, cfg.DataDir, func(url string) {
		if db != nil {
			db.SetSetting("tunnel_url", url)
		}
	}, func(enabled bool) {
		if db != nil {
			if enabled {
				db.SetSetting("tunnel_enabled", "true")
			} else {
				db.SetSetting("tunnel_enabled", "false")
			}
		}
	})

	// Create server
	frontendFS := embed.FS()
	server := api.NewServer(api.ServerOptions{
		Registry:     registry,
		AccountMgr:   acctMgr,
		DefaultModel: cfg.DefaultModel,
		EnvApiKeys:   envApiKeys,
		DB:           db,
		Tunnel:       tunnelMgr,
		FrontendFS:   frontendFS,
	})

	if len(envApiKeys) == 0 && db != nil {
		keys, _ := db.ListApiKeys()
		if len(keys) == 0 {
			log.Println("[MAIN] WARNING: No API keys configured. API is open to all requests.")
			log.Println("[MAIN] Create a key via dashboard or set PROXY_API_KEY env var.")
		}
	}

	// Auto-start tunnel if it was enabled before shutdown
	if db != nil {
		settings := db.GetSettings()
		tunnelMgr.AutoStart(settings.TunnelEnabled, settings.TunnelToken)
	}

	log.Printf("[MAIN] Code Proxy listening on :%s", cfg.Port)
	log.Printf("[MAIN] Default model: %s, Providers: %v", cfg.DefaultModel, registry.ListProviders())
	log.Println("[MAIN] Endpoints:")
	log.Println("[MAIN]   POST /v1/chat/completions  (OpenAI-compatible)")
	log.Println("[MAIN]   GET  /v1/models")
	log.Println("[MAIN]   GET  /health")
	if frontendFS != nil {
		log.Printf("[MAIN]   GET  /                      (Dashboard)")
	}
	log.Println("[MAIN]   GET  /api/keys, /api/providers, /api/accounts, /api/settings, /api/logs, /api/stats")
	log.Println("[MAIN]   POST /api/tunnel/enable, /api/tunnel/disable, /api/tunnel/status")

	if err := http.ListenAndServe("127.0.0.1:"+cfg.Port, server.Handler()); err != nil {
		log.Fatal(err)
	}
}

func zedModelSyncLoop(ctx context.Context, settingsPath string, db *database.DB, interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			syncZedModels(ctx, settingsPath, db)
		}
	}
}

func syncZedModels(ctx context.Context, settingsPath string, db *database.DB) {
	models := provider.ClaudeOAuthModels()
	if db != nil {
		accounts, err := db.GetAvailableAccounts("anthropic-api")
		if err != nil {
			log.Printf("[ZED] Could not load Claude accounts for model discovery: %v", err)
		} else {
			for _, acct := range accounts {
				if acct.AuthMode != "oauth" || acct.AccessToken == "" {
					continue
				}
				discoveryCtx, cancelDiscovery := context.WithTimeout(ctx, 15*time.Second)
				discovered, discoverErr := provider.DiscoverClaudeOAuthModels(discoveryCtx, acct.AccessToken)
				cancelDiscovery()
				if discoverErr != nil {
					log.Printf("[ZED] Live Claude model discovery failed for one account: %v", discoverErr)
					continue
				}
				models = discovered
				log.Printf("[ZED] Discovered %d Claude models from Anthropic", len(models))
				break
			}
		}
	}

	changed, err := zed.SyncCodeProxyModels(settingsPath, zed.CodeProxyModels(models))
	if err != nil {
		log.Printf("[ZED] Model sync skipped: %v", err)
	} else if changed {
		log.Printf("[ZED] Updated Code Proxy models in %s", settingsPath)
	} else {
		log.Printf("[ZED] Code Proxy models already current")
	}
}

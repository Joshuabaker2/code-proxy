package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"code-proxy/modules/account"
	"code-proxy/modules/api"
	"code-proxy/modules/auth"
	"code-proxy/modules/database"
	"code-proxy/modules/provider"
)

var version = "dev"

const (
	sidecarLockWait       = 10 * time.Second
	parentCheckInterval   = time.Second
	sidecarLockRetryDelay = 50 * time.Millisecond
)

type readiness struct {
	Type    string `json:"type"`
	Port    int    `json:"port"`
	Version string `json:"version"`
}

func main() {
	log.SetOutput(os.Stderr)
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)

	if err := run(); err != nil {
		log.Printf("[SIDECAR] Fatal: %v", err)
		os.Exit(1)
	}
}

func run() error {
	controlToken := os.Getenv("CODE_PROXY_CONTROL_TOKEN")
	if controlToken == "" {
		return fmt.Errorf("CODE_PROXY_CONTROL_TOKEN is required")
	}

	dataDir := os.Getenv("DATA_DIR")
	if dataDir == "" {
		return fmt.Errorf("DATA_DIR is required")
	}
	if err := os.MkdirAll(dataDir, 0o700); err != nil {
		return fmt.Errorf("create data directory: %w", err)
	}
	if err := os.Chmod(dataDir, 0o700); err != nil {
		return fmt.Errorf("secure data directory: %w", err)
	}
	if os.Getppid() <= 1 {
		return fmt.Errorf("code-proxy sidecar requires a live parent process")
	}
	lock, err := acquireSidecarLock(filepath.Join(dataDir, ".sidecar.lock"), sidecarLockWait)
	if err != nil {
		return err
	}
	defer lock.close()

	dbPath := filepath.Join(dataDir, "data.db")
	db, err := database.Open(dbPath)
	if err != nil {
		return fmt.Errorf("open database: %w", err)
	}
	defer db.Close()
	if err := os.Chmod(dbPath, 0o600); err != nil {
		return fmt.Errorf("secure database: %w", err)
	}

	registry := provider.NewRegistry()
	registry.Register("anthropic-api", provider.NewAnthropicAPI())
	accountManager := account.NewManager(db)
	accountManager.SetTokenRefresher(func(storedAccount *provider.Account) (account.RefreshedTokens, error) {
		if storedAccount.RefreshToken == "" {
			return account.RefreshedTokens{}, fmt.Errorf("OAuth account has no refresh token")
		}
		config, ok := auth.GetConfig(auth.ProviderNameForType(storedAccount.ProviderType))
		if !ok {
			return account.RefreshedTokens{}, fmt.Errorf("OAuth config not found for %s", storedAccount.ProviderType)
		}
		tokens, err := auth.RefreshTokens(config, storedAccount.RefreshToken)
		if err != nil {
			return account.RefreshedTokens{}, err
		}
		return account.RefreshedTokens{
			AccessToken:  tokens.AccessToken,
			RefreshToken: tokens.RefreshToken,
			ExpiresAt:    tokens.ExpiresAt,
		}, nil
	})

	defaultModel := os.Getenv("CLAUDE_MODEL")
	if defaultModel == "" {
		defaultModel = "sonnet"
	}

	embeddedServer, err := api.NewEmbeddedServer(api.EmbeddedServerOptions{
		Registry:     registry,
		AccountMgr:   accountManager,
		DB:           db,
		DefaultModel: defaultModel,
		ControlToken: controlToken,
	})
	if err != nil {
		return err
	}

	signalContext, stopSignals := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stopSignals()
	ctx, cancel := context.WithCancel(signalContext)
	defer cancel()
	go monitorParent(ctx, os.Getppid(), cancel)
	go accountManager.RefreshLoop(ctx, 5*time.Minute)

	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return fmt.Errorf("listen: %w", err)
	}
	defer listener.Close()

	tcpAddress, ok := listener.Addr().(*net.TCPAddr)
	if !ok {
		return fmt.Errorf("unexpected listener address %T", listener.Addr())
	}
	if err := json.NewEncoder(os.Stdout).Encode(readiness{
		Type:    "ready",
		Port:    tcpAddress.Port,
		Version: version,
	}); err != nil {
		return fmt.Errorf("write readiness: %w", err)
	}

	httpServer := &http.Server{
		Handler:           embeddedServer.Handler(),
		ReadHeaderTimeout: 10 * time.Second,
		IdleTimeout:       90 * time.Second,
	}
	serveErrors := make(chan error, 1)
	go func() {
		serveErrors <- httpServer.Serve(listener)
	}()

	select {
	case <-ctx.Done():
		shutdownContext, shutdownCancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer shutdownCancel()
		if err := httpServer.Shutdown(shutdownContext); err != nil {
			return fmt.Errorf("shutdown: %w", err)
		}
		return nil
	case err := <-serveErrors:
		if errors.Is(err, http.ErrServerClosed) {
			return nil
		}
		return fmt.Errorf("serve: %w", err)
	}
}

type sidecarLock struct {
	file *os.File
}

func acquireSidecarLock(path string, wait time.Duration) (*sidecarLock, error) {
	file, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return nil, fmt.Errorf("open sidecar lock: %w", err)
	}
	deadline := time.Now().Add(wait)
	for {
		err = syscall.Flock(int(file.Fd()), syscall.LOCK_EX|syscall.LOCK_NB)
		if err == nil {
			return &sidecarLock{file: file}, nil
		}
		if !errors.Is(err, syscall.EWOULDBLOCK) && !errors.Is(err, syscall.EAGAIN) {
			_ = file.Close()
			return nil, fmt.Errorf("lock sidecar data directory: %w", err)
		}
		if time.Now().After(deadline) {
			_ = file.Close()
			return nil, fmt.Errorf("another code-proxy sidecar still owns %s", filepath.Dir(path))
		}
		time.Sleep(sidecarLockRetryDelay)
	}
}

func (lock *sidecarLock) close() {
	if lock == nil || lock.file == nil {
		return
	}
	_ = syscall.Flock(int(lock.file.Fd()), syscall.LOCK_UN)
	_ = lock.file.Close()
}

func monitorParent(ctx context.Context, parentPID int, cancel context.CancelFunc) {
	ticker := time.NewTicker(parentCheckInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			if os.Getppid() != parentPID {
				log.Printf("[SIDECAR] Parent process exited; shutting down")
				cancel()
				return
			}
		}
	}
}

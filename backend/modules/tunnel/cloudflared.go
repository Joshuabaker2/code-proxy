package tunnel

import (
	"archive/tar"
	"bufio"
	"compress/gzip"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"sync"
	"time"
)

type Manager struct {
	mu        sync.Mutex
	cmd       *exec.Cmd
	url       string
	running   bool
	localPort string
	binDir    string
	token     string // Cloudflare tunnel token for persistent tunnels
	onURL     func(string)
	onState   func(enabled bool) // persists enabled/disabled state to DB
}

func NewManager(localPort string, dataDir string, onURL func(string), onState func(bool)) *Manager {
	return &Manager{
		localPort: localPort,
		binDir:    filepath.Join(dataDir, "bin"),
		onURL:     onURL,
		onState:   onState,
		token:     os.Getenv("TUNNEL_TOKEN"),
	}
}

func (m *Manager) IsRunning() bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.running
}

func (m *Manager) URL() string {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.url
}

func (m *Manager) Token() string {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.token
}

func (m *Manager) SetToken(token string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.token = token
}

func (m *Manager) Enable() error {
	m.mu.Lock()
	if m.running {
		m.mu.Unlock()
		return nil
	}
	m.mu.Unlock()

	binPath, err := m.ensureBinary()
	if err != nil {
		return fmt.Errorf("cloudflared binary: %w", err)
	}

	m.mu.Lock()
	token := m.token
	m.mu.Unlock()

	var cmd *exec.Cmd
	if token != "" {
		// Named tunnel with token (persistent URL)
		log.Println("[TUNNEL] Starting named tunnel with token")
		cmd = exec.Command(binPath, "tunnel", "run", "--token", token)
	} else {
		// Quick tunnel (random URL, changes on restart)
		log.Println("[TUNNEL] Starting quick tunnel")
		cmd = exec.Command(binPath, "tunnel", "--url", "http://localhost:"+m.localPort)
	}

	stderr, err := cmd.StderrPipe()
	if err != nil {
		return fmt.Errorf("stderr pipe: %w", err)
	}

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("start cloudflared: %w", err)
	}

	m.mu.Lock()
	m.cmd = cmd
	m.running = true
	m.mu.Unlock()

	// Persist enabled state so tunnel auto-starts on server restart
	if m.onState != nil {
		m.onState(true)
	}

	// Parse URL from stderr
	urlCh := make(chan string, 1)
	go func() {
		scanner := bufio.NewScanner(stderr)
		// Quick tunnel: https://xxx.trycloudflare.com
		// Named tunnel: URL comes from Cloudflare config, look for "Registered tunnel connection"
		urlRegex := regexp.MustCompile(`https://[a-zA-Z0-9.-]+\.(trycloudflare\.com|cfargotunnel\.com)`)
		connRegex := regexp.MustCompile(`Registered tunnel connection`)
		for scanner.Scan() {
			line := scanner.Text()
			log.Printf("[TUNNEL] %s", line)
			if match := urlRegex.FindString(line); match != "" {
				select {
				case urlCh <- match:
				default:
				}
			}
			// For named tunnels, connection registered means tunnel is up
			if token != "" && connRegex.MatchString(line) {
				log.Println("[TUNNEL] Named tunnel connection registered")
			}
		}
	}()

	// Wait for URL with timeout (only for quick tunnels)
	go func() {
		if token != "" {
			// Named tunnel - URL is configured in Cloudflare dashboard
			// We wait a bit for connection registration
			time.Sleep(10 * time.Second)
			m.mu.Lock()
			if m.running && m.url == "" {
				m.url = "(configured in Cloudflare dashboard)"
			}
			m.mu.Unlock()
			return
		}

		select {
		case url := <-urlCh:
			m.mu.Lock()
			m.url = url
			m.mu.Unlock()
			log.Printf("[TUNNEL] Active: %s", url)
			if m.onURL != nil {
				m.onURL(url)
			}
		case <-time.After(90 * time.Second):
			log.Println("[TUNNEL] Timeout waiting for URL")
			m.Disable()
		}
	}()

	// Monitor process for auto-restart
	go func() {
		cmd.Wait()
		m.mu.Lock()
		wasRunning := m.running
		m.running = false
		m.url = ""
		m.cmd = nil
		m.mu.Unlock()

		if wasRunning {
			log.Println("[TUNNEL] Process exited unexpectedly, retrying in 5s")
			if m.onURL != nil {
				m.onURL("")
			}
			time.Sleep(5 * time.Second)
			m.Enable()
		}
	}()

	return nil
}

func (m *Manager) Disable() error {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.running = false
	m.url = ""

	if m.cmd != nil && m.cmd.Process != nil {
		m.cmd.Process.Kill()
		m.cmd = nil
	}

	if m.onURL != nil {
		m.onURL("")
	}
	if m.onState != nil {
		m.onState(false)
	}

	log.Println("[TUNNEL] Disabled")
	return nil
}

// AutoStart starts the tunnel if it was previously enabled (from DB settings)
func (m *Manager) AutoStart(tunnelEnabled bool, savedToken string) {
	if savedToken != "" {
		m.mu.Lock()
		if m.token == "" {
			m.token = savedToken
		}
		m.mu.Unlock()
	}
	if tunnelEnabled {
		log.Println("[TUNNEL] Auto-starting (was enabled before shutdown)")
		go m.Enable()
	}
}

func (m *Manager) ensureBinary() (string, error) {
	if err := os.MkdirAll(m.binDir, 0755); err != nil {
		return "", fmt.Errorf("create bin directory: %w", err)
	}

	name := "cloudflared"
	if runtime.GOOS == "windows" {
		name = "cloudflared.exe"
	}
	binPath := filepath.Join(m.binDir, name)

	if _, err := os.Stat(binPath); err == nil {
		if err := validateCloudflared(binPath); err == nil {
			return binPath, nil
		} else {
			log.Printf("[TUNNEL] Replacing invalid cloudflared at %s: %v", binPath, err)
		}
	}

	if path, err := exec.LookPath("cloudflared"); err == nil {
		if err := validateCloudflared(path); err == nil {
			return path, nil
		}
	}

	asset, ok := cloudflaredDownloadAsset(runtime.GOOS, runtime.GOARCH)
	if !ok {
		return "", fmt.Errorf("unsupported platform: %s/%s", runtime.GOOS, runtime.GOARCH)
	}

	log.Printf("[TUNNEL] Downloading cloudflared from %s", asset.url)

	resp, err := http.Get(asset.url)
	if err != nil {
		return "", fmt.Errorf("download: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return "", fmt.Errorf("download failed: HTTP %d", resp.StatusCode)
	}

	tempPattern := ".cloudflared-*"
	if runtime.GOOS == "windows" {
		tempPattern += ".exe"
	}
	temp, err := os.CreateTemp(m.binDir, tempPattern)
	if err != nil {
		return "", fmt.Errorf("create temporary file: %w", err)
	}
	tempPath := temp.Name()
	defer os.Remove(tempPath)

	if err := copyCloudflaredPayload(temp, resp.Body, asset.archive); err != nil {
		temp.Close()
		return "", fmt.Errorf("extract download: %w", err)
	}

	if runtime.GOOS != "windows" {
		if err := temp.Chmod(0755); err != nil {
			temp.Close()
			return "", fmt.Errorf("make executable: %w", err)
		}
	}
	if err := temp.Sync(); err != nil {
		temp.Close()
		return "", fmt.Errorf("sync download: %w", err)
	}
	if err := temp.Close(); err != nil {
		return "", fmt.Errorf("close download: %w", err)
	}
	if err := validateCloudflared(tempPath); err != nil {
		return "", fmt.Errorf("validate download: %w", err)
	}

	if runtime.GOOS == "windows" {
		// Windows cannot atomically replace an existing executable.
		if err := os.Remove(binPath); err != nil && !os.IsNotExist(err) {
			return "", fmt.Errorf("remove invalid binary: %w", err)
		}
	}
	if err := os.Rename(tempPath, binPath); err != nil {
		return "", fmt.Errorf("install binary: %w", err)
	}

	log.Printf("[TUNNEL] Downloaded cloudflared to %s", binPath)
	return binPath, nil
}

type cloudflaredAsset struct {
	url     string
	archive bool
}

func cloudflaredDownloadAsset(goos, goarch string) (cloudflaredAsset, bool) {
	base := "https://github.com/cloudflare/cloudflared/releases/latest/download/"

	switch {
	case goos == "linux" && goarch == "amd64":
		return cloudflaredAsset{url: base + "cloudflared-linux-amd64"}, true
	case goos == "linux" && goarch == "arm64":
		return cloudflaredAsset{url: base + "cloudflared-linux-arm64"}, true
	case goos == "darwin" && goarch == "amd64":
		return cloudflaredAsset{
			url:     base + "cloudflared-darwin-amd64.tgz",
			archive: true,
		}, true
	case goos == "darwin" && goarch == "arm64":
		return cloudflaredAsset{
			url:     base + "cloudflared-darwin-arm64.tgz",
			archive: true,
		}, true
	case goos == "windows" && goarch == "amd64":
		return cloudflaredAsset{url: base + "cloudflared-windows-amd64.exe"}, true
	}
	return cloudflaredAsset{}, false
}

func copyCloudflaredPayload(dst io.Writer, src io.Reader, archive bool) error {
	if !archive {
		_, err := io.Copy(dst, src)
		return err
	}

	gzipReader, err := gzip.NewReader(src)
	if err != nil {
		return fmt.Errorf("open gzip archive: %w", err)
	}
	defer gzipReader.Close()

	tarReader := tar.NewReader(gzipReader)
	for {
		header, err := tarReader.Next()
		if err == io.EOF {
			return fmt.Errorf("cloudflared executable not found in archive")
		}
		if err != nil {
			return fmt.Errorf("read tar archive: %w", err)
		}
		isRegular := header.Typeflag == tar.TypeReg || header.Typeflag == tar.TypeRegA
		if !isRegular || filepath.Base(header.Name) != "cloudflared" {
			continue
		}
		_, err = io.Copy(dst, tarReader)
		return err
	}
}

func validateCloudflared(path string) error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	output, err := exec.CommandContext(ctx, path, "--version").CombinedOutput()
	if ctx.Err() != nil {
		return fmt.Errorf("version check timed out")
	}
	if err != nil {
		return fmt.Errorf("version check: %w", err)
	}
	if !strings.Contains(strings.ToLower(string(output)), "cloudflared version") {
		return fmt.Errorf("unexpected version output: %.120s", strings.TrimSpace(string(output)))
	}
	return nil
}

// RegisterRoutes adds tunnel API endpoints. onSaveToken persists token to DB.
func (m *Manager) RegisterRoutes(mux *http.ServeMux, onSaveToken func(string)) {
	mux.HandleFunc("/api/tunnel/status", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		hasToken := m.Token() != ""
		json.NewEncoder(w).Encode(map[string]any{
			"running":   m.IsRunning(),
			"url":       m.URL(),
			"has_token": hasToken,
		})
	})

	mux.HandleFunc("/api/tunnel/enable", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}

		// Accept optional token in body: {"token": "eyJ..."}
		body, _ := io.ReadAll(r.Body)
		if len(body) > 0 {
			var req struct {
				Token string `json:"token"`
			}
			if json.Unmarshal(body, &req) == nil && req.Token != "" {
				m.SetToken(req.Token)
				if onSaveToken != nil {
					onSaveToken(req.Token)
				}
				log.Printf("[TUNNEL] Token configured via API")
			}
		}

		if err := m.Enable(); err != nil {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusInternalServerError)
			fmt.Fprintf(w, `{"error":"%s"}`, strings.ReplaceAll(err.Error(), `"`, `\"`))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `{"status":"enabling"}`)
	})

	mux.HandleFunc("/api/tunnel/disable", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}
		m.Disable()
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `{"status":"disabled"}`)
	})

	mux.HandleFunc("/api/tunnel/token", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.Method {
		case "GET":
			// Returns whether a token exists (does not expose the full value)
			token := m.Token()
			masked := ""
			if token != "" {
				if len(token) > 12 {
					masked = token[:6] + "..." + token[len(token)-6:]
				} else {
					masked = "***"
				}
			}
			json.NewEncoder(w).Encode(map[string]any{
				"has_token": token != "",
				"masked":    masked,
			})
		case "PUT":
			body, _ := io.ReadAll(r.Body)
			var req struct {
				Token string `json:"token"`
			}
			if json.Unmarshal(body, &req) != nil {
				w.WriteHeader(http.StatusBadRequest)
				fmt.Fprintf(w, `{"error":"invalid JSON"}`)
				return
			}

			m.SetToken(req.Token)
			if onSaveToken != nil {
				onSaveToken(req.Token)
			}

			// If tunnel is running, restart with the new token
			if m.IsRunning() {
				m.Disable()
				go m.Enable()
			}

			action := "saved"
			if req.Token == "" {
				action = "removed"
			}
			log.Printf("[TUNNEL] Token %s via API", action)
			json.NewEncoder(w).Encode(map[string]string{"status": "ok"})

		case "DELETE":
			m.SetToken("")
			if onSaveToken != nil {
				onSaveToken("")
			}
			if m.IsRunning() {
				m.Disable()
			}
			log.Printf("[TUNNEL] Token removed via API")
			json.NewEncoder(w).Encode(map[string]string{"status": "ok"})

		default:
			w.WriteHeader(http.StatusMethodNotAllowed)
		}
	})
}

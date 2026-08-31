package auth

import (
	"fmt"
	"net"
	"strings"
	"testing"
	"time"
)

// registerTestProvider adds a throwaway OAuth config bound to a free port so these
// tests never contend with the real Claude callback port (54545), which a running
// sidecar on the same machine may already hold.
func registerTestProvider(t *testing.T) (name string, port int) {
	t.Helper()

	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("could not reserve a free port: %v", err)
	}
	port = listener.Addr().(*net.TCPAddr).Port
	if err := listener.Close(); err != nil {
		t.Fatalf("could not release the reserved port: %v", err)
	}

	name = fmt.Sprintf("test-provider-%d", port)
	Configs[name] = OAuthConfig{
		Provider:     name,
		ClientID:     "test-client",
		AuthURL:      "https://example.invalid/authorize",
		TokenURL:     "https://example.invalid/token",
		CallbackPort: port,
		CallbackPath: "/callback",
		UsePKCE:      true,
	}
	t.Cleanup(func() { delete(Configs, name) })

	return name, port
}

func callbackPortIsFree(t *testing.T, port int) bool {
	t.Helper()

	listener, err := net.Listen("tcp", fmt.Sprintf(":%d", port))
	if err != nil {
		return false
	}
	if err := listener.Close(); err != nil {
		t.Fatalf("could not close the probe listener: %v", err)
	}
	return true
}

func callbackServerFor(t *testing.T, fm *FlowManager, flowID string) *CallbackServer {
	t.Helper()

	fm.mu.RLock()
	defer fm.mu.RUnlock()
	flow, ok := fm.flows[flowID]
	if !ok {
		t.Fatalf("flow %q is not registered", flowID)
	}
	return flow.callback
}

// A failed flow used to keep the fixed callback port bound for the full 10 minute
// flow lifetime, which silently downgraded every later attempt to manual paste mode.
func TestFailedFlowReleasesTheCallbackPort(t *testing.T) {
	provider, port := registerTestProvider(t)
	fm := NewFlowManager()

	flowID, _, err := fm.StartFlow(provider)
	if err != nil {
		t.Fatalf("StartFlow: %v", err)
	}
	callback := callbackServerFor(t, fm, flowID)
	if callback == nil {
		t.Fatal("the first flow did not get a callback server")
	}
	if callbackPortIsFree(t, port) {
		t.Fatal("the callback server is not listening")
	}

	// Deliver a callback carrying the wrong state, the way a stale browser tab would.
	callback.resultCh <- CallbackResult{Code: "test-code", State: "not-the-expected-state"}

	if _, err := fm.WaitForCallback(flowID, 2*time.Second); err == nil {
		t.Fatal("expected a state mismatch error")
	}

	if !callbackPortIsFree(t, port) {
		t.Fatal("a failed flow leaked the callback port")
	}
}

func TestCallbackTimeoutReleasesTheCallbackPort(t *testing.T) {
	provider, port := registerTestProvider(t)
	fm := NewFlowManager()

	flowID, _, err := fm.StartFlow(provider)
	if err != nil {
		t.Fatalf("StartFlow: %v", err)
	}

	if _, err := fm.WaitForCallback(flowID, 10*time.Millisecond); err == nil {
		t.Fatal("expected a timeout error")
	}

	if !callbackPortIsFree(t, port) {
		t.Fatal("a timed out flow leaked the callback port")
	}
}

// Even if a listener does survive some future path, starting a new flow has to
// reclaim the port rather than fall back to manual mode.
func TestStartFlowReclaimsThePortFromAnEarlierFlow(t *testing.T) {
	provider, port := registerTestProvider(t)
	fm := NewFlowManager()

	firstID, _, err := fm.StartFlow(provider)
	if err != nil {
		t.Fatalf("StartFlow: %v", err)
	}
	if callbackServerFor(t, fm, firstID) == nil {
		t.Fatal("the first flow did not get a callback server")
	}

	// Abandon the first flow without completing it, then start another.
	secondID, _, err := fm.StartFlow(provider)
	if err != nil {
		t.Fatalf("second StartFlow: %v", err)
	}
	if secondID == firstID {
		t.Fatal("the second flow reused the first flow id")
	}
	if callbackServerFor(t, fm, secondID) == nil {
		t.Fatal("the second flow fell back to manual mode instead of reclaiming the port")
	}
	if callbackServerFor(t, fm, firstID) != nil {
		t.Fatal("the superseded flow kept its callback server")
	}
	if callbackPortIsFree(t, port) {
		t.Fatal("the second flow is not listening on the callback port")
	}

	fm.stopCallback(secondID)
}

func TestParseAuthorizationInput(t *testing.T) {
	cases := []struct {
		name      string
		raw       string
		code      string
		state     string
		wantError bool
	}{
		{
			name:  "full callback url from the address bar",
			raw:   "http://localhost:54545/callback?code=abc123&state=xyz789",
			code:  "abc123",
			state: "xyz789",
		},
		{
			name:  "url whose code carries the state after a hash",
			raw:   "http://localhost:54545/callback?code=abc123%23xyz789",
			code:  "abc123",
			state: "xyz789",
		},
		{
			name:  "bare code",
			raw:   "abc123",
			code:  "abc123",
			state: "",
		},
		{
			name:  "bare code and state joined by a hash",
			raw:   "abc123#xyz789",
			code:  "abc123",
			state: "xyz789",
		},
		{
			name:  "surrounding whitespace from a sloppy paste",
			raw:   "  http://localhost:54545/callback?code=abc123&state=xyz789\n",
			code:  "abc123",
			state: "xyz789",
		},
		{
			name:  "code in the fragment",
			raw:   "http://localhost:54545/callback#code=abc123&state=xyz789",
			code:  "abc123",
			state: "xyz789",
		},
		{
			name:      "provider reported an error",
			raw:       "http://localhost:54545/callback?error=access_denied",
			wantError: true,
		},
		{
			name:      "empty",
			raw:       "   ",
			wantError: true,
		},
		{
			name:      "url with no code at all",
			raw:       "http://localhost:54545/callback",
			wantError: true,
		},
	}

	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			code, state, err := ParseAuthorizationInput(testCase.raw)
			if testCase.wantError {
				if err == nil {
					t.Fatalf("expected an error, got code=%q state=%q", code, state)
				}
				return
			}
			if err != nil {
				t.Fatalf("ParseAuthorizationInput: %v", err)
			}
			if code != testCase.code {
				t.Errorf("code = %q, want %q", code, testCase.code)
			}
			if state != testCase.state {
				t.Errorf("state = %q, want %q", state, testCase.state)
			}
		})
	}
}

// The pasted value carries the authorization code, so it must never reach an error message.
func TestParseAuthorizationInputDoesNotEchoTheInput(t *testing.T) {
	secret := "http://localhost:54545/callback?error=access_denied&code=super-secret-code"
	_, _, err := ParseAuthorizationInput(secret)
	if err == nil {
		t.Fatal("expected an error")
	}
	if strings.Contains(err.Error(), "super-secret-code") {
		t.Fatalf("error echoed the authorization code: %v", err)
	}
}

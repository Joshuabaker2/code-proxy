package provider

import (
	"encoding/json"
	"strings"
	"testing"
)

// translateMessages runs the real translator over an OpenAI message list and
// returns Anthropic's messages array.
func translateMessages(t *testing.T, messages string) []map[string]any {
	t.Helper()
	body := `{"model":"claude-sonnet-5","messages":` + messages + `}`
	translated, _, err := translateOpenAIToAnthropic([]byte(body), "claude-sonnet-5", nil)
	if err != nil {
		t.Fatalf("translate: %v", err)
	}
	var claude struct {
		Messages []map[string]any `json:"messages"`
	}
	if err := json.Unmarshal(translated, &claude); err != nil {
		t.Fatalf("parse translated request: %v", err)
	}
	return claude.Messages
}

func blocksOf(t *testing.T, msg map[string]any) []map[string]any {
	t.Helper()
	blocks := contentBlocks(msg["content"])
	if blocks == nil {
		t.Fatalf("message has no content blocks: %v", msg)
	}
	return blocks
}

const assistantCallsA = `{"role":"assistant","content":null,"tool_calls":[{"id":"call_A","type":"function","function":{"name":"read_uploaded_pdf","arguments":"{\"documentId\":\"d1\"}"}}]}`
const assistantCallsAB = `{"role":"assistant","content":null,"tool_calls":[{"id":"call_A","type":"function","function":{"name":"read_uploaded_pdf","arguments":"{}"}},{"id":"call_B","type":"function","function":{"name":"get_document_facts","arguments":"{}"}}]}`
const toolResultA = `{"role":"tool","tool_call_id":"call_A","content":"page text"}`
const toolResultB = `{"role":"tool","tool_call_id":"call_B","content":"facts"}`

// The reported failure: a tool call with nothing after it. Anthropic rejects
// the request; the repair answers the call with an explicit error result.
func TestOrphanedTrailingToolCallGetsAnErrorResult(t *testing.T) {
	messages := translateMessages(t, `[{"role":"user","content":"hi"},`+assistantCallsA+`]`)

	last := messages[len(messages)-1]
	if last["role"] != "user" {
		t.Fatalf("last message role = %v, want a synthetic user turn", last["role"])
	}
	blocks := blocksOf(t, last)
	if len(blocks) != 1 || blocks[0]["type"] != "tool_result" || blocks[0]["tool_use_id"] != "call_A" {
		t.Fatalf("synthetic turn = %v, want one tool_result for call_A", blocks)
	}
	if blocks[0]["is_error"] != true {
		t.Errorf("synthetic result must be flagged is_error so the model does not trust it")
	}
}

// A user typing over an unanswered call: the answer must come before the text.
func TestOrphanedCallFollowedByUserTextIsAnsweredFirst(t *testing.T) {
	messages := translateMessages(t, `[{"role":"user","content":"hi"},`+assistantCallsA+`,{"role":"user","content":"never mind, summarize"}]`)

	last := messages[len(messages)-1]
	blocks := blocksOf(t, last)
	if len(blocks) != 2 {
		t.Fatalf("blocks = %v, want [tool_result, text]", blocks)
	}
	if blocks[0]["type"] != "tool_result" || blocks[0]["tool_use_id"] != "call_A" {
		t.Errorf("first block = %v, want the synthetic tool_result", blocks[0])
	}
	if blocks[1]["type"] != "text" || !strings.Contains(blocks[1]["text"].(string), "summarize") {
		t.Errorf("second block = %v, want the user's text", blocks[1])
	}
}

// Two assistant turns in a row with the first one's call unanswered.
func TestOrphanedCallFollowedByAssistantGetsAUserTurnInserted(t *testing.T) {
	messages := translateMessages(t, `[{"role":"user","content":"hi"},`+assistantCallsA+`,{"role":"assistant","content":"I could not read it."}]`)

	if len(messages) != 4 {
		t.Fatalf("got %d messages, want user, assistant, synthetic user, assistant", len(messages))
	}
	inserted := blocksOf(t, messages[2])
	if messages[2]["role"] != "user" || inserted[0]["tool_use_id"] != "call_A" {
		t.Fatalf("messages[2] = %v, want a synthetic user turn answering call_A", messages[2])
	}
	if messages[3]["role"] != "assistant" {
		t.Errorf("messages[3] role = %v, want the trailing assistant", messages[3]["role"])
	}
}

// The normal Goose shape: two calls, two separate tool messages. They must
// fold into one user turn with both results and no synthetic ones.
func TestSeparateToolMessagesMergeIntoOneAnsweredTurn(t *testing.T) {
	messages := translateMessages(t, `[{"role":"user","content":"hi"},`+assistantCallsAB+`,`+toolResultA+`,`+toolResultB+`]`)

	if len(messages) != 3 {
		t.Fatalf("got %d messages, want user, assistant, one merged user turn", len(messages))
	}
	blocks := blocksOf(t, messages[2])
	if len(blocks) != 2 {
		t.Fatalf("merged turn = %v, want exactly the two real results", blocks)
	}
	for i, id := range []string{"call_A", "call_B"} {
		if blocks[i]["tool_use_id"] != id || blocks[i]["is_error"] == true {
			t.Errorf("block %d = %v, want the real result for %s", i, blocks[i], id)
		}
	}
}

// One of two calls answered: only the missing one is synthesized, first.
func TestPartiallyAnsweredTurnSynthesizesOnlyTheMissingResult(t *testing.T) {
	messages := translateMessages(t, `[{"role":"user","content":"hi"},`+assistantCallsAB+`,`+toolResultB+`]`)

	blocks := blocksOf(t, messages[2])
	if len(blocks) != 2 {
		t.Fatalf("turn = %v, want synthetic A then real B", blocks)
	}
	if blocks[0]["tool_use_id"] != "call_A" || blocks[0]["is_error"] != true {
		t.Errorf("first block = %v, want the synthetic result for call_A", blocks[0])
	}
	if blocks[1]["tool_use_id"] != "call_B" || blocks[1]["is_error"] == true {
		t.Errorf("second block = %v, want the real result for call_B", blocks[1])
	}
}

// A result nothing asked for (e.g. compaction dropped the assistant turn) is
// the mirror-image 400; it is dropped rather than sent.
func TestToolResultWithoutAToolUseIsDropped(t *testing.T) {
	messages := translateMessages(t, `[{"role":"user","content":"hi"},{"role":"assistant","content":"ok"},`+toolResultA+`,{"role":"user","content":"and now?"}]`)

	for _, msg := range messages {
		for _, block := range contentBlocks(msg["content"]) {
			if block["type"] == "tool_result" {
				t.Fatalf("stray tool_result survived: %v", block)
			}
		}
	}
	last := blocksOf(t, messages[len(messages)-1])
	if last[0]["type"] != "text" {
		t.Errorf("the user's text should still be sent, got %v", last)
	}
}

// A well-formed history is passed through untouched.
func TestWellFormedHistoryIsUnchangedByTheRepair(t *testing.T) {
	messages := translateMessages(t, `[{"role":"user","content":"hi"},`+assistantCallsA+`,`+toolResultA+`,{"role":"assistant","content":"done"}]`)

	if len(messages) != 4 {
		t.Fatalf("got %d messages, want 4", len(messages))
	}
	blocks := blocksOf(t, messages[2])
	if len(blocks) != 1 || blocks[0]["tool_use_id"] != "call_A" || blocks[0]["is_error"] == true {
		t.Fatalf("answer turn = %v, want exactly the real result", blocks)
	}
}

package provider

import (
	"encoding/json"
	"fmt"
	"strings"
)

// describeMessageShape renders an Anthropic request's messages as a compact
// structural outline — roles, block types and tool ids, never content — so a
// pairing rejection from upstream can be diagnosed from the log after the
// fact. The request log table keeps no bodies, and the failing history is
// gone once the client moves on.
func describeMessageShape(body []byte) string {
	var request struct {
		Messages []map[string]any `json:"messages"`
	}
	if err := json.Unmarshal(body, &request); err != nil {
		return "unparseable request body"
	}
	lines := make([]string, 0, len(request.Messages))
	for index, message := range request.Messages {
		role, _ := message["role"].(string)
		var blocks []string
		for _, block := range contentBlocks(message["content"]) {
			blockType, _ := block["type"].(string)
			switch blockType {
			case "tool_use":
				id, _ := block["id"].(string)
				name, _ := block["name"].(string)
				blocks = append(blocks, fmt.Sprintf("tool_use(%s %s)", name, id))
			case "tool_result":
				id, _ := block["tool_use_id"].(string)
				if isError, _ := block["is_error"].(bool); isError {
					blocks = append(blocks, fmt.Sprintf("tool_result!(%s)", id))
				} else {
					blocks = append(blocks, fmt.Sprintf("tool_result(%s)", id))
				}
			default:
				blocks = append(blocks, blockType)
			}
		}
		if text, ok := message["content"].(string); ok && len(blocks) == 0 {
			blocks = append(blocks, fmt.Sprintf("text[%d]", len(text)))
		}
		lines = append(lines, fmt.Sprintf("%d %s: %s", index, role, strings.Join(blocks, ", ")))
	}
	return strings.Join(lines, "\n")
}

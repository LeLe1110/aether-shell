#!/bin/bash
# =============================================================================
# gmn.chuangzuoli.com Responses API Verification Script
# Quick version: no auto-retry, fails fast on 502
# =============================================================================

# --- Configuration (edit these or pass as env vars) ---
API_KEY="${API_KEY:-sk-19d90c66b68f233c9c0cd9c5e4aa3df06ebd62f49652b5b7cc8c2b4fbf79c762}"
BASE_URL="${BASE_URL:-https://gmn.chuangzuoli.com}"
MODEL="${MODEL:-gpt-5.2}"
PROXY="${PROXY:-}"  # e.g. "http://127.0.0.1:7897" or leave empty
TIMEOUT="${TIMEOUT:-30}"

# --- Helpers ---
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; CYAN='\033[0;36m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[OK]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
info() { echo -e "     $*"; }

CURL_OPTS=("-s" "--connect-timeout" "10" "--max-time" "$TIMEOUT")
if [ -n "$PROXY" ]; then
    CURL_OPTS+=("--noproxy" "*" "-x" "$PROXY")
fi

do_post() {
    local url="$1"
    local data="$2"
    local resp
    local http_code
    resp=$(curl "${CURL_OPTS[@]}" -w '\n__HTTP_CODE__%{http_code}' \
        -H "Authorization: Bearer $API_KEY" \
        -H "Content-Type: application/json" \
        -d "$data" "$url" 2>/dev/null)
    http_code=$(echo "$resp" | grep '__HTTP_CODE__' | sed 's/__HTTP_CODE__//')
    body=$(echo "$resp" | grep -v '__HTTP_CODE__')
    echo "$http_code"
    echo "$body"
}

do_get() {
    local url="$1"
    curl "${CURL_OPTS[@]}" -w '\n__HTTP_CODE__%{http_code}' \
        -H "Authorization: Bearer $API_KEY" "$url" 2>/dev/null
}

extract_json_field() {
    python3 -c "import sys,json; print(json.load(sys.stdin).get('$1',''))" 2>/dev/null
}

extract_text() {
    python3 -c "
import sys,json
d=json.load(sys.stdin)
t=d.get('output_text','')
if not t:
    for i in d.get('output',[]):
        if i.get('type')=='message':
            for c in i.get('content',[]):
                if c.get('type')=='output_text': t+=c.get('text','')
print(t[:500])
" 2>/dev/null
}

# =============================================================================
echo -e "${CYAN}============================================================${NC}"
echo -e "${CYAN} Responses API Verification (quick mode)${NC}"
echo -e "${CYAN}============================================================${NC}"
echo "Time:      $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "Endpoint:  $BASE_URL"
echo "Model:     $MODEL"
echo "Proxy:     ${PROXY:-none}"
echo "Timeout:   ${TIMEOUT}s"
echo ""

# =============================================================================
# Test 1: GET /v1/models
# =============================================================================
echo -e "${CYAN}--- Test 1: GET /v1/models ---${NC}"
MODELS_RAW=$(do_get "$BASE_URL/v1/models")
MODELS_CODE=$(echo "$MODELS_RAW" | grep '__HTTP_CODE__' | sed 's/__HTTP_CODE__//')
MODELS_BODY=$(echo "$MODELS_RAW" | grep -v '__HTTP_CODE__')

if [ "$MODELS_CODE" = "200" ]; then
    MODEL_LIST=$(echo "$MODELS_BODY" | python3 -c "import sys,json; print(', '.join(m['id'] for m in json.load(sys.stdin).get('data',[])))" 2>/dev/null)
    ok "HTTP $MODELS_CODE — Models: $MODEL_LIST"
else
    fail "HTTP $MODELS_CODE"
    info "$(echo "$MODELS_BODY" | head -c 100)"
fi
echo ""

# =============================================================================
# Test 2: POST /v1/chat/completions
# =============================================================================
echo -e "${CYAN}--- Test 2: POST /v1/chat/completions ---${NC}"
CHAT_RAW=$(do_post "$BASE_URL/v1/chat/completions" \
    "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":16}")
CHAT_CODE=$(echo "$CHAT_RAW" | head -1)
if [ "$CHAT_CODE" = "200" ]; then
    ok "HTTP $CHAT_CODE — Chat Completions supported"
elif [ "$CHAT_CODE" = "404" ]; then
    warn "HTTP $CHAT_CODE — Chat Completions NOT supported (only Responses API)"
else
    warn "HTTP $CHAT_CODE"
fi
echo ""

# =============================================================================
# Test 3: POST /v1/responses (basic)
# =============================================================================
echo -e "${CYAN}--- Test 3: POST /v1/responses (basic) ---${NC}"
T3_RAW=$(do_post "$BASE_URL/v1/responses" \
    "{\"model\":\"$MODEL\",\"input\":[{\"role\":\"user\",\"content\":\"Say OK\"}],\"max_output_tokens\":32,\"temperature\":0.1}")
T3_CODE=$(echo "$T3_RAW" | head -1)
T3_BODY=$(echo "$T3_RAW" | tail -n +2)

if [ "$T3_CODE" = "200" ]; then
    T3_ID=$(echo "$T3_BODY" | extract_json_field "id")
    T3_TEXT=$(echo "$T3_BODY" | extract_text)
    T3_STORE=$(echo "$T3_BODY" | extract_json_field "store")
    ok "HTTP $T3_CODE"
    info "ID:    $T3_ID"
    info "Text:  $T3_TEXT"
    info "Store: $T3_STORE"
else
    fail "HTTP $T3_CODE — API unavailable"
    info "$(echo "$T3_BODY" | head -c 100)"
    echo ""
    echo -e "${RED}Cannot continue: /v1/responses returned $T3_CODE${NC}"
    echo "Try again later or check source server."
    exit 1
fi
echo ""

# =============================================================================
# Test 4: With system prompt + store=true
# =============================================================================
echo -e "${CYAN}--- Test 4: System prompt + store=true ---${NC}"
T4_RAW=$(do_post "$BASE_URL/v1/responses" \
    "{\"model\":\"$MODEL\",\"input\":[{\"role\":\"system\",\"content\":\"Your name is TestBot-7890. Always say TestBot-7890 in every reply.\"},{\"role\":\"user\",\"content\":\"What is your name? Reply in one sentence.\"}],\"max_output_tokens\":64,\"temperature\":0.1,\"store\":true}")
T4_CODE=$(echo "$T4_RAW" | head -1)
T4_BODY=$(echo "$T4_RAW" | tail -n +2)

if [ "$T4_CODE" = "200" ]; then
    T4_ID=$(echo "$T4_BODY" | extract_json_field "id")
    T4_TEXT=$(echo "$T4_BODY" | extract_text)
    T4_STORE=$(echo "$T4_BODY" | extract_json_field "store")
    T4_INSTR=$(echo "$T4_BODY" | python3 -c "import sys,json; print(json.load(sys.stdin).get('instructions','')[:100])" 2>/dev/null)
    ok "HTTP $T4_CODE"
    info "ID:           $T4_ID"
    info "Text:         $T4_TEXT"
    info "Store:        $T4_STORE"
    info "Instructions: $T4_INSTR"

    if echo "$T4_TEXT" | grep -qi "TestBot-7890"; then
        ok "System prompt applied (TestBot-7890 mentioned)"
    else
        warn "System prompt may NOT be applied (TestBot-7890 not in output)"
    fi
else
    fail "HTTP $T4_CODE"
    info "$(echo "$T4_BODY" | head -c 100)"
    echo ""
    echo -e "${RED}Cannot continue: system prompt request failed${NC}"
    exit 1
fi
echo ""

# =============================================================================
# Test 5: previous_response_id (KEY TEST)
# =============================================================================
echo -e "${CYAN}--- Test 5: previous_response_id (KEY TEST) ---${NC}"
echo "  Sending: previous_response_id=$T4_ID"
echo "  Sending: ONLY user message (no system prompt)"
echo ""

T5_RAW=$(do_post "$BASE_URL/v1/responses" \
    "{\"model\":\"$MODEL\",\"input\":[{\"role\":\"user\",\"content\":\"What is your name? Say it again.\"}],\"previous_response_id\":\"$T4_ID\",\"max_output_tokens\":64,\"temperature\":0.1,\"store\":true}")
T5_CODE=$(echo "$T5_RAW" | head -1)
T5_BODY=$(echo "$T5_RAW" | tail -n +2)

if [ "$T5_CODE" = "200" ]; then
    T5_ID=$(echo "$T5_BODY" | extract_json_field "id")
    T5_TEXT=$(echo "$T5_BODY" | extract_text)
    T5_PREV=$(echo "$T5_BODY" | extract_json_field "previous_response_id")
    T5_INSTR=$(echo "$T5_BODY" | python3 -c "import sys,json; print(json.load(sys.stdin).get('instructions','')[:100])" 2>/dev/null)
    ok "HTTP $T5_CODE"
    info "Response ID:       $T5_ID"
    info "Previous Resp ID:  $T5_PREV"
    info "Text:              $T5_TEXT"
    info "Instructions:      $T5_INSTR"
else
    fail "HTTP $T5_CODE — Chain request failed"
    info "$(echo "$T5_BODY" | head -c 100)"
    T5_TEXT=""
fi
echo ""

# =============================================================================
# Test 6: Third request in chain
# =============================================================================
if [ -n "$T5_ID" ] && [ "$T5_CODE" = "200" ]; then
    echo -e "${CYAN}--- Test 6: Third request in chain ---${NC}"
    T6_RAW=$(do_post "$BASE_URL/v1/responses" \
        "{\"model\":\"$MODEL\",\"input\":[{\"role\":\"user\",\"content\":\"What was the first question I asked you?\"}],\"previous_response_id\":\"$T5_ID\",\"max_output_tokens\":128,\"temperature\":0.1,\"store\":true}")
    T6_CODE=$(echo "$T6_RAW" | head -1)
    T6_BODY=$(echo "$T6_RAW" | tail -n +2)

    if [ "$T6_CODE" = "200" ]; then
        T6_TEXT=$(echo "$T6_BODY" | extract_text)
        ok "HTTP $T6_CODE"
        info "Text: $T6_TEXT"
    else
        warn "HTTP $T6_CODE — Third request failed (non-critical)"
    fi
    echo ""
fi

# =============================================================================
# VERDICT
# =============================================================================
echo -e "${CYAN}============================================================${NC}"
echo -e "${CYAN} VERDICT${NC}"
echo -e "${CYAN}============================================================${NC}"

# Check 1: Valid resp_* IDs
if [[ "$T4_ID" == resp_* ]]; then
    ok "API returns valid resp_* IDs"
else
    fail "API does NOT return valid resp_* IDs (got: $T4_ID)"
fi

# Check 2: previous_response_id context preservation
if [ -n "$T5_TEXT" ] && echo "$T5_TEXT" | grep -qi "TestBot-7890"; then
    ok "previous_response_id WORKS — system prompt preserved without re-sending"
    echo ""
    echo -e "${GREEN}CONCLUSION: Native session mode is supported by this API.${NC}"
    echo -e "${GREEN}nanobot can safely use previous_response_id for session continuity.${NC}"
else
    fail "previous_response_id NOT WORKING — TestBot-7890 not in follow-up"
    if [ -n "$T5_TEXT" ]; then
        info "Follow-up response was: $T5_TEXT"
    fi
    echo ""
    echo -e "${RED}CONCLUSION: Native session may NOT be supported.${NC}"
    echo -e "${RED}nanobot should use stateless mode (send system prompt every time).${NC}"
fi

echo -e "${CYAN}============================================================${NC}"

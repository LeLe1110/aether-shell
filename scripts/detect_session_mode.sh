#!/bin/bash
# =============================================================================
# Nanobot Session Mode Detection Script
#
# Detects which session mode an API endpoint supports:
#   - native:    Supports previous_response_id (full server-side session)
#   - stateless: Supports /v1/responses but NOT previous_response_id
#   - completions: Only supports /v1/chat/completions (no Responses API)
#   - unavailable: Endpoint not reachable
#
# Usage:
#   API_KEY="sk-xxx" BASE_URL="https://example.com" bash scripts/detect_session_mode.sh
#   API_KEY="sk-xxx" BASE_URL="https://example.com" MODEL="gpt-4o" bash scripts/detect_session_mode.sh
#
# Environment variables:
#   API_KEY     (required) API key
#   BASE_URL    (required) API base URL, e.g. https://example.com or https://example.com/v1
#   MODEL       (optional) Model name (default: gpt-4o)
#   PROXY       (optional) Proxy URL, e.g. http://127.0.0.1:7897
#   TIMEOUT     (optional) Request timeout in seconds (default: 60)
# =============================================================================

set -euo pipefail

API_KEY="${API_KEY:-}"
BASE_URL="${BASE_URL:-}"
MODEL="${MODEL:-gpt-4o}"
PROXY="${PROXY:-}"
TIMEOUT="${TIMEOUT:-60}"

if [ -z "$API_KEY" ] || [ -z "$BASE_URL" ]; then
    echo "Usage: API_KEY=\"sk-xxx\" BASE_URL=\"https://example.com\" bash $0"
    echo ""
    echo "Environment variables:"
    echo "  API_KEY     (required) API key"
    echo "  BASE_URL    (required) API base URL"
    echo "  MODEL       (optional) Model name (default: gpt-4o)"
    echo "  PROXY       (optional) Proxy URL"
    echo "  TIMEOUT     (optional) Request timeout in seconds (default: 60)"
    exit 1
fi

# Use Python for all HTTP requests to handle SSE streaming properly
PYTHON="${PYTHON:-python3}"

exec "$PYTHON" - "$API_KEY" "$BASE_URL" "$MODEL" "$PROXY" "$TIMEOUT" << 'PYEOF'
import sys, json, time

api_key = sys.argv[1]
base_url = sys.argv[2].rstrip("/")
model = sys.argv[3]
proxy = sys.argv[4] if len(sys.argv) > 4 else ""
timeout = int(sys.argv[5]) if len(sys.argv) > 5 else 60

try:
    import httpx
except ImportError:
    print("ERROR: httpx not installed. Run: pip install httpx")
    sys.exit(1)

# --- Colors ---
RED = "\033[0;31m"; GREEN = "\033[0;32m"; YELLOW = "\033[0;33m"
CYAN = "\033[0;36m"; NC = "\033[0m"

def ok(msg):   print(f"{GREEN}[OK]{NC} {msg}")
def fail(msg): print(f"{RED}[FAIL]{NC} {msg}")
def warn(msg): print(f"{YELLOW}[WARN]{NC} {msg}")
def info(msg): print(f"     {msg}")

# --- Build URLs ---
if base_url.endswith("/responses"):
    responses_url = base_url
    base_for_others = base_url.rsplit("/responses", 1)[0]
elif base_url.endswith("/v1"):
    responses_url = base_url + "/responses"
    base_for_others = base_url
else:
    responses_url = base_url + "/v1/responses"
    base_for_others = base_url + "/v1"

models_url = base_for_others + "/models"
completions_url = base_for_others + "/chat/completions"

headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
client_kwargs = {"timeout": float(timeout)}
if proxy:
    client_kwargs["proxy"] = proxy


def parse_sse(text):
    """Extract the response.completed event from SSE stream text."""
    for line in text.split("\n"):
        line = line.strip()
        if not line.startswith("data: "):
            continue
        data_str = line[6:].strip()
        if data_str == "[DONE]":
            continue
        try:
            d = json.loads(data_str)
            if d.get("type") == "response.completed":
                return d.get("response", d)
        except json.JSONDecodeError:
            continue
    return None


def do_responses_post(url, body):
    """POST to Responses API. Handles both JSON and SSE responses."""
    with httpx.Client(**client_kwargs) as client:
        r = client.post(url, headers=headers, json=body)
        status = r.status_code
        text = r.text.strip()

        if status != 200:
            # Try to parse error JSON
            error_msg = text[:300]
            try:
                d = json.loads(text)
                error_msg = d.get("detail", "") or json.dumps(d.get("error", {}))
            except:
                if "<html" in text.lower():
                    error_msg = f"(HTML error page, {len(text)} bytes)"
            return status, None, error_msg

        # Try JSON first
        if text.startswith("{"):
            try:
                return 200, json.loads(text), None
            except:
                pass

        # Try SSE
        parsed = parse_sse(text)
        if parsed:
            return 200, parsed, None

        return 200, None, "Could not parse response"


def extract_text(d):
    t = d.get("output_text", "")
    if not t:
        for item in d.get("output", []):
            if item.get("type") == "message":
                for c in item.get("content", []):
                    if c.get("type") == "output_text":
                        t += c.get("text", "")
    return t[:300]


# =============================================================================
print(f"{CYAN}{'='*60}{NC}")
print(f"{CYAN} Nanobot Session Mode Detection{NC}")
print(f"{CYAN}{'='*60}{NC}")
print(f"Time:      {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
print(f"Endpoint:  {base_url}")
print(f"Model:     {model}")
print(f"Proxy:     {proxy or 'none'}")
print()

has_responses = False
has_completions = False
has_native = False
store_works = False

# =============================================================================
# Test 1: GET /v1/models
# =============================================================================
print(f"{CYAN}--- Test 1: Connectivity (GET /v1/models) ---{NC}")
try:
    with httpx.Client(**client_kwargs) as client:
        r = client.get(models_url, headers={"Authorization": f"Bearer {api_key}"})
        if r.status_code == 200:
            try:
                model_ids = [m["id"] for m in r.json().get("data", [])]
                ok(f"HTTP 200 — Models: {', '.join(model_ids)[:200]}")
            except:
                ok(f"HTTP 200")
        else:
            warn(f"HTTP {r.status_code}")
except Exception as e:
    warn(f"Connection error: {e}")
print()

# =============================================================================
# Test 2: POST /v1/chat/completions
# =============================================================================
print(f"{CYAN}--- Test 2: Chat Completions API ---{NC}")
try:
    with httpx.Client(**client_kwargs) as client:
        r = client.post(completions_url, headers=headers, json={
            "model": model,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 16,
        })
        if r.status_code == 200:
            ok("HTTP 200 — /v1/chat/completions supported")
            has_completions = True
        elif r.status_code == 404:
            warn("HTTP 404 — /v1/chat/completions not available")
        else:
            warn(f"HTTP {r.status_code}")
except Exception as e:
    warn(f"Connection error: {e}")
print()

# =============================================================================
# Test 3: POST /v1/responses (basic)
# =============================================================================
print(f"{CYAN}--- Test 3: Responses API (basic) ---{NC}")
status, d3, err3 = do_responses_post(responses_url, {
    "model": model,
    "input": [{"role": "user", "content": "Say OK"}],
    "max_output_tokens": 32,
    "temperature": 0.1,
})

if status == 200 and d3:
    resp_id = d3.get("id", "")
    resp_text = extract_text(d3)
    resp_store = d3.get("store")
    ok("HTTP 200")
    info(f"ID:    {resp_id}")
    info(f"Text:  {resp_text}")
    info(f"Store: {resp_store}")
    has_responses = True
    if resp_id.startswith("resp_"):
        ok("Valid resp_* ID format")
    else:
        warn(f"Non-standard ID format: {resp_id}")
else:
    fail(f"HTTP {status} — Responses API not available")
    if err3:
        info(err3)
    print()
    result_mode = "completions" if has_completions else "unavailable"
    # Skip to verdict
    print(f"{CYAN}{'='*60}{NC}")
    print(f"{CYAN} VERDICT{NC}")
    print(f"{CYAN}{'='*60}{NC}")
    if result_mode == "completions":
        print(f"{YELLOW}Session mode: completions{NC}")
        print()
        print("This API only supports /v1/chat/completions (no Responses API).")
        print()
        print("Recommended nanobot config:")
        print('  (do NOT set api_type to openai-responses)')
    else:
        print(f"{RED}Session mode: unavailable{NC}")
        print()
        print("Could not reach the API or all endpoints returned errors.")
    print()
    print("DETECT_RESULT=" + result_mode)
    sys.exit(0)
print()

# =============================================================================
# Test 4: System prompt + store=true
# =============================================================================
print(f"{CYAN}--- Test 4: System prompt + store=true ---{NC}")
status4, d4, err4 = do_responses_post(responses_url, {
    "model": model,
    "input": [
        {"role": "system", "content": "Your name is TestBot-7890. Always say TestBot-7890 in every reply."},
        {"role": "user", "content": "What is your name? Reply in one sentence."},
    ],
    "max_output_tokens": 64,
    "temperature": 0.1,
    "store": True,
})

t4_id = ""
if status4 == 200 and d4:
    t4_id = d4.get("id", "")
    t4_text = extract_text(d4)
    t4_store = d4.get("store")
    ok("HTTP 200")
    info(f"ID:    {t4_id}")
    info(f"Text:  {t4_text}")
    info(f"Store: {t4_store}")
    if t4_store is True or str(t4_store).lower() == "true":
        store_works = True
        ok("store=true respected")
    else:
        warn("store forced to false (server override)")
else:
    fail(f"HTTP {status4}")
    if err4:
        info(err4)
print()

# =============================================================================
# Test 5: previous_response_id (KEY TEST)
# =============================================================================
if t4_id:
    print(f"{CYAN}--- Test 5: previous_response_id (KEY TEST) ---{NC}")
    info(f"Sending: previous_response_id={t4_id}")
    info("Sending: ONLY user message (no system prompt)")
    print()

    status5, d5, err5 = do_responses_post(responses_url, {
        "model": model,
        "input": [
            {"role": "user", "content": "What is your name? Say it again."},
        ],
        "previous_response_id": t4_id,
        "max_output_tokens": 64,
        "temperature": 0.1,
        "store": True,
    })

    if status5 == 200 and d5:
        t5_text = extract_text(d5)
        t5_prev = d5.get("previous_response_id")
        ok("HTTP 200")
        info(f"Text:               {t5_text}")
        info(f"previous_response_id: {t5_prev}")
        if "testbot-7890" in t5_text.lower():
            has_native = True
            ok("System prompt preserved via previous_response_id!")
        else:
            warn("System prompt NOT preserved (TestBot-7890 not in output)")
            info("The API accepts previous_response_id but may not use it for context")
    elif status5 == 400:
        fail("HTTP 400 — previous_response_id explicitly rejected")
        if err5:
            info(f"Error: {err5}")
    else:
        fail(f"HTTP {status5}")
        if err5:
            info(err5)
    print()

    # =============================================================================
    # Test 6: Third request (optional, for deeper validation)
    # =============================================================================
    if has_native and d5:
        t5_id = d5.get("id", "")
        if t5_id:
            print(f"{CYAN}--- Test 6: Third request in chain ---{NC}")
            status6, d6, err6 = do_responses_post(responses_url, {
                "model": model,
                "input": [
                    {"role": "user", "content": "What was the first question I asked you?"},
                ],
                "previous_response_id": t5_id,
                "max_output_tokens": 128,
                "temperature": 0.1,
                "store": True,
            })
            if status6 == 200 and d6:
                t6_text = extract_text(d6)
                ok(f"HTTP 200")
                info(f"Text: {t6_text}")
            else:
                warn(f"HTTP {status6} (non-critical)")
            print()

# =============================================================================
# DETERMINE RESULT
# =============================================================================
if has_native:
    result_mode = "native"
elif has_responses:
    result_mode = "stateless"
elif has_completions:
    result_mode = "completions"
else:
    result_mode = "unavailable"

# =============================================================================
# VERDICT
# =============================================================================
print(f"{CYAN}{'='*60}{NC}")
print(f"{CYAN} VERDICT{NC}")
print(f"{CYAN}{'='*60}{NC}")

if result_mode == "native":
    print(f"{GREEN}Session mode: native{NC}")
    print()
    print("This API supports previous_response_id for server-side session continuity.")
    print("System prompts are preserved across requests without re-sending.")
    print()
    print("Recommended nanobot config:")
    print('  "api_type": "openai-responses"')
    print('  "session_mode": "native"')
elif result_mode == "stateless":
    print(f"{YELLOW}Session mode: stateless{NC}")
    print()
    print("This API supports /v1/responses format but NOT previous_response_id.")
    print("System prompts must be sent with every request.")
    print()
    print("Recommended nanobot config:")
    print('  "api_type": "openai-responses"')
    print('  "session_mode": "stateless"')
elif result_mode == "completions":
    print(f"{YELLOW}Session mode: completions{NC}")
    print()
    print("This API only supports /v1/chat/completions (no Responses API).")
    print()
    print("Recommended nanobot config:")
    print('  (do NOT set api_type to openai-responses)')
else:
    print(f"{RED}Session mode: unavailable{NC}")
    print()
    print("Could not reach the API or all endpoints returned errors.")
    print("Check your API_KEY, BASE_URL, and network connectivity.")

r = lambda v: f"{GREEN}supported{NC}" if v else f"{RED}not available{NC}"
print()
print(f"{CYAN}Summary:{NC}")
print(f"  /v1/chat/completions:   {r(has_completions)}")
print(f"  /v1/responses:          {r(has_responses)}")
print(f"  previous_response_id:   {r(has_native)}")
store_str = f"{GREEN}respected{NC}" if store_works else f"{YELLOW}forced false{NC}"
print(f"  store=true:             {store_str}")
print(f"{CYAN}{'='*60}{NC}")
print()
print(f"DETECT_RESULT={result_mode}")
PYEOF

#!/bin/bash

URL="https://gmn.chuangzuoli.com/v1/models"
PROXY="http://127.0.0.1:7897"
INTERVAL=30
LOG_FILE="monitor_llm.log"

echo "=== LLM API Monitor ==="
echo "Target: $URL"
echo "Interval: ${INTERVAL}s"
echo "Log: $LOG_FILE"
echo ""
echo "Direct  = bypass Clash (--noproxy '*')"
echo "Proxy   = via Clash ($PROXY)"
echo ""
echo "Press Ctrl+C to stop"
echo ""

total=0
direct_ok=0
direct_fail=0
proxy_ok=0
proxy_fail=0

cleanup() {
    echo ""
    echo "========================================="
    echo "  Summary (total: $total checks)"
    echo "========================================="
    echo "  Direct (bypass Clash): OK=$direct_ok  FAIL=$direct_fail"
    echo "  Proxy  (via Clash):    OK=$proxy_ok  FAIL=$proxy_fail"
    echo "========================================="
    echo ""
    if [ "$direct_fail" -gt 0 ] && [ "$proxy_fail" -gt 0 ]; then
        echo "  -> Both fail: source server (gmn.chuangzuoli.com) is unstable"
    elif [ "$direct_fail" -gt 0 ] && [ "$proxy_fail" -eq 0 ]; then
        echo "  -> Direct fails, Proxy OK: your direct network to the server is unstable, Clash proxy helps"
    elif [ "$direct_fail" -eq 0 ] && [ "$proxy_fail" -gt 0 ]; then
        echo "  -> Direct OK, Proxy fails: Clash proxy node is unstable"
    else
        echo "  -> All OK: no issues detected during monitoring"
    fi
    exit 0
}
trap cleanup INT

while true; do
    ts=$(date '+%Y-%m-%d %H:%M:%S')
    total=$((total + 1))

    # Direct request — force bypass all proxies (including system proxy set by Clash)
    direct_result=$(curl -s -o /dev/null -w '%{http_code} %{time_total}' --noproxy '*' --connect-timeout 10 --max-time 15 "$URL" 2>/dev/null)
    direct_code=$(echo "$direct_result" | awk '{print $1}')
    direct_time=$(echo "$direct_result" | awk '{print $2}')

    # Proxy request — explicitly via Clash
    proxy_result=$(curl -s -o /dev/null -w '%{http_code} %{time_total}' --noproxy '*' -x "$PROXY" --connect-timeout 10 --max-time 15 "$URL" 2>/dev/null)
    proxy_code=$(echo "$proxy_result" | awk '{print $1}')
    proxy_time=$(echo "$proxy_result" | awk '{print $2}')

    # Handle connection failure
    [ "$direct_code" = "000" ] && direct_code="TIMEOUT"
    [ "$proxy_code" = "000" ] && proxy_code="TIMEOUT"

    # Count results
    if [ "$direct_code" = "200" ]; then
        direct_ok=$((direct_ok + 1))
        direct_mark="OK"
    else
        direct_fail=$((direct_fail + 1))
        direct_mark="FAIL"
    fi

    if [ "$proxy_code" = "200" ]; then
        proxy_ok=$((proxy_ok + 1))
        proxy_mark="OK"
    else
        proxy_fail=$((proxy_fail + 1))
        proxy_mark="FAIL"
    fi

    line="[$ts] #$total  Direct: $direct_code (${direct_time}s) [$direct_mark]  |  Proxy: $proxy_code (${proxy_time}s) [$proxy_mark]"

    echo "$line"
    echo "$line" >> "$LOG_FILE"

    sleep "$INTERVAL"
done

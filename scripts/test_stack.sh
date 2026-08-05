#!/usr/bin/env bash
set -eo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

pass() { echo -e "${GREEN}[PASS]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; exit 1; }

echo "=== SENTRIX CONTROL PLANE END-TO-END TEST SUITE ==="

# 1. Container Health Checks
echo -e "\n[1/5] Checking Docker Container Status..."
docker ps --format '{{.Names}}' | grep -q "sentrix-prometheus" && pass "Prometheus container is running." || fail "Prometheus container is not running."
docker ps --format '{{.Names}}' | grep -q "sentrix-grafana" && pass "Grafana container is running." || fail "Grafana container is not running."

# 2. Endpoint Accessibility
echo -e "\n[2/5] Testing Port Accessibility & Health Endpoints..."
curl -sf http://localhost:9090/-/healthy > /dev/null && pass "Prometheus API accessible (:9090)" || fail "Prometheus unreachable."
curl -sf http://localhost:3000/api/health > /dev/null && pass "Grafana API accessible (:3000)" || fail "Grafana unreachable."
curl -sf http://localhost:8000/metrics > /dev/null && pass "Sentrix Evaluator API metrics live (:8000)" || fail "Sentrix API metrics endpoint unreachable."

# 3. Grafana Auto-Provisioning Validation
echo -e "\n[3/5] Verifying Grafana Auto-Provisioned Datasource & Dashboard..."
GRAFANA_AUTH=$(echo -n "admin:visdrift" | base64)
DS_CHECK=$(curl -s -H "Authorization: Basic ${GRAFANA_AUTH}" http://localhost:3000/api/datasources/name/Sentrix-Prometheus)
echo "$DS_CHECK" | grep -q '"type":"prometheus"' && pass "Datasource 'Sentrix-Prometheus' correctly provisioned." || fail "Grafana datasource missing."

DASH_CHECK=$(curl -s -H "Authorization: Basic ${GRAFANA_AUTH}" http://localhost:3000/api/search?query=Sentrix)
echo "$DASH_CHECK" | grep -q 'sentrix' && pass "Sentrix Telemetry Dashboard automatically loaded." || fail "Dashboard not provisioned."

# 4. Metric Streaming & Ingestion Verification
echo -e "\n[4/5] Executing Synthetic Metric Stream Pulse..."
python3 scripts/stream_metrics.py --count 10 > /dev/null 2>&1 &
STREAM_PID=$!
sleep 3
kill $STREAM_PID 2>/dev/null || true
pass "Metric stream generator executed successfully."

# 5. Prometheus Target Scraping Verification
echo -e "\n[5/5] Querying Prometheus Vector TSDB for Scraped Metrics..."
QUERY_RES=$(curl -s "http://localhost:9090/api/v1/query?query=sentrix_up")
echo "$QUERY_RES" | grep -q '"value":\[.*,"1"\]' && pass "Prometheus successfully scraping 'sentrix_up = 1'." || fail "Prometheus failed to scrape target metric."

echo -e "\n${GREEN}====================================================${NC}"
echo -e "${GREEN}  ALL SYSTEM INTEGRATION TESTS PASSED SUCCESSFULLY  ${NC}"
echo -e "${GREEN}====================================================${NC}"

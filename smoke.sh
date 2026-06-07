#!/usr/bin/env bash
#
# smoke.sh — end-to-end console check for nostr-merchant.
#
# Runs the agent's CLI surface top-to-bottom with section headers:
#   1. version        (local)
#   2. config-print   (local; secrets masked)
#   3. doctor         (spawns the MCP substrate — needs node + built dists)
#   4. budget         (local)
#   5. audit          (local)
#   6. ask            (real LLM call — read-only, so no sats can move)
#
# Each step runs independently; a failure is reported but does NOT abort the
# rest, so you always get the full picture. A summary table prints at the end.
#
# Usage:
#   ./smoke.sh                       # full run, default question, read-only
#   ./smoke.sh "list my last 3 txs"  # full run with a custom question
#   ./smoke.sh --no-ask              # skip the LLM call (no API tokens spent)
#   SMOKE_READ_ONLY=false ./smoke.sh # allow the ask to use paid tools (careful!)

set -uo pipefail

# Always operate from the project directory, wherever this script is invoked.
cd "$(dirname "$0")" || exit 1

# ---- options -----------------------------------------------------------------
RUN_ASK=1
QUESTION="What's my Lightning wallet balance?"
for arg in "$@"; do
  case "$arg" in
    --no-ask) RUN_ASK=0 ;;
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
    *) QUESTION="$arg" ;;
  esac
done
READ_ONLY="${SMOKE_READ_ONLY:-true}"

# ---- pretty helpers ----------------------------------------------------------
if [ -t 1 ]; then
  BOLD=$'\033[1m'; DIM=$'\033[2m'; GREEN=$'\033[32m'; RED=$'\033[31m'; CYAN=$'\033[36m'; RST=$'\033[0m'
else
  BOLD=""; DIM=""; GREEN=""; RED=""; CYAN=""; RST=""
fi

declare -a STEP_NAMES=()
declare -a STEP_RESULTS=()

section() { printf '\n%s━━━ %s ━━━%s\n' "$BOLD$CYAN" "$1" "$RST"; }

# run_step <label> <cmd...> : run a command, record pass/fail, never abort.
run_step() {
  local label="$1"; shift
  section "$label"
  printf '%s$ %s%s\n' "$DIM" "$*" "$RST"
  if "$@"; then
    STEP_NAMES+=("$label"); STEP_RESULTS+=("PASS")
  else
    local code=$?
    printf '%s[step exited %d]%s\n' "$RED" "$code" "$RST"
    STEP_NAMES+=("$label"); STEP_RESULTS+=("FAIL")
  fi
}

# ---- preflight ---------------------------------------------------------------
section "preflight"
if ! command -v uv >/dev/null 2>&1; then
  printf '%suv not found on PATH — install it first: https://docs.astral.sh/uv/%s\n' "$RED" "$RST"
  exit 1
fi
printf 'project: %s\n' "$(pwd)"
printf 'env file: %s %s\n' "$HOME/.nostr-merchant/.env" \
  "$( [ -f "$HOME/.nostr-merchant/.env" ] && echo '(found)' || echo "${RED}(MISSING — copy .env.example)${RST}")"
printf 'ask step: %s\n' "$( [ "$RUN_ASK" = 1 ] && echo "enabled (read-only=$READ_ONLY)" || echo 'skipped (--no-ask)')"

# ---- 1..5: local + substrate checks -----------------------------------------
run_step "1. version"      uv run nostr-merchant version
run_step "2. config-print" uv run nostr-merchant config-print
run_step "3. doctor"       uv run nostr-merchant doctor
run_step "4. budget"       uv run nostr-merchant budget
run_step "5. audit"        uv run nostr-merchant audit -n 10

# ---- 6: the real agent loop --------------------------------------------------
if [ "$RUN_ASK" = 1 ]; then
  section "6. ask"
  printf '%squestion: %s%s\n' "$DIM" "$QUESTION" "$RST"
  printf '%s$ AGENT_READ_ONLY=%s uv run nostr-merchant ask "%s"%s\n' "$DIM" "$READ_ONLY" "$QUESTION" "$RST"
  if AGENT_READ_ONLY="$READ_ONLY" uv run nostr-merchant ask "$QUESTION"; then
    STEP_NAMES+=("6. ask"); STEP_RESULTS+=("PASS")
  else
    printf '%s[ask failed — check the API key in ~/.nostr-merchant/.env and the doctor step above]%s\n' "$RED" "$RST"
    STEP_NAMES+=("6. ask"); STEP_RESULTS+=("FAIL")
  fi
fi

# ---- summary -----------------------------------------------------------------
section "summary"
fails=0
for i in "${!STEP_NAMES[@]}"; do
  if [ "${STEP_RESULTS[$i]}" = "PASS" ]; then
    printf '  %s✓%s %s\n' "$GREEN" "$RST" "${STEP_NAMES[$i]}"
  else
    printf '  %s✗%s %s\n' "$RED" "$RST" "${STEP_NAMES[$i]}"; fails=$((fails+1))
  fi
done
echo
if [ "$fails" -eq 0 ]; then
  printf '%sAll steps passed.%s\n' "$GREEN$BOLD" "$RST"; exit 0
else
  printf '%s%d step(s) failed.%s\n' "$RED$BOLD" "$fails" "$RST"; exit 1
fi

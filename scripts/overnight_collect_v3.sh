#!/usr/bin/env bash
# overnight_collect_v3.sh — Phase 2 large-scale collection (target: 20,000 articles)
#
# Current status (2026-03-02):
#   eLife  2,019 articles  → target +3,000 (2013-2022 historical data, diverse subjects)
#   PLOS   2,677 articles  → target +3,000 (2019-2020 gap, 2023 gap of 77 articles)
#   F1000  3,094 articles  → target +3,000 (2022-2024 completely missing)
#   PeerJ    245 articles  → target +2,000 (per-year collection to bypass cursor limit)
#   Nature    66 articles  → target +1,000 (2022-2024 expansion)
#
# Strategy:
#   Phase 1 (4 parallel streams): eLife + PLOS + F1000 + PeerJ — ~12-16 hours
#   Phase 2 (sequential): Nature — ~5 hours
#   Phase 3: rebuild_splits (elife + plos + f1000 + peerj + nature)
#
# Usage:
#   mkdir -p data/logs
#   nohup bash scripts/overnight_collect_v3.sh > data/logs/overnight_v3.log 2>&1 &
#   echo "PID: $!"
#
# Monitoring:
#   tail -f data/logs/overnight_v3.log
#   tail -f data/logs/v3_elife.log   # eLife detailed log
#   tail -f data/logs/v3_plos.log
#   tail -f data/logs/v3_f1000.log
#   tail -f data/logs/v3_peerj.log
#   tail -f data/logs/v3_nature.log

set -uo pipefail
# NOTE: set -e disabled. Individual collector failures should not terminate the entire script.

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p data/logs

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a data/logs/overnight_v3.log; }

# Safe wait function — handles gracefully if PID does not exist
wait_pid() {
    local pid="$1" name="$2"
    if kill -0 "$pid" 2>/dev/null; then
        log "  Waiting for $name (PID $pid)..."
        wait "$pid"
        local exit_code=$?
        if [ $exit_code -ne 0 ]; then
            log "  [WARN] $name (PID $pid) exited with code $exit_code"
        else
            log "  [OK] $name (PID $pid) completed"
        fi
    else
        log "  [WARN] $name (PID $pid) already finished"
    fi
}

# Current collection status display helper
show_counts() {
    log "  Current collection status:"
    for f in elife_v1.1.jsonl plos_v1.jsonl f1000_v1.jsonl peerj_v1.jsonl nature_v1.jsonl; do
        if [ -f "data/processed/$f" ]; then
            cnt=$(wc -l < "data/processed/$f")
            log "    $f: ${cnt} articles"
        fi
    done
}

log "======================================================"
log "=== overnight_collect_v3.sh started (target: 20,000 articles) ==="
log "======================================================"
show_counts

# ═══════════════════════════════════════════════════════════════
# PHASE 1 — 4 parallel stream collection
# ═══════════════════════════════════════════════════════════════
log ""
log "=== PHASE 1: Starting 4 parallel collection streams ==="
log ""

# ─────────────────────────────────────────────────────────────
# Stream 1: eLife — 2013-2022 historical data (diverse subjects)
# Target: +3,000 articles (current 2,019, almost all from 2024-2026)
# Strategy: oldest-first ordering to prioritize 2013-2022 articles
# ─────────────────────────────────────────────────────────────
(
    log "[eLife Stream] Starting: 2013-2022 historical data (target 3000 articles)"

    uv run python -m bioreview_bench.scripts.collect_elife \
        --max-articles 3000 \
        --start-date 2013-01-01 \
        --end-date 2022-12-31 \
        --order asc \
        --subjects genetics-genomics \
        --subjects evolutionary-biology \
        --subjects ecology \
        --subjects plant-biology \
        --subjects physics-of-living-systems \
        --subjects biochemistry-chemical-biology \
        --subjects cancer-biology \
        --subjects structural-biology-molecular-biophysics \
        --subjects computational-systems-biology \
        --subjects epidemiology-global-health \
        --subjects stem-cells-regenerative-medicine \
        --subjects cell-biology \
        --subjects developmental-biology \
        --subjects microbiology-infectious-disease \
        --subjects immunology-inflammation \
        --subjects neuroscience \
        --append \
        >> data/logs/v3_elife.log 2>&1

    ELIFE_NEW=$(wc -l < data/processed/elife_v1.1.jsonl)
    log "[eLife Stream] Completed: current ${ELIFE_NEW} articles (v1.1.jsonl)"
) &
ELIFE_PID=$!
log "  eLife Stream started (PID: $ELIFE_PID)"

# ─────────────────────────────────────────────────────────────
# Stream 2: PLOS — 2019-2020 missing range + 2023 gap fill
# Target: +3,000 articles
# Strategy: run two date ranges sequentially with --append
# ─────────────────────────────────────────────────────────────
(
    log "[PLOS Stream] Starting: 2019-2020 missing range (1500 articles)"

    # Round 1: 2019-2020 (completely missing range)
    uv run python -m bioreview_bench.scripts.collect_plos \
        --max-articles 1500 \
        --journals PLoSBiology \
        --journals PLoSGenetics \
        --journals PLoSCompBiol \
        --journals PLoSMedicine \
        --start-date 2019-01-01 \
        --end-date 2020-12-31 \
        --append \
        >> data/logs/v3_plos.log 2>&1

    PLOS_MID=$(wc -l < data/processed/plos_v1.jsonl)
    log "[PLOS Stream] Round 1 completed: current ${PLOS_MID} articles, starting 2023 gap collection"

    # Round 2: 2023 gap (expand from 77 articles)
    uv run python -m bioreview_bench.scripts.collect_plos \
        --max-articles 1500 \
        --journals PLoSBiology \
        --journals PLoSGenetics \
        --journals PLoSCompBiol \
        --journals PLoSMedicine \
        --start-date 2023-01-01 \
        --end-date 2023-12-31 \
        --append \
        >> data/logs/v3_plos.log 2>&1

    PLOS_NEW=$(wc -l < data/processed/plos_v1.jsonl)
    log "[PLOS Stream] Completed: current ${PLOS_NEW} articles"
) &
PLOS_PID=$!
log "  PLOS Stream started (PID: $PLOS_PID)"

# ─────────────────────────────────────────────────────────────
# Stream 3: F1000 — 2022-2024 completely missing range
# Target: +3,000 articles (currently no 2022, 2023, 2024 data)
# ─────────────────────────────────────────────────────────────
(
    log "[F1000 Stream] Starting: 2022-2024 missing range (target 3000 articles)"

    uv run python -m bioreview_bench.scripts.collect_f1000 \
        --max-articles 3000 \
        --start-date 2022-01-01 \
        --end-date 2024-12-31 \
        --append \
        >> data/logs/v3_f1000.log 2>&1

    F1K_NEW=$(wc -l < data/processed/f1000_v1.jsonl)
    log "[F1000 Stream] Completed: current ${F1K_NEW} articles"
) &
F1000_PID=$!
log "  F1000 Stream started (PID: $F1000_PID)"

# ─────────────────────────────────────────────────────────────
# Stream 4: PeerJ — per-year sequential collection (cursor limit bypass)
# Target: +2,000 articles (current 245)
# Strategy: collect up to 350 per year sequentially to avoid annual cursor limit
# Yield: ~53% (2018+ articles) → 350 x 7 years = 2,450 queries → ~1,300 new
# ─────────────────────────────────────────────────────────────
(
    log "[PeerJ Stream] Starting: per-year sequential collection (2019-2025)"

    for YEAR in 2019 2020 2021 2022 2023 2024 2025; do
        YEAR_START="${YEAR}-01-01"
        YEAR_END="${YEAR}-12-31"
        log "[PeerJ Stream] Collecting ${YEAR} (max 400 articles)..."

        uv run python -m bioreview_bench.scripts.collect_peerj \
            --max-articles 400 \
            --start-date "${YEAR_START}" \
            --end-date "${YEAR_END}" \
            --append \
            >> data/logs/v3_peerj.log 2>&1

        PEERJ_NOW=$(wc -l < data/processed/peerj_v1.jsonl 2>/dev/null || echo 0)
        log "[PeerJ Stream] ${YEAR} completed: current ${PEERJ_NOW} articles"
    done

    PEERJ_FINAL=$(wc -l < data/processed/peerj_v1.jsonl 2>/dev/null || echo 0)
    log "[PeerJ Stream] All years completed: final ${PEERJ_FINAL} articles"
) &
PEERJ_PID=$!
log "  PeerJ Stream started (PID: $PEERJ_PID)"

log ""
log "Phase 1: all 4 streams started. Waiting for completion..."
log "  eLife PID: $ELIFE_PID"
log "  PLOS  PID: $PLOS_PID"
log "  F1000 PID: $F1000_PID"
log "  PeerJ PID: $PEERJ_PID"
log ""

# Progress monitoring (every 10 minutes)
while kill -0 "$ELIFE_PID" 2>/dev/null || \
      kill -0 "$PLOS_PID"  2>/dev/null || \
      kill -0 "$F1000_PID" 2>/dev/null || \
      kill -0 "$PEERJ_PID" 2>/dev/null; do
    sleep 600  # 10 minute interval
    log "--- Phase 1 progress (10 min elapsed) ---"
    show_counts
done

# Explicit wait for all Phase 1 processes
wait_pid "$ELIFE_PID" "eLife"
wait_pid "$PLOS_PID"  "PLOS"
wait_pid "$F1000_PID" "F1000"
wait_pid "$PEERJ_PID" "PeerJ"

log ""
log "=== PHASE 1 COMPLETED ==="
show_counts

# ═══════════════════════════════════════════════════════════════
# PHASE 2 — Nature portfolio collection (PDF parsing, sequential)
# Prioritized by hit rate (from probe_nature_coverage.py, 2026-03-02):
#   NatComm:         92% hit rate (mandatory TPR since Nov 2022)
#   Comm E&E:        75% hit rate (mandatory TPR)
#   Comm Medicine:   67% hit rate (mandatory TPR)
#   Comm Biology:    42% hit rate
#   Comm Chemistry:  33% hit rate
#   Opt-in journals: 25-33% hit rate (lower priority)
# Target: +1,000 articles  (actual yield ~600-700 given mixed hit rates)
# ═══════════════════════════════════════════════════════════════
log ""
log "=== PHASE 2: Nature portfolio collection (target +1000, prioritized by hit rate) ==="
log ""

(
    # Round 1: Highest hit-rate mandatory journals (NatComm, CommE&E, CommMed)
    # Expected yield: ~80-90% of articles → ~450 usable from 500 tried
    log "[Nature] Round 1: NatComm + CommE&E + CommMed (2022-2024)"
    uv run python -m bioreview_bench.scripts.collect_nature \
        --max-articles 500 \
        --journals "Nature Communications" \
        --journals "Communications Earth and Environment" \
        --journals "Communications Medicine" \
        --start-date 2022-01-01 \
        --end-date 2024-12-31 \
        --append \
        >> data/logs/v3_nature.log 2>&1

    NAT_MID=$(wc -l < data/processed/nature_v1.jsonl 2>/dev/null || echo 0)
    log "[Nature] Round 1 completed: current ${NAT_MID} articles"

    # Round 2: Medium hit-rate journals (CommsBio, CommsChem, Nature flagship)
    # Expected yield: ~33-42% of articles → ~165 usable from 500 tried
    log "[Nature] Round 2: CommsBio + CommsChem + Nature (2022-2024)"
    uv run python -m bioreview_bench.scripts.collect_nature \
        --max-articles 500 \
        --journals "Communications Biology" \
        --journals "Communications Chemistry" \
        --journals "Nature" \
        --start-date 2022-01-01 \
        --end-date 2024-12-31 \
        --append \
        >> data/logs/v3_nature.log 2>&1

    NAT_FINAL=$(wc -l < data/processed/nature_v1.jsonl 2>/dev/null || echo 0)
    log "[Nature] Round 2 completed: final ${NAT_FINAL} articles"
) &
NATURE_PID=$!
log "  Nature Stream started (PID: $NATURE_PID)"

wait_pid "$NATURE_PID" "Nature"

log ""
log "=== PHASE 2 COMPLETED ==="
show_counts

# ═══════════════════════════════════════════════════════════════
# PHASE 3 — Split reconstruction
# ═══════════════════════════════════════════════════════════════
log ""
log "=== PHASE 3: rebuild_splits (5 sources -> v2 splits) ==="
log ""

uv run python scripts/rebuild_splits.py \
    --sources elife \
    --sources plos \
    --sources f1000 \
    --sources peerj \
    --sources nature \
    >> data/logs/v3_rebuild.log 2>&1

log "rebuild_splits completed"

# ═══════════════════════════════════════════════════════════════
# Final report
# ═══════════════════════════════════════════════════════════════
log ""
log "======================================================"
log "=== overnight_collect_v3.sh fully completed ==="
log "======================================================"
log ""
log "Final collection status (raw):"
for f in elife_v1.1.jsonl plos_v1.jsonl f1000_v1.jsonl peerj_v1.jsonl nature_v1.jsonl; do
    if [ -f "data/processed/$f" ]; then
        cnt=$(wc -l < "data/processed/$f")
        log "  $f: ${cnt} articles"
    fi
done

log ""
log "Split v2 status:"
if ls data/splits/v2/*.jsonl 2>/dev/null | head -1 | grep -q .; then
    for f in data/splits/v2/train.jsonl data/splits/v2/val.jsonl data/splits/v2/test.jsonl; do
        if [ -f "$f" ]; then
            cnt=$(wc -l < "$f")
            log "  $(basename $f): ${cnt} articles"
        fi
    done
fi

log ""
log "Log files:"
log "  Main:    data/logs/overnight_v3.log"
log "  eLife:   data/logs/v3_elife.log"
log "  PLOS:    data/logs/v3_plos.log"
log "  F1000:   data/logs/v3_f1000.log"
log "  PeerJ:   data/logs/v3_peerj.log"
log "  Nature:  data/logs/v3_nature.log"
log "  Rebuild: data/logs/v3_rebuild.log"
log ""
log "Completed at: $(date '+%Y-%m-%d %H:%M:%S')"

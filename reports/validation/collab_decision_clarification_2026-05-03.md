# COMOSA Decision Clarification (Workspace-Verified)

Date: 2026-05-03
Purpose: Clarify current project status to avoid misunderstanding before Phase 3 work.

## Executive Summary

Out of the 5 major decisions discussed, current workspace status is:
- Already decided and implemented: Decision 1, Decision 2, Decision 5.
- Still pending explicit research choice: Decision 3, Decision 4.

This note is based on direct inspection of data files, scripts, and validation reports in this repository.

## Decision-by-Decision Status

## 1) Leverage Operationalization (Option A / B / A+B)

Current factual status in workspace:
- The statement "Event_Dynamics_100ms.csv has no leverage column" is NOT correct.
- `leverage_proxy` already exists in:
  - data/processed/tardis/confounder_outputs/Event_Dynamics_100ms.csv
  - data/processed/tardis/confounder_outputs/Event_Dynamics_100ms_gridded.csv
- Current implementation corresponds to Option B logic (velocity x acceleration / realized_vol_50) in:
  - scripts/stage2_economics/09_produce_confounder_outputs.py

Practical interpretation:
- Baseline leverage proxy is already operationalized as Option B.
- If thesis framing requires a stronger financial-leverage claim, Option A (funding rate) can be added as an extension rather than claiming no leverage exists now.

## 2) Crash Threshold in Simulator

Current factual status in workspace:
- Threshold has already been calibrated and locked at 1.93% for rolling 10 ticks (1 second at 100ms).
- This is documented in:
  - reports/validation/phase2_crash_calibration_decision_2026-05-02.md
- Default in simulator is already updated:
  - scripts/stage2_economics/18_lob_mini_runner.py (`--crash-threshold-pct` default = 1.93)

Empirical reference (drop phase quantiles from current Event_Dynamics_100ms.csv):
- drop_from_local_pct: p50=0.032213, p90=0.166395, p95=0.250243, p99=0.526910
- drop_1s_pct: p50=0.079643, p90=0.343597, p95=0.536768, p99=1.238638

Practical interpretation:
- This decision is not pending anymore in codebase state.

## 3) Baseline Calibration Data (pre phase vs normal-market week)

Current factual status in workspace:
- Current anchor-building pipeline is event-centric (pre/drop/recovery/post from event windows).
- No separate normal-market week dataset is integrated yet for baseline calibration.

Practical interpretation:
- This is still a real research choice:
  - Keep pre-phase proxy (faster, but acknowledge bias), or
  - Add dedicated normal-market download for cleaner baseline.

## 4) DAG Benchmark Strategy (Kirilenko vs BTC-native split)

Current factual status in workspace:
- Plan text recommends BTC-native temporal hold-out logic.
- Current causal discovery script does not yet implement explicit temporal train/test split workflow.
- No dedicated validation artifact for such split is present yet.

Practical interpretation:
- This decision remains open and should be explicitly locked before Phase 3 claims are finalized.

## 5) Confirmation-Bias / Claim Framing

Current factual status in workspace:
- Plan already frames Phase 3 as confirmatory causal validation (not blind discovery).
- This framing is explicitly documented in Comosa-Plan-Rewritten.md.

Practical interpretation:
- Claim level is already aligned with a methodological benchmark contribution.
- Avoid rewording into "true BTC mechanism discovery" unless additional real-data external validation is added.

## Recommended Wording for Collaboration Thread

Use this exact short summary if needed:

"Workspace audit update (2026-05-03): leverage proxy is already present and implemented (Option B), crash threshold is already calibrated and locked at 1.93% in simulator defaults, and Phase-3 framing is already confirmatory in the rewritten plan. The two decisions still genuinely open are: (i) whether to add separate normal-market baseline data, and (ii) exact BTC-native temporal split protocol for DAG benchmark validation."

## Scope Note

This clarification only reflects current repository state and does not by itself lock future thesis framing decisions.

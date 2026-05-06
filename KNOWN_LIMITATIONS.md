# Known Limitations

## Phase 2 Canonical Tuned Legacy

Canonical config:

- `config/phase2_canonical_config.json`

Current known limitation:

- `min_price_fraction = 0.50` is a binding floor in the tuned legacy setup and must be treated as a structural regime boundary, not as an empirically calibrated lower bound.

Required reporting rules for any Phase 2 or Phase 3 result derived from this config:

- Report `raw crash rate` and `pre-floor crash rate` separately.
- Keep floor-touched runs analytically distinct from floor-clean runs.
- Do not describe `min_price_fraction = 0.50` as empirically anchored to BTC event bottoms.

Operational interpretation:

- The tuned legacy config remains the Phase 2 canonical configuration because its other core knobs have a clearer audit trail than the current cfgC family.
- The current runner includes a post-audit resilience-damping mechanism, so the canonical config explicitly fixes `resilience_min_damp = 1.0` to preserve the pre-2026-05-05 tuned-legacy behavior.
- cfgC outputs remain sensitivity and stress-test branches, not the main causal-analysis pipeline.

Until a dedicated floor-hit indicator is materialized in downstream analysis tables, use the per-run minimum close relative to the run start as the proxy needed to separate floor-touched from floor-clean runs.
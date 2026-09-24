# Notebooks

The analysis is deliberately script-first: everything in the README is produced
by a committed, testable code path rather than by notebook state.

- `python scripts/run_evaluation.py --source walmart` — decomposition,
  walk-forward comparison, posterior summary and the +20% scenario, all written
  to `reports/`.
- `streamlit run dashboard/app.py` — the same numbers, interactively.

If you want a scratch notebook for exploration, create it here; keep
`forecast/` and `scripts/` as the sources of truth for any number you cite.

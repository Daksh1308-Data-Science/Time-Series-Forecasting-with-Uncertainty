# Deploying to Streamlit Community Cloud

The dashboard deploys as-is. No build step, no secrets, no Docker.

## What Cloud runs

| Setting | Value | Why |
|---|---|---|
| Repository | `Daksh1308-Data-Science/Time-Series-Forecasting-with-Uncertainty` | must be **public** for the free tier |
| Branch | `main` | |
| Entrypoint file path | `dashboard/app.py` | Community Cloud accepts entrypoints in any subdirectory; the path is typed into the deploy form because its repo suggestions are incomplete |
| Dependencies | root `requirements.txt` | Cloud accepts requirements at the root or beside the entrypoint |
| Python version | **3.12** (Advanced settings) | Cloud's default and best-tested; 3.14 is new enough that Cloud has little track record. Verified locally on 3.11 and 3.14 |
| Config | `.streamlit/config.toml` (repo root) | must be at the root even though the entrypoint is in a subdirectory |
| Secrets | **none** | the processed series is committed, so the app never needs Kaggle credentials |

## One-time deploy

1. Go to <https://share.streamlit.io> → **Create app** (top right).
2. "Do you already have an app?" → **Yup, I have an app**.
3. Fill in repository, branch `main`, and file path `dashboard/app.py`
   (or click **Paste GitHub URL** and paste the link to `dashboard/app.py`).
4. **Advanced settings** → set Python version to 3.12. Leave **Secrets empty**.
5. Optionally set a memorable **App URL** subdomain.
6. **Deploy.** Most apps are live in a few minutes.

Pushes to `main` redeploy automatically. Code changes appear immediately;
dependency changes take a few minutes to reinstall.

## What is deliberately not installed

`requirements.txt` contains the core dependencies only. `pymc` and `prophet`
live in `requirements-optional.txt` and are **not installed on Cloud**:

- **Prophet** compiles a Stan model on first fit. That needs a C++ toolchain in
  the container, so it is slow and fragile on a free serverless tier. Its
  results were computed offline (see below) instead.
- **PyMC** (plus arviz/xarray) is heavy to import and fit on 1 CPU / ~1 GB.
  The dashboard's default engine is the exact conjugate posterior, which is
  instant and deterministic.

Neither absence is a bug, and neither breaks the app — see the next section.

## How the app behaves on Cloud

- **Prophet tab** — explains that Prophet is unavailable and why, in
  platform-neutral language. It never suggests a Windows CmdStan command to a
  Linux visitor.
- **Bayesian engine** — the sidebar's NUTS option is labelled *local only*. If
  selected anyway, `resolve_engine` falls back to the exact engine and the tab
  says so; it does not crash.
- **Comparison tab** — defaults to **Measured results**, read from the committed
  `reports/comparison_table.csv` and `reports/metrics.json`. This is the full
  three-model comparison **including Prophet**, because those numbers were
  produced by `scripts/run_evaluation.py` on the full pipeline. Switch the radio
  to **Live re-run** to refit in-session (Bayesian + seasonal-naive only on
  Cloud). If the sidebar settings differ from how the table was measured, a
  warning says so instead of quietly answering a different question.

### Keeping the measured view truthful

Because the deployed app serves committed numbers, refresh them whenever the
models change:

```bash
python scripts/run_evaluation.py --source walmart
git add reports/ && git commit -m "refresh measured results" && git push
```

## Local parity

Cloud's working directory is always the repository root, so run the same way:

```bash
streamlit run dashboard/app.py     # from the repo root
```

To reproduce Cloud's dependency set exactly, use a venv with only
`requirements.txt` — no `pymc`, no `prophet` — and the app behaves identically.

## Free-tier expectations

- The app **sleeps when inactive**; the first request after a pause is slow
  while it wakes. That is normal, not an error.
- ~1 CPU and ~1 GB RAM. The instant exact engine keeps it comfortable; avoid
  selecting the NUTS engine in the sidebar.
- **Cloud logs are visible only to users with write access to the repository** —
  that is where you read tracebacks if a deploy fails.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| "Unable to determine entrypoint" | file path typo — use `dashboard/app.py`, not `/dashboard/app.py` |
| Dependency resolution error in the log | a transitive pin conflicts with Python 3.12; add a constraint to `requirements.txt` and redeploy |
| First load hangs, then works | the app was asleep; normal |
| Prophet tab shows an explanation | expected on Cloud — see above |
| Comparison tab warns "settings differ" | you changed folds/horizon/engine; the table is the committed run, not a re-run |
| "Committed measured results unavailable" | `reports/comparison_table.csv` missing — run the evaluation script and push |

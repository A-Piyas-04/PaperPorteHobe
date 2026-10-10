# Local development

Updated 2026-10-10. Run from the repository root in PowerShell.

## Start the existing app

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open http://localhost:8501. The app uses the configured data release; it does not need a new pipeline run for UI changes. Restart Streamlit after Python edits because file watching is disabled in `.streamlit/config.toml`.

If the environment is missing:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[app,dev]"
```

Install the ML extras for bundles that require MiniLM/UMAP/HDBSCAN. A missing query embedding model falls back to keyword search and shows a notice.

## Rebuild the workspace frontend

```powershell
Set-Location app/components/explorer/frontend
npm ci
npm run build
Set-Location ../../../..
```

The built `frontend/dist` files are committed so running the app does not require Node. Include regenerated assets with frontend changes.

## Verify

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m mypy
```

The workspace also has browser regression tests:

```powershell
Set-Location app/components/explorer/frontend
npm ci
npm run build
npx playwright install chromium
npm test
Set-Location ../../../..
```

To use an already installed Microsoft Edge locally, set
`$env:PLAYWRIGHT_CHANNEL='msedge'` before `npm test`. CI installs Chromium.

Try the tour (including Skip, Back and replay), submit a search, save and remove papers, export the reading list, and open the Workspace. In the Workspace, use the **Map**: from the overview click an area bubble, click a paper to see its neighbourhood and details, use *Focus this paper's connections* / *+ Expand*, and the zoom / Fit / Overview controls; confirm hovering only shows a tooltip and never changes the selection or camera, and that saving a paper keeps the selection and viewport. Switch to the **List** view for keyboard selection and filtering, select related papers, and repeat at a narrow phone width (the map uses a bottom-sheet for details). The reading list and tour state last for the current Streamlit session. Export before closing or reloading.

## Data and deployment

Without a bundle, the setup page offers an explicitly labelled synthetic demo. For real data, use [the runbook](runbook.md). The app reads a versioned release selected by configuration, so inspecting only `data/processed/meta.json` may not describe what it serves.

This UI upgrade does not deploy, rebuild the research corpus, change production validation gates, or prove the launch criteria in the historical production plan. Dataset quality and operational readiness need their own checks before launch.

# Next Instructions — getting ScholarGrid fully running

The code, pipeline, and web app are done. Below is exactly what to do next, in
order, and **where** each step happens (🖥️ terminal · 🌐 website · 📝 file
edit). Most people need only Steps 1–4.

> **Project root:** `E:\Projects\My Apps\PaperPorteHobe`
> **Shell:** Windows PowerShell (Cursor's default terminal). All commands below
> are PowerShell and are run from the project root.

### Where the project stands right now

| Item | State |
|---|---|
| Python | 3.12.2 via the `py` launcher (`python3` does **not** work on Windows) |
| `.venv` | ✅ created and populated (core stack + MiniLM / UMAP / HDBSCAN) |
| Real arXiv data | ✅ built in `data/processed/` — 1,275 papers · 26 clusters · 12 leads · precision@k = 0.88 · using the SRS baseline backends (MiniLM · UMAP · HDBSCAN) |
| Authors / citations | ✅ authors for every paper (arXiv); OpenAlex matched 643 papers (citations are ~0 because the papers are a week old) |
| Paper graph | ✅ 5,977 similarity links in `edges.json` / `neighbors.json` |
| Explore view | ✅ React component built into `app/components/explorer/frontend/dist` (committed — no Node needed to run or deploy) |
| Validation report | ✅ `reports/validation_report.md` up to date |
| GitHub | ✅ repo pushed to `origin` → <https://github.com/A-Piyas-04/PaperPorteHobe> |
| Git LFS | installed (needed only for Step 5, Track B) |

> **➡️ Where to pick up now: [Step 4](#step-4----️-run-the-web-app-locally).**
> Steps 1–3 are already done. Steps 5–7 are deployment and refresh — do them
> only if/when you need them.

---

## Step 1 — 🖥️ Activate the environment and install (once)

> ✅ **Done.** The venv is already set up and populated. Just reactivate it in
> every new terminal:
>
> ```powershell
> cd "E:\Projects\My Apps\PaperPorteHobe"
> .\.venv\Scripts\Activate.ps1
> ```
>
> Your prompt should start with `(.venv)`. Inside the venv, use `python`
> (not `python3`, not `py`).

<details>
<summary>If you ever need to recreate it from scratch</summary>

```powershell
Remove-Item -Recurse -Force .venv
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

- `source .venv/bin/activate` is the Linux/macOS command — it fails in
  PowerShell. Use `.\.venv\Scripts\Activate.ps1`.
- If PowerShell says *"running scripts is disabled on this system"*, run once:
  `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then activate again.

</details>

---

## Step 2 — 🖥️ (Recommended, optional) Install the baseline ML methods

> ✅ **Done.** `sentence-transformers`, `umap-learn`, and `hdbscan` are already
> installed, and the pipeline is actively using all three (see `meta.json` →
> `backends`).

<details>
<summary>If you ever need to reinstall</summary>

```powershell
pip install sentence-transformers umap-learn hdbscan
```

This pulls in PyTorch (~a few hundred MB). The first pipeline run also
downloads the `all-MiniLM-L6-v2` model (~90 MB) once.

Skip this and ScholarGrid still works using the built-in fallbacks (TF-IDF,
PCA, DBSCAN/KMeans).

</details>

---

## Step 3 — 🖥️ Build (or rebuild) the data

> ✅ **Done.** The current bundle in `data/processed/` is the SRS-baseline
> build (MiniLM + UMAP + HDBSCAN) with **1,275 papers, 26 clusters, 12
> investigation leads, precision@k = 0.88, silhouette = 0.51**.

Re-run only if you change `configs/config.yaml` or want fresher papers:

```powershell
python pipeline\run_pipeline.py
```

- The raw harvest is cached in `data\raw\arxiv_harvest.csv`, so re-running
  **reuses the same papers** and only redoes the analysis.
- Citations, references and venue come from OpenAlex (`enrich:` in the
  config), cached in `data\raw\openalex.json`. Matches refresh weekly and misses
  are retried daily; if OpenAlex is unreachable the run simply continues
  without citations. Set `enrich.openalex: false` to skip it, and put your email
  in `enrich.mailto` for faster, politer access.
- The paper graph links each paper to its `graph.k_neighbors` most similar
  papers; pairs that cite the same works get a boost once references exist.
- Runtime: ~2–3 minutes on this machine.
- When it finishes you'll see `papers=… clusters=… leads=… precision@k=…` and
  an updated [`reports/validation_report.md`](../reports/validation_report.md).
- Check which methods were actually used in `data\processed\meta.json` →
  `"backends"`.

📝 To change the corpus, edit [`configs/config.yaml`](../configs/config.yaml):
`data.date_end`, `data.max_papers`, `data.arxiv_queries` — then force a fresh
harvest (see Step 7).

> **Know this about the current corpus:** the arXiv harvest takes the *newest*
> papers per category, so 1,275 papers only spans about one week
> (2026-09-25 → 2026-10-02). Clusters and search are fine, but the **growth**
> numbers (3/6/12-month windows) need a longer span. For meaningful trends,
> raise `data.max_papers` substantially (e.g. `8000`, slower harvest) or use the
> Kaggle snapshot (`data.source: kaggle`, see README).

> **Clustering tuning (`configs/config.yaml → cluster`).** HDBSCAN's
> `hdbscan_cluster_selection_method: leaf` (the default) extracts fine-grained
> topic clusters — fits a landscape view. Switch to `eom` if you want fewer,
> larger clusters. If HDBSCAN or DBSCAN ever yields fewer than
> `fallback_min_clusters`, the pipeline automatically falls back to a
> silhouette-selected KMeans.

**Offline alternative:** with no data at all, the app's first screen offers a
**"Load demo data"** button that builds a small synthetic landscape in seconds.
(You don't need it — you already have real data.)

---

## Step 4 — 🖥️ Run the web app locally ← **start here**

```powershell
streamlit run app\streamlit_app.py
```

The app runs headless, so it **won't open a browser by itself** — 🌐 open
<http://localhost:8501> manually. Stop it with `Ctrl+C`.

**Using it:**

- **Search** — type a topic or click an example; results show authors,
  citations and the area each paper belongs to. *See the graph* opens the
  matches in Explore.
- **Explore** — three panes: a paper list on the left, the paper graph in the
  middle, details on the right. Hover a paper (in the list or the graph) to
  light up its similar papers; click to select it and read the abstract, open
  arXiv/PDF, or jump to similar papers. Search, area and category filters sit
  in the top bar. `Esc` clears the selection, *Fit* resets the view.
- **Areas / Trends / Leads / About** — browse topics, activity, sparse
  neighbourhoods and how it all works.

✅ At this point ScholarGrid is fully functional on your machine.

### Editing the Explore view (only if you change the React code)

The Explore page is a custom Streamlit component in
`app\components\explorer\frontend` (Vite + React + TypeScript + sigma.js).
The built bundle in `frontend\dist` is committed, so you only need Node.js
(v18+) when you edit it.

```powershell
cd app\components\explorer\frontend
npm install          # once
npm run dev          # terminal 1: live dev server on http://localhost:5173
```

In a second terminal, from the project root, point Streamlit at the dev server:

```powershell
$env:SCHOLARGRID_DEV = "1"
streamlit run app\streamlit_app.py
```

Edits now hot-reload inside the app. When you're done, build and commit:

```powershell
cd app\components\explorer\frontend
npm run build        # type-checks, then writes dist\
cd ..\..\..\..
Remove-Item Env:SCHOLARGRID_DEV
git add app\components\explorer\frontend\dist
```

---

## Step 5 — 🌐🖥️ Deploy a public web app (Hugging Face Spaces)

This is the deployment target named in the SRS. Two tracks — pick one.

### One-time setup (both tracks)

1. 🌐 Create a free account at <https://huggingface.co/join>.
2. 🌐 Create an **access token** with **write** permission at
   <https://huggingface.co/settings/tokens>. When `git push` asks for a
   password, paste this token (your username is your HF username).
3. 🌐 Click **New → Space**, give it a name, choose **Streamlit** as the SDK
   (if Streamlit isn't listed, pick **Docker → Streamlit**), and create it.
4. 📝 Add this metadata block to the **very top** of the project's
   `README.md` (your push replaces the Space's README, so it must live in your
   repo). `python_version` matters: the pinned `numpy`/`scipy` need Python ≥ 3.11.

   ```yaml
   ---
   title: ScholarGrid
   emoji: 🔭
   colorFrom: blue
   colorTo: indigo
   sdk: streamlit
   sdk_version: 1.46.1
   python_version: "3.12"
   app_file: app.py
   pinned: false
   ---
   ```

   Commit it:

   ```powershell
   git add README.md
   git commit -m "Add Hugging Face Space metadata"
   git push origin main
   ```

5. 🖥️ Add the Space as a second remote (replace `USER`/`SPACE`):

   ```powershell
   git remote add space https://huggingface.co/spaces/USER/SPACE
   ```

### Track A — Quick public demo (code only)

```powershell
git push --force space main
```

`--force` is needed once because the new Space has its own initial commit.
`data/` and `reports/` are gitignored, so only code is pushed. 🌐 Wait for the
build (*Logs* tab), open the Space, and click **"Load demo data."** (The demo
rebuilds whenever the container restarts — fine for a demo.)

### Track B — Public app with your real arXiv landscape

Upload the precomputed artifacts too, so the Space opens straight onto real
data. Do this on a separate **deploy branch** so the large files never end up
on GitHub's `main`.

```powershell
git checkout -b hf-deploy
git lfs install
git lfs track "*.npy" "*.pkl" "data/processed/papers.csv"
git add .gitattributes
git add -f data/processed reports
git commit -m "ScholarGrid v1 + precomputed landscape"
git push --force space hf-deploy:main
git checkout main
```

🌐 Open the Space — it loads your real landscape with no button press.

> - LFS is required: Hugging Face rejects binary files that aren't in LFS, and
>   the TF-IDF model (`data/processed/embedder/tfidf.pkl`) is ~37 MB.
> - If you did Step 2 and rebuilt (Step 3), the bundle is much smaller (MiniLM
>   stores no large model file). Add `sentence-transformers` to
>   `requirements.txt` on the deploy branch so the Space can embed queries the
>   same way.
> - Don't `git push origin hf-deploy` — keep that branch for the Space only.

---

## Step 6 — 🖥️ Keep GitHub up to date

The repo is already on GitHub (`origin`). After any change:

```powershell
git add -A
git commit -m "Describe your change"
git push origin main
```

Generated data stays local (gitignored) — that's intended.

---

## Step 7 — 🖥️ (Optional, later) Refresh the data

To pull newer papers, update `data.date_end` in `configs/config.yaml` to today,
delete the cache to force a fresh harvest, and re-run:

```powershell
Remove-Item data\raw\arxiv_harvest.csv
python pipeline\run_pipeline.py
```

If you deployed Track B, update the live app:

```powershell
git checkout hf-deploy
git merge main
git add -f data/processed reports
git commit -m "Refresh landscape"
git push space hf-deploy:main
git checkout main
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Python was not found` when running `python3` | On Windows use `py` (outside the venv) or `python` (inside the activated venv). Optionally disable the Store aliases: *Settings → Apps → Advanced app settings → App execution aliases*. |
| `source : The term 'source' is not recognized` | That's the Linux command. Use `.\.venv\Scripts\Activate.ps1`. |
| `running scripts is disabled on this system` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then activate again. |
| `ModuleNotFoundError` (e.g. `streamlit`, `yaml`) | The venv isn't active or Step 1 wasn't run — activate it and `pip install -r requirements.txt`. |
| App says *"No precomputed artifacts found."* | Run Step 3, or click **Load demo data**. |
| `meta.json` still shows `tfidf` / `pca` / `kmeans` after Step 2 | Install into the **activated** venv, then re-run Step 3. |
| Harvest returns 0 papers | `data.date_end` is before the newest papers — set it to today's date. |
| Growth numbers look flat/meaningless | The corpus spans ~1 week — see the note in Step 3. |
| Explore says *"The explorer component has not been built yet"* | `frontend\dist` is missing — run `npm install` and `npm run build` in `app\components\explorer\frontend`. |
| Explore stays blank with `SCHOLARGRID_DEV=1` | The dev server isn't running — start `npm run dev`, or `Remove-Item Env:SCHOLARGRID_DEV` to use the built bundle. |
| Every paper shows 0 citations | Expected for week-old papers; node size then uses the number of similar papers. Citations appear as OpenAlex catches up (re-run Step 3 after a week or two). |
| Space build fails on `numpy`/`scipy` install | Add `python_version: "3.12"` to the README metadata (Step 5). |
| Space push rejected for binary/large files | Use Git LFS (Track B) or deploy Track A. |
| `git push space` asks for a password | Use your Hugging Face **access token**, not your account password. |

That's everything. After Step 4 it's usable locally; after Step 5 it's a public
web app.

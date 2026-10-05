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
| `.venv` | created, but **no packages installed yet** |
| Real arXiv data | already built in `data/processed/` (1,275 papers, 9 clusters, 12 leads) using the **fallback** backends (TF-IDF · PCA · KMeans) |
| Validation report | `reports/validation_report.md` exists |
| GitHub | repo pushed to `origin` → <https://github.com/A-Piyas-04/PaperPorteHobe> |
| Git LFS | installed (needed only for Step 5, Track B) |

---

## Step 1 — 🖥️ Activate the environment and install (once)

The venv already exists, so just activate it and install:

```powershell
cd "E:\Projects\My Apps\PaperPorteHobe"
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Your prompt should now start with `(.venv)`. Inside the venv, use `python`
(not `python3`, not `py`).

> **Notes**
> - `source .venv/bin/activate` is the Linux/macOS command — it fails in
>   PowerShell. Use `.\.venv\Scripts\Activate.ps1`.
> - If PowerShell says *"running scripts is disabled on this system"*, run once:
>   `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then activate again.
> - Re-activate the venv every time you open a new terminal.
> - If you ever need to recreate it: `Remove-Item -Recurse -Force .venv` then
>   `py -3.12 -m venv .venv`.

---

## Step 2 — 🖥️ (Recommended, optional) Install the baseline ML methods

These give sharper clusters and better search (SRS baseline: MiniLM embeddings,
UMAP, HDBSCAN). The pipeline auto-detects them — no code change needed. Your
Python 3.12 is supported.

```powershell
pip install sentence-transformers umap-learn hdbscan
```

> This pulls in PyTorch (~a few hundred MB). The first pipeline run also
> downloads the `all-MiniLM-L6-v2` model (~90 MB) once.

Skip this and ScholarGrid still works using the built-in fallbacks (which is
what your current data was built with).

---

## Step 3 — 🖥️ Build (or rebuild) the data

You **already have** a real-arXiv landscape, so this step is optional. Run it if
you did Step 2 (to rebuild with the better methods) or changed the config:

```powershell
python pipeline\run_pipeline.py
```

- The raw harvest is cached in `data\raw\arxiv_harvest.csv`, so re-running
  **reuses the same papers** and only redoes the analysis (no network).
- When it finishes you'll see `papers=… clusters=… leads=… precision@k=…` and
  an updated [`reports/validation_report.md`](../reports/validation_report.md).
- Check which methods were actually used in `data\processed\meta.json` →
  `"backends"` (e.g. `"embedding": "minilm"` instead of `"tfidf"`).

📝 To change the corpus, edit [`configs/config.yaml`](../configs/config.yaml):
`data.date_end`, `data.max_papers`, `data.arxiv_queries` — then force a fresh
harvest (see Step 7).

> **Know this about the current corpus:** the arXiv harvest takes the *newest*
> papers per category, so 1,500 papers only spans about one week
> (2026-09-25 → 2026-10-02). Clusters and search are fine, but the **growth**
> numbers (3/6/12-month windows) need a longer span. For meaningful trends,
> raise `data.max_papers` substantially (e.g. `8000`, slower harvest) or use the
> Kaggle snapshot (`data.source: kaggle`, see README).

**Offline alternative:** with no data at all, the app's first screen offers a
**"Load demo data"** button that builds a small synthetic landscape in seconds.
(You don't need it — you already have real data.)

---

## Step 4 — 🖥️ Run the web app locally

```powershell
streamlit run app\streamlit_app.py
```

The app runs headless, so it **won't open a browser by itself** — 🌐 open
<http://localhost:8501> manually. Stop it with `Ctrl+C`.

**Using it:** type a topic (or click an example) → the map highlights matches →
click or lasso points to inspect papers → open any area from *Research areas* →
expand *Growth*, *Investigation leads*, *How it works*, *Limitations* at the
bottom. Every paper links to arXiv.

✅ At this point ScholarGrid is fully functional on your machine.

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
| Space build fails on `numpy`/`scipy` install | Add `python_version: "3.12"` to the README metadata (Step 5). |
| Space push rejected for binary/large files | Use Git LFS (Track B) or deploy Track A. |
| `git push space` asks for a password | Use your Hugging Face **access token**, not your account password. |

That's everything. After Step 4 it's usable locally; after Step 5 it's a public
web app.

# Next Instructions — getting ScholarGrid fully running

My part (the code, pipeline, and web app) is done and tested. Below is exactly
what **you** do next, in order, and **where** each step happens
(🖥️ terminal · 🌐 website · 📝 file edit). Most people need only Steps 1–4.

> **Project root:** everything runs from `/home/amt/snap/RFE`.

---

## Step 1 — 🖥️ Create the environment and install (once)

```bash
cd /home/amt/snap/RFE
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

> **Python version note.** This installs the core stack (works on any recent
> Python). The *optional* higher-quality backends in Step 2 need **Python 3.11
> or 3.12** — they don't yet have wheels for Python 3.14. Check with
> `python3 --version`. If you're on 3.14 and want the better methods, install
> 3.12 and recreate the venv with `python3.12 -m venv .venv`.

---

## Step 2 — 🖥️ (Recommended, optional) Install the baseline ML methods

These give sharper clusters and better search. The app auto-detects them — no
code change needed.

```bash
pip install sentence-transformers umap-learn hdbscan
```

Skip this and ScholarGrid still works using the built-in fallbacks.

---

## Step 3 — 🖥️ Build the data (pick ONE)

**Option A — Real arXiv data (recommended).** Harvests ~1,500 recent CS papers
(cached afterwards, so it only hits the network once).

```bash
python pipeline/run_pipeline.py
```

- 📝 To change the corpus, edit [`configs/config.yaml`](configs/config.yaml):
  `data.date_end` (set it to today's date for the freshest papers),
  `data.max_papers`, and `data.arxiv_queries`.
- When it finishes you'll see `papers=… clusters=… precision@k=…` and a report
  in [`reports/validation_report.md`](reports/validation_report.md).

**Option B — Offline demo.** Skip this step entirely and use the **“Load demo
data”** button on the app's first screen (Step 4). Builds a small synthetic
landscape in a few seconds, no internet needed.

---

## Step 4 — 🖥️ Run the web app locally

```bash
streamlit run app/streamlit_app.py
```

Then 🌐 open the URL it prints (default <http://localhost:8501>).

**Using it:** type a topic (or click an example) → the map highlights matches →
click or lasso points to inspect papers → open any area from *Research areas* →
expand *Growth*, *Investigation leads*, *How it works*, *Limitations* at the
bottom. Every paper links to arXiv.

✅ At this point ScholarGrid is fully functional on your machine.

---

## Step 5 — 🌐🖥️ Deploy a public web app (Hugging Face Spaces)

This is the deployment target named in the SRS. Two tracks — pick one.

### Track A — Quick public demo (no data to upload)

1. 🌐 Go to <https://huggingface.co/join> and create a free account.
2. 🌐 Click **New → Space**. Set **SDK = Streamlit**, give it a name, create it.
3. 🌐 Open the Space's **Files** tab → edit **`README.md`** so the top metadata
   block reads:
   ```
   ---
   title: ScholarGrid
   emoji: 🔭
   sdk: streamlit
   app_file: app.py
   pinned: false
   ---
   ```
4. 🖥️ Push the code (replace `USER`/`SPACE`):
   ```bash
   cd /home/amt/snap/RFE
   git init && git add -A && git commit -m "ScholarGrid v1"
   git remote add space https://huggingface.co/spaces/USER/SPACE
   git push space main
   ```
   > `data/` is gitignored, so only code is pushed — that's intended here.
5. 🌐 Wait for the Space to build, then open it and click **“Load demo data.”**
   (The demo landscape rebuilds per container restart — fine for a demo.)

### Track B — Public app with your real arXiv landscape

Same as Track A, but also upload the precomputed artifacts so the app loads
instantly with real data.

1. Do Track A steps 1–3.
2. 🖥️ Build the data locally first (Step 3, Option A).
3. 🖥️ Store the large artifacts with Git LFS and force-add them (they're
   normally gitignored):
   ```bash
   cd /home/amt/snap/RFE
   git lfs install
   git lfs track "*.npy" "*.pkl" "data/processed/papers.csv"
   git add .gitattributes
   git add -f data/processed reports
   git commit -m "ScholarGrid v1 + precomputed landscape"
   git remote add space https://huggingface.co/spaces/USER/SPACE
   git push space main
   ```
   > Need Git LFS first? Install from <https://git-lfs.com> (`git lfs install`).
4. 🌐 Open the Space — it loads your real landscape with no button press.

> **Tip:** Track B is smaller and faster if you did Step 2 with
> `sentence-transformers` (the MiniLM embedder stores no large matrix; the
> model is fetched on the Space at build time).

---

## Step 6 — 🌐🖥️ (Optional) Put the source on GitHub

```bash
cd /home/amt/snap/RFE
git init            # skip if already a repo from Step 5
gh repo create scholargrid --public --source=. --push     # needs the GitHub CLI
# or: create an empty repo on github.com, then:
# git remote add origin https://github.com/USER/scholargrid.git && git push -u origin main
```

---

## Step 7 — 🖥️ (Optional, later) Refresh the data

Re-run the pipeline whenever you want newer papers. Delete the cache to force a
fresh harvest:

```bash
rm -f data/raw/arxiv_harvest.csv
python pipeline/run_pipeline.py
```

If you deployed Track B, repeat the Step 5B `git add -f … && git push` to update
the live app.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| App says *“No precomputed artifacts found.”* | Do Step 3, or click **Load demo data**. |
| `pip install sentence-transformers/umap/hdbscan` fails | You're likely on Python 3.14 — use 3.11/3.12 (Step 1 note), or skip Step 2. |
| Harvest returns 0 papers | `data.date_end` in the config is before today — set it to today's date. |
| Space build fails on big files | Use Git LFS (Step 5B) or deploy Track A. |
| Want to force fresh data | `rm data/raw/arxiv_harvest.csv` then re-run the pipeline. |

That's everything. After Step 4 it's usable locally; after Step 5 it's a public
web app.

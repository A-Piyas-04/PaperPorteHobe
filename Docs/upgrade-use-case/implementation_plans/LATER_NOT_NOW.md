# Later, not now: deliberately deferred items

> **Status:** NOT part of the current implementation series (plans 00–11).
> Do **not** implement anything in this file while working through 00–11. An item moves into an active plan
> only when this file is edited to say so.

The current series has two hard constraints:

1. **$0 spend.** No paid API, paid hosting, paid tool or paid credits. The only AI provider used by default is
   the Google **Gemini API free tier** (Flash / Flash-Lite), with optional **free** cloud fallbacks (Groq free
   tier, OpenRouter `:free` models).
2. **Laptop-light.** The product is a website that visitors use in their browser. The developer's laptop only
   writes code and runs tests and tiny smoke checks. Heavy work runs on free cloud machines (GitHub Actions for
   the pipeline, Hugging Face Spaces for the app). See `00_INDEX_READ_FIRST.md` §4.7.

Everything below breaks one of those constraints, so it is parked here with a short note on how to add it later.

---

## 1. Local AI models (Ollama)

**What:** run open models (Qwen, Llama, Gemma…) on the developer's PC through Ollama's OpenAI-compatible server
at `http://localhost:11434/v1`, as an unlimited fallback for bulk extraction.

**Why deferred:**
- Visitors use the app in a browser on Hugging Face Spaces. Neither their browsers nor the hosted app can reach a
  model running on the developer's PC, so Ollama could only ever help the offline pipeline, never users.
- It is resource-heavy: RAM/GPU, power, and the PC tied up for hours.
- Small local models are weaker than Gemini Flash at structured extraction.

**What we do instead now:** Gemini free tier, optional Groq / OpenRouter free fallbacks, request packing, the
disk cache, a narrow `scope`, and the daily GitHub Actions continuation (plan 11 §9.2).

**How to add it later (no new backend code needed):** plan 01's `openai_compat` backend already speaks the
same protocol.
1. Install Ollama and `ollama pull <model>`.
2. Add a provider in `configs/config.yaml`:
   `ollama: {kind: openai_compat, base_url: "http://localhost:11434/v1", api_key_env: "", json_mode: schema, quota_tz: UTC}`
3. Let `OpenAICompatBackend` accept an empty `api_key_env` (localhost needs no key; pass a dummy key string).
4. Append `"ollama:<model>"` to the `extract` route **only**, never to online tasks.
5. Run plan 09's model-comparison harness (§8) to confirm the quality before relying on it.

---

## 2. Paid AI providers and paid model tiers

**What:**
- xAI **Grok** (no reliable free API tier; needs purchased credits).
- Gemini **Pro** / Gemini 4 models through the API (not on the free tier), or switching the Google Cloud project
  to the paid tier (enabling billing).
- The OpenRouter one-time credit purchase that raises the `:free` daily limit.
- Any other paid API (Claude, OpenAI, …).

**Why deferred:** $0 budget. Note that a consumer **Gemini Pro / Google AI Pro subscription does not include API
quota**. The API is billed separately; the free API tier covers Flash / Flash-Lite only.

**Rule for now:** never attach a billing account to the Google Cloud project behind the AI Studio key. Without
billing, exceeding quota only returns HTTP 429 and the app falls back to heuristics. With billing on, the same
traffic is charged.

**How to add it later (budget idea: at most $1–2/month):** plan 01 was built for this.
1. A new backend class in `scholargrid/llm/backends/` (or reuse `openai_compat` for OpenAI-compatible APIs such
   as xAI: `base_url: "https://api.x.ai/v1"`, `api_key_env: XAI_API_KEY`).
2. A `prices` entry for the route, so the usage ledger reports real cost.
3. Put the paid route **last** in the judgment-heavy task routes (`synthesize_gap`, `plan`), and cap it with
   `llm.budget.offline_max_requests` / `online_daily_requests`.
4. Add its key pattern to the secret scrubber (plan 11 §7), e.g. `xai-[0-9A-Za-z]{20,}`.

---

## 3. Paid hosting (own server + domain)

**What:** `Docs/deployment.md` **Option A**: a rented server (Hetzner about €4–6/month, DigitalOcean about
$24/month), a domain (about $10/year), and Caddy for HTTPS.

**Why deferred:** paid. **Option B (Hugging Face Spaces, CPU basic, free) is the only deployment target** for
this series, with an external free Postgres (Neon recommended; Supabase's free tier pauses inactive projects).

**Also deferred:** paid Hugging Face upgrades (persistent storage, better hardware, no-sleep), paid Neon /
Supabase / Sentry tiers.

**Known free-tier limits we accept:** the Space sleeps after inactivity (first visit wakes it, slowly), its disk
is reset on restart (hence the external Postgres), and it has a modest CPU.

**How to add it later:** follow `Docs/deployment.md` Option A. The container from plan 11 §8 runs unchanged.

---

## 4. Heavy local tooling on the developer's laptop

| Deferred item | Where it was | What we do instead now |
|---|---|---|
| **Docker Desktop** builds and `docker run` acceptance tests | plan 11 §14 | Hugging Face builds the image from the `Dockerfile` on its own servers. Test on the deployed Space. |
| **Local Postgres in Docker** | plan 08 §tests | SQLite locally. Postgres tests run in the CI `postgres` service container on GitHub (plan 11 §10). |
| **Playwright browser end-to-end tests** (downloads Chromium) | plan 10 | Streamlit `AppTest` page tests (already in CI) + a manual walkthrough on the deployed Space. Later: a CI-only Playwright job, never local. |
| **`ml` extra on the laptop** (sentence-transformers → PyTorch, UMAP, HDBSCAN, several GB) | 00 §1 install command | Locally install `.[app,dev,llm,fulltext,db]` only. `embedding.backend: auto` falls back to TF-IDF. Real MiniLM embeddings are built in GitHub Actions. |
| **Large pipeline runs on the laptop** (thousands of papers, embeddings, full-text backfill of ~10 hours "overnight") | plans 01 §8, 02 §2, 03, 04, 06 | GitHub Actions: the manual `dev-build` workflow (plan 01 §8.2), then the weekly refresh and daily continuation (plan 11 §9). Locally only the tiny `configs/domain_nlp_tiny.yaml` (≈ 20 papers). |
| **`pymupdf`** (better PDF headings; AGPL-3.0) | plan 02 §7 | `pypdf` only (BSD). Most papers come from arXiv HTML, so PDF is a rare fallback. |

**How to add them later:** each is an optional dev extra or one workflow job; nothing in the core design depends
on them. For `pymupdf`: re-add the `fulltext-agpl` extra, widen `fulltext.pdf_backend` to
`Literal["pypdf", "pymupdf"]`, add the `pymupdf` branch to `parse_pdf.pdf_to_text`, and accept the AGPL
obligations.

---

## 5. Bulk arXiv data (requester-pays S3)

**What:** arXiv's bulk source/PDF dumps on Amazon S3, which charge the **downloader** for transfer.

**Why deferred:** paid, and far beyond what the free pipeline needs.

**What we do instead now:** polite per-paper fetching (1 request / 3 s) inside GitHub Actions, capped by
`fulltext.max_papers_per_run` and resumed daily (plan 02, plan 11 §9.2).

---

## 6. Things to keep an eye on (still free today, but limits apply)

- **GitHub Actions minutes:** unlimited on **public** repositories; 2,000 minutes/month on a private repository
  with a free account. GitHub does not charge without a payment method; jobs just stop. If the repo is private
  and minutes run low, make it public (secrets stay hidden) or reduce `fulltext.max_papers_per_run` and the
  daily schedule.
- **Actions cache:** 10 GB per repository; entries unused for 7 days are evicted. Keep `data/fulltext` capped.
- **Free API limits change.** Re-check AI Studio's rate-limit page and the Groq/OpenRouter consoles regularly and
  update `llm.limits`.

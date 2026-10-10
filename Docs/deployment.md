# Deploying ScholarGrid for the public

A step-by-step guide to putting ScholarGrid on the internet so anyone can use it
at an address such as `https://papers.example.com`.

Commands marked **(your PC)** are typed in PowerShell in the project folder on
your Windows computer. Commands marked **(server)** are typed on the Linux server
after you connect to it with SSH.

---

## 0. How the deployed app works

```
 Your PC                               Server (Docker)                      Visitors
 ───────                               ───────────────                      ────────
 build the paper collection  ──copy──►  data/releases/<id>/  ◄── app reads ──  https://your-domain
 (pipeline, 1–2 hours)                  ScholarGrid container      │
                                        Caddy (HTTPS)              └── asks arXiv, OpenAlex and
                                                                       Semantic Scholar live
```

- The **heavy work** (collecting 50,000 papers, embedding them, clustering them)
runs **on your PC** and produces a *release*: one folder under `data/releases/`.
- The **server** only runs the app. It reads the release, and every search also
asks arXiv, OpenAlex and Semantic Scholar for more papers. Papers found that way
are saved under `data/live/` on the server, so the collection grows with use.
- To update the data later, you build a new release on your PC and copy it up.
The running app switches to it automatically.

---



## 1. Choose where to host it


|                             | **Option A: your own server (recommended)**                                              | **Option B: Hugging Face Spaces**                                             |
| --------------------------- | ---------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| Cost                        | about €4–6/month (Hetzner) or about $24/month (DigitalOcean) + a domain (about $10/year) | Free                                                                          |
| Address                     | your own domain, e.g. `papers.example.com`                                               | `https://<user>-<space>.hf.space`                                             |
| Always on                   | Yes                                                                                      | Sleeps after 48 hours without visitors; the next visitor waits about a minute |
| Papers found by live search | Kept permanently                                                                         | Lost whenever the Space restarts                                              |
| Updating data               | Copy a folder, no restart                                                                | Re-upload and rebuild (10–15 minutes)                                         |
| Difficulty                  | Medium (Linux commands, all given below)                                                 | Easy                                                                          |


**Streamlit Community Cloud is not suitable:** it has little memory for the
embedding model, and it deploys from GitHub, where the data folder is not stored.

Pick one. Section 3 is for Option A, section 4 is for Option B. Sections 2 and 5
apply to both.

---



## 2. Accounts and API keys to set up first


| What                     | Needed?                | Where                                                                                                                                                                                                | What you get                           |
| ------------------------ | ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------- |
| GitHub                   | Yes (you have it)      | [https://github.com](https://github.com)                                                                                                                                                             | The code is pulled from here           |
| OpenAlex contact email   | Strongly recommended   | No sign-up, see 2.1                                                                                                                                                                                  | Faster, more reliable OpenAlex answers |
| Semantic Scholar API key | Recommended            | [https://www.semanticscholar.org/product/api](https://www.semanticscholar.org/product/api)                                                                                                           | Semantic Scholar results in searches   |
| arXiv                    | Nothing to sign up for | [https://info.arxiv.org/help/api/tou.html](https://info.arxiv.org/help/api/tou.html)                                                                                                                 | Read the terms once                    |
| Server provider          | Option A               | [https://www.hetzner.com/cloud](https://www.hetzner.com/cloud) or [https://www.digitalocean.com](https://www.digitalocean.com)                                                                       | The machine that runs the app          |
| Domain name              | Option A               | [https://www.cloudflare.com/products/registrar](https://www.cloudflare.com/products/registrar), [https://porkbun.com](https://porkbun.com) or [https://www.namecheap.com](https://www.namecheap.com) | `papers.example.com`                   |
| Hugging Face             | Option B               | [https://huggingface.co/join](https://huggingface.co/join)                                                                                                                                           | Free hosting + an access token         |
| Sentry                   | Optional               | [https://sentry.io/signup](https://sentry.io/signup)                                                                                                                                                 | Emails when the app crashes            |
| UptimeRobot              | Optional               | [https://uptimerobot.com](https://uptimerobot.com)                                                                                                                                                   | Alerts when the site goes down         |




### 2.1 OpenAlex (no key)

OpenAlex needs no account. You only give it an email address it can contact if
something goes wrong; that puts the app in OpenAlex's faster "polite pool".
Use a real address you read. You will put it in the `OPENALEX_MAILTO` setting
later.

At the time of writing OpenAlex needs no API key. If
[https://docs.openalex.org](https://docs.openalex.org) says a key is required when you deploy, the source
adapter in `scholargrid/sources/openalex.py` needs a small change.

### 2.2 Semantic Scholar API key

Without a key, Semantic Scholar refuses most requests from a shared server,
and the app silently falls back to arXiv and OpenAlex. With a key it adds a
third source.

1. Open [https://www.semanticscholar.org/product/api](https://www.semanticscholar.org/product/api) and choose **Request an API key**.
2. Fill in the form: say it is a free, public research-discovery website, and
  give the expected traffic (for example "about 1 request per second at most").
3. Wait for the email. Approval can take a few days, so request it now; you can
  launch without it and add it later.
4. You will put the key in the `SEMANTIC_SCHOLAR_API_KEY` setting.



### 2.3 arXiv

No key. The app already follows arXiv's rule of at most one request every
3 seconds, and the page footer carries the required "Thank you to arXiv"
acknowledgement. Read the terms of use once:
[https://info.arxiv.org/help/api/tou.html](https://info.arxiv.org/help/api/tou.html)

### 2.4 Sentry (optional)

1. Sign up at [https://sentry.io](https://sentry.io), create a project, and choose platform **Python**.
2. Copy the **DSN** (looks like `https://abc123@o123.ingest.sentry.io/456`).
3. You will put it in the `SENTRY_DSN` setting. Crashes are then emailed to you.

---



## 3. Before deploying: prepare everything on your PC

Do these on your PC for both options.

### 3.1 Save the code to GitHub (your PC)

The server downloads the code from GitHub, so all your changes must be pushed.

```powershell
git status                       # review what changed
git add -A
git commit -m "Search upgrade and deployment setup"
git push origin main
```

`data/` and `.env` are in `.gitignore`, so the paper data and your secrets are
never uploaded to GitHub. That is intended.

### 3.2 Build the 50,000-paper collection (your PC, 1–2 hours)

The server serves whatever release you give it, so build the full collection
first.

```powershell
.venv\Scripts\Activate.ps1
$env:OPENALEX_MAILTO = "you@example.com"
python pipeline/run_pipeline.py --background
```

It runs in the background. Follow it on the app's **About** page (run
`streamlit run app/streamlit_app.py` and open [http://localhost:8501](http://localhost:8501)), or with:

```powershell
Get-Content data\grow_status.json
Get-Content data\grow.log -Tail 20 -Wait     # Ctrl+C to stop watching
```

When `grow_status.json` says `"state": "done"`, check the result:

```powershell
Get-Content data\releases\CURRENT                          # the new release id, e.g. 2026.10.11
python -c "import json; m=json.load(open('data/releases/' + open('data/releases/CURRENT').read().strip() + '/meta.json')); print(m['counts']['papers'], 'papers;', m['embedding_model'])"
```

You should see tens of thousands of papers and the model
`BAAI/bge-small-en-v1.5@latest`. Also open
`data\releases\<id>\validation_report.md` and skim it.

If the run fails (for example the network dropped), run the same command again.
Finished months are cached, so it continues where it stopped.

### 3.3 Write down your release id

```powershell
Get-Content data\releases\CURRENT
```

This guide calls it `<id>` (for example `2026.10.11`).

### 3.4 Pack the release into one file (your PC)

```powershell
tar -czf scholargrid-release.tgz -C data releases/<id> releases/CURRENT
```

Replace `<id>` with your id. This creates `scholargrid-release.tgz` (a few hundred
megabytes) in the project folder.

### 3.5 Optional: test the production container on your PC

This checks the exact image the server will run. It needs Docker Desktop
([https://www.docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop)).

```powershell
docker build -t scholargrid .
docker run --rm -p 8501:8501 -v "${PWD}\data:/app/data" -e OPENALEX_MAILTO=you@example.com scholargrid
```

Open [http://localhost:8501](http://localhost:8501), wait for the loading screen (about 20 seconds), and
search "technical debt". Press Ctrl+C in PowerShell to stop it.

---



## 4A. Option A: your own server



### A1. Create an SSH key (your PC, once)

An SSH key lets you log in to the server without a password.

```powershell
ssh-keygen -t ed25519 -C "you@example.com"     # press Enter three times
Get-Content $HOME\.ssh\id_ed25519.pub           # copy this whole line
```



### A2. Rent the server

Using Hetzner ([https://console.hetzner.cloud](https://console.hetzner.cloud)), DigitalOcean works the same way:

1. Create an account and a project.
2. **Add Server**:
  - **Location:** the one closest to your users.
  - **Image:** Ubuntu 24.04.
  - **Type:** at least **2 vCPU, 4 GB RAM, 40 GB disk** (Hetzner "CX22" or
  larger; on DigitalOcean the 4 GB droplet).
  - **SSH key:** paste the line you copied in A1.
  - **Name:** `scholargrid`.
3. Create it and write down its **IPv4 address**, for example `203.0.113.10`.

4 GB of memory is the minimum: the app loads PyTorch, the embedding model and
the paper collection.

### A3. Buy a domain and point it at the server

1. Buy a domain at a registrar (Cloudflare, Porkbun, Namecheap…).
2. In the registrar's **DNS** settings add a record:
  - **Type:** `A`
  - **Name:** `papers` (gives `papers.example.com`), or `@` for the bare domain
  - **Value:** your server IP from A2
  - **TTL:** automatic
  - On Cloudflare: set the proxy to **DNS only** (grey cloud) for now, so the
  HTTPS certificate in step A9 can be issued.
3. Check it after a few minutes (your PC):

```powershell
Resolve-DnsName papers.example.com
```

It should show your server IP. It can take up to an hour.

### A4. Log in and secure the server (server)

```powershell
ssh root@203.0.113.10          # (your PC) use your IP; answer "yes" the first time
```

Now you are on the server:

```bash
apt update && apt upgrade -y
apt install -y ufw unattended-upgrades
ufw allow OpenSSH
ufw allow 80
ufw allow 443
ufw --force enable
dpkg-reconfigure -plow unattended-upgrades     # choose "Yes": automatic security updates
```

Port 8501 (the app itself) stays closed to the internet. Visitors reach it only
through HTTPS on port 443.

### A5. Install Docker (server)

```bash
curl -fsSL https://get.docker.com | sh
docker --version
```



### A6. Download the code (server)

```bash
mkdir -p /opt/scholargrid && cd /opt/scholargrid
git clone https://github.com/A-Piyas-04/PaperPorteHobe.git code
```

If the GitHub repository is private, create a token at GitHub → Settings →
Developer settings → Personal access tokens (read access to the repository), and
use it as the password when `git clone` asks.

### A7. Upload the paper data (your PC, then server)

From your PC, in the project folder:

```powershell
scp scholargrid-release.tgz root@203.0.113.10:/opt/scholargrid/
```

Then on the server:

```bash
cd /opt/scholargrid
mkdir -p data
tar -xzf scholargrid-release.tgz -C data
chown -R 1000:1000 data          # the app runs as user 1000 and saves papers here
ls data/releases                 # shows your <id> folder and CURRENT
```



### A8. Add your secrets and start the app (server)

Create the settings file:

```bash
nano /opt/scholargrid/.env
```

Paste this, filling in your values (leave a line empty if you don't have that
key yet):

```
OPENALEX_MAILTO=you@example.com
SEMANTIC_SCHOLAR_API_KEY=
SENTRY_DSN=
```

Save with Ctrl+O and Enter, exit with Ctrl+X. Then lock the file and build and
start the app:

```bash
chmod 600 /opt/scholargrid/.env
cd /opt/scholargrid/code
docker build -t scholargrid .          # 5–15 minutes the first time
docker run -d --name scholargrid --restart unless-stopped \
  --env-file /opt/scholargrid/.env \
  -v /opt/scholargrid/data:/app/data \
  -p 127.0.0.1:8501:8501 \
  scholargrid
```

`--restart unless-stopped` restarts the app after crashes and server reboots.

Check that it is running:

```bash
docker ps                                       # STATUS should become "healthy" after about a minute
docker logs --tail 50 scholargrid
curl -s http://127.0.0.1:8501/_stcore/health    # prints "ok"
```

**If your release was built before the search upgrade** (its `meta.json` says
`all-MiniLM-L6-v2`), build the image with that model instead:
`docker build --build-arg EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2 -t scholargrid .`

### A9. Turn on HTTPS with Caddy (server)

Caddy is a web server that gets and renews free HTTPS certificates
automatically.

```bash
apt install -y debian-keyring debian-archive-keyring apt-transport-https curl gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | tee /etc/apt/sources.list.d/caddy-stable.list
apt update && apt install -y caddy
```

Replace the Caddy configuration:

```bash
nano /etc/caddy/Caddyfile
```

Delete everything in it and paste (with your domain):

```
papers.example.com {
    encode gzip
    reverse_proxy 127.0.0.1:8501
}
```

Save, exit, and apply:

```bash
systemctl reload caddy
journalctl -u caddy --since "5 minutes ago" | tail -20     # look for "certificate obtained successfully"
```



### A10. Open your site

Go to `https://papers.example.com`. Then run through the checklist in section 6.

If you use Cloudflare, you can now switch the DNS record back to **Proxied**
(orange cloud) and set SSL/TLS mode to **Full (strict)**.

---



## 4B. Option B: Hugging Face Spaces (free)



### B1. Create the Space

1. Sign up at [https://huggingface.co/join](https://huggingface.co/join).
2. Go to [https://huggingface.co/new-space](https://huggingface.co/new-space) and choose:
  - **Name:** `scholargrid`
  - **SDK:** Docker → **Blank**
  - **Hardware:** CPU basic (free)
  - **Visibility:** Public
3. Create an access token: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) → **Create new
  token** → type **Write** → copy it.



### B2. Add your secrets

In the Space: **Settings → Variables and secrets → New secret**. Add each one
separately:

- `OPENALEX_MAILTO` = your email
- `SEMANTIC_SCHOLAR_API_KEY` = your key (if you have one)
- `SENTRY_DSN` = your DSN (optional)



### B3. Prepare the upload folder (your PC)

A Space needs the data inside the image, so you upload a copy of the project
that includes the release.

```powershell
pip install -U huggingface_hub
hf auth login                    # paste the Write token (older versions: huggingface-cli login)

$dst = "$HOME\scholargrid-space"
New-Item -ItemType Directory -Force $dst | Out-Null
Copy-Item -Recurse -Force scholargrid, app, configs, .streamlit, app.py, requirements.txt, Dockerfile $dst
New-Item -ItemType Directory -Force "$dst\data\releases" | Out-Null
Copy-Item -Recurse -Force data\releases\<id> "$dst\data\releases\"
Copy-Item -Force data\releases\CURRENT "$dst\data\releases\"
Get-ChildItem -Recurse -Directory -Filter __pycache__ $dst | Remove-Item -Recurse -Force
```

Do **not** copy `.dockerignore`: it would exclude the data folder from the image.

### B4. Make two edits in the copy

1. Create `$HOME\scholargrid-space\README.md` with exactly this content (Hugging
  Face reads the settings from it):
2. In `$HOME\scholargrid-space\Dockerfile`, find the line
  ```
   RUN mkdir -p data && chown scholargrid:scholargrid data
  ```
   and add this line directly below it:



### B5. Upload (your PC)

```powershell
hf upload <your-hf-username>/scholargrid $HOME\scholargrid-space . --repo-type=space
```

Large files are handled automatically. Open the Space page and click **Logs**: the
first build takes 10–15 minutes, then the app starts. Share the Space page
(`https://huggingface.co/spaces/<user>/scholargrid`) or the direct app address
`https://<user>-scholargrid.hf.space`.

### B6. Limits of the free Space

- It sleeps after 48 hours without visitors; the next visitor waits about a minute.
- Its disk resets on every restart, so papers found by live search are forgotten
(search still works; they are just fetched again).
- To update data or code, repeat B3–B5 with the new release.

---



## 5. After launch: running it



### 5.1 Watch it

- **Uptime:** in UptimeRobot add an **HTTP(s)** monitor for
`https://papers.example.com/_stcore/health` every 5 minutes. You get an email
when the site goes down.
- **Errors:** with `SENTRY_DSN` set, crashes appear in Sentry and are emailed to you.
- **Logs (server):** `docker logs --tail 100 -f scholargrid` (Ctrl+C to stop).
- **Disk (server):** `df -h /` once a month.



### 5.2 Update the code (Option A)

On your PC, commit and push. Then on the server:

```bash
cd /opt/scholargrid/code
git pull
docker build -t scholargrid .
docker rm -f scholargrid
docker run -d --name scholargrid --restart unless-stopped \
  --env-file /opt/scholargrid/.env \
  -v /opt/scholargrid/data:/app/data \
  -p 127.0.0.1:8501:8501 \
  scholargrid
```

The site is down for about a minute while the new container starts.

### 5.3 Update the paper data (Option A)

1. On your PC, build a new release (section 3.2) and pack it (section 3.4) with
  the **new** id.
2. Upload and unpack it on the server:
  ```powershell
   scp scholargrid-release.tgz root@203.0.113.10:/opt/scholargrid/     # (your PC)
  ```

Unpacking updates `CURRENT`, and the app switches to the new release within
about 20 seconds; the first visitor after that waits briefly while it loads. No
restart is needed.

Rebuild about once a month so new papers appear in Areas and Trends. Search
always includes the newest papers through the live sources anyway.

### 5.4 Roll back to an earlier release (server)

```bash
ls /opt/scholargrid/data/releases
echo 2026.10.11 > /opt/scholargrid/data/releases/CURRENT    # an id from the list
```

The app switches within about 20 seconds.

### 5.5 Housekeeping (server)

Saved search answers older than a day are no longer used. Delete them nightly:

```bash
crontab -e      # choose nano if asked, then add this line at the bottom:
0 3 * * * find /opt/scholargrid/data/live/query_cache -type f -mtime +2 -delete
```

Back up the papers found by live search occasionally (your PC):

```powershell
scp -r root@203.0.113.10:/opt/scholargrid/data/live .\backup-live
```

Old releases you no longer need can be deleted from `data/releases/` (never the
one named in `CURRENT`).

---



## 6. Launch checklist

- [ ] The site opens over `https://` with no browser warning.
- [ ] The loading screen appears, then the home page (about 20 seconds on a fresh start).
- [ ] Searching `technical debt` shows **Exact matches** and a line like
  ```
  "Showing 40 of about 13,000 papers · arXiv … · OpenAlex …".
  ```
- [ ] Searching `SATD` says "Also searched: self admitted technical debt".
- [ ] No grey note says a source "could not be reached". "Semantic Scholar is
  ```
  limiting requests" is expected until you have its API key.
  ```
- [ ] Save to reading list → Reading list page → Download BibTeX works.
- [ ] More like this, Explore results, Areas, Trends, Leads and About all open.
- [ ] The About page shows the dataset with your expected paper count, and **no**
  ```
  "Grow collection" button (it is hidden on public sites).
  ```
- [ ] No "Demo data" or "Semantic search is temporarily unavailable" banner.
- [ ] The site works on a phone.
- [ ] UptimeRobot (and Sentry, if used) are set up.

---



## 7. Troubleshooting


| Problem                                                                            | Cause and fix                                                                                                                                                                             |
| ---------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Browser shows "502 Bad Gateway"                                                    | The app is still starting (wait a minute) or crashed: `docker logs --tail 100 scholargrid`                                                                                                |
| Container keeps restarting, log says `ConfigError ... synthetic`                   | The image was built from old code. `git pull` and rebuild (the image must use `configs/deploy.yaml`)                                                                                      |
| "Semantic search is temporarily unavailable" banner                                | The image has a different embedding model than the release. Check `embedding_model` in the release's `meta.json` and rebuild with the matching `--build-arg EMBED_MODEL=...` (section A8) |
| "No published release" / setup page appears                                        | The data folder is missing or not mounted: check `ls /opt/scholargrid/data/releases` and the `-v` part of `docker run`                                                                    |
| Searches never show "found online"; log says `Permission denied` under `data/live` | Run `chown -R 1000:1000 /opt/scholargrid/data`                                                                                                                                            |
| "Semantic Scholar is limiting requests" on every search                            | No API key, or a wrong one: check `SEMANTIC_SCHOLAR_API_KEY` in `.env`, then `docker rm -f scholargrid` and run it again (section 5.2)                                                    |
| OpenAlex often "took too long"                                                     | Set `OPENALEX_MAILTO`; check [https://status.openalex.org](https://status.openalex.org)                                                                                                   |
| Caddy cannot get a certificate                                                     | DNS does not point to the server yet, ports 80/443 are closed (`ufw status`), or Cloudflare proxy is on (switch to DNS only)                                                              |
| The server runs out of memory (`docker logs` shows "Killed")                       | Use a server with at least 4 GB RAM                                                                                                                                                       |
| `docker build` fails with "no space left on device"                                | Free space: `docker system prune -af` (removes unused images), or use a bigger disk                                                                                                       |


---



## 8. Security notes

- Keep secrets only in `/opt/scholargrid/.env` (Option A) or Space secrets
(Option B). Never put them in YAML files or Git.
- Only serve releases you built yourself: release folders contain files that can
run code when loaded.
- The public site uses `configs/deploy.yaml`: production checks are on and the
"Grow collection" button is hidden, so visitors cannot start background jobs.
- The app runs as a normal (non-root) user inside the container, and only Caddy
is reachable from the internet.
- Keep the server updated (unattended upgrades from A4) and rebuild the image
every month or two to pick up library security fixes.


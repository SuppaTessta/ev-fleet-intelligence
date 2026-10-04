# Setup

Two ways to run this. **Docker is the easy one** — three commands, no Python needed.

---

## Option A — Docker (recommended)

Install [Docker Desktop](https://www.docker.com/products/docker-desktop/), then:

```bash
git clone https://github.com/SuppaTessta/ev-fleet-intelligence.git
cd ev-fleet-intelligence
docker compose up -d
```

The first run takes 10–15 minutes, almost all of it downloading TensorFlow. After that it starts in
seconds.

Open **http://localhost:8501** for the dashboard, or **http://localhost:8000/docs** to try the API directly.

Stop it with `docker compose down`.

### What you'll see

Six of the seven agents work immediately. **Manufacturing Quality will show as degraded** — it needs two 98 MB image classifiers that are too large for GitHub, so you train them yourself if you want them ([below](#optional-the-seventh-agent)).

This is deliberate, not a broken install: `http://localhost:8000/ready` lists exactly which agents can answer and which cannot, and the dashboard says so on the page instead of showing empty charts.

---

## Option B — Python, no Docker

**Use Python 3.12.** TensorFlow has no build for 3.13 or newer, and `pip` on those versions tries to compile numpy from source and fails without a C compiler. Installing 3.12 does not remove any Python you already have.

Get it from [python.org/downloads](https://www.python.org/downloads/windows/) — scroll down to 3.12.x, not the big button at the top. Tick **"Add python.exe to PATH"** and **"py launcher"** during install.

```bash
git clone https://github.com/SuppaTessta/ev-fleet-intelligence.git
cd ev-fleet-intelligence
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r backend/requirements.txt
```

On macOS or Linux, use `python3.12 -m venv .venv` and `source .venv/bin/activate`.

Then start the two services in **two separate terminals**, activating the venv in both:

```bash
python -m uvicorn app.main:app --app-dir backend --port 8000
```

```bash
python -m streamlit run dashboard/main.py
```

Same URLs as above. On Windows you can use `.\make.ps1 run-api` and `.\make.ps1 run-dashboard` instead; on macOS/Linux, `make run-api` and `make run-dashboard`.

---

## Optional: the seventh agent

The two quality classifiers are the only things not included, because they are 98 MB each. Training both takes roughly 30–60 minutes on a CPU. You need Option B set up first.

**Cast components** — download the "Real-life Industrial Dataset of Casting Product" and put the **`casting_512x512` originals** (781 defective, 519 OK) where the script expects them:

```bash
python data/prepare_casting.py
python train/train_quality_model.py
```

Use the originals, not the dataset's own `casting_data` split — that split was made after augmentation, so 97.5% of its test parts also appear in training. A model trained on it reports 99.44% and scores 88.08% on genuinely unseen castings ([ADR-0003](docs/adr/0003-casting-split-leakage.md)).

**Steel surfaces** — download the NEU Surface Defect Database (NEU-DET) to `data/raw/neu_det_source/`:

```bash
python data/prepare_neu_det.py
python train/train_quality_model_neu_det.py
```

Both write into `backend/models/`, which Docker mounts, so `docker compose restart api` picks them up and `/ready` goes green.

### Rebuilding everything else

Nothing else needs this — the small artifacts ship with the repo. But every published number is reproducible:

```bash
python train/train_risk_model.py         # seconds
python train/train_fleet_readiness.py    # seconds
python train/train_battery_model.py      # needs the NASA data, see below
```

The battery model ships trained. To rebuild it, `python data/download_battery_data.py` fetches 4 of the 19 cells automatically; the other 15 come from the "Li-ion Battery Dataset from NASA PCoE" on Kaggle, unzipped into `data/raw/nasa_battery/`, then `python data/parse_new_batteries.py`.

---

## If something goes wrong

| Problem | Fix |
|---|---|
| `docker: command not found` | Docker Desktop isn't installed or isn't running. |
| Port 8000 or 8501 already in use | Something else is using it — stop that, or change the port in `docker-compose.yml`. |
| `ModuleNotFoundError` | The venv isn't active. Run the activate command again — your prompt should start with `(.venv)`. |
| `running scripts is disabled` (Windows) | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then retry. |
| `py -3.12` not recognised | The py launcher wasn't installed. Reinstall Python 3.12 with that box ticked. |
| A dashboard page says "Run `python train/...` first" | That agent's artifact is missing. See the section above. |

Still stuck? Open an issue with the error text and the output of `python -V`.

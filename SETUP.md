# Local Setup Guide (VS Code, from scratch)

Free end to end — just Python, Git, and VS Code, all free tools. Covers
all 7 agents currently on the platform.

## 0. Why Python 3.14 failed, and what we're using instead

Your error was `pip` trying to **compile numpy from source** because no
prebuilt wheel exists for Python 3.14 at the pinned versions — and there's
no C compiler on your machine to do that compile (that's what the
`Unknown compiler(s): [cl, gcc, clang...]` error meant).

It's not just a version-pin problem, either: **TensorFlow — which the
Manufacturing Quality agent needs — has no build at all for Python 3.14.**
This is normal, expected friction with a Python version this new — most ML
libraries take 6-12+ months to catch up.

**Fix: install Python 3.12 side-by-side with your existing 3.14.** This
doesn't touch or uninstall 3.14 — Windows happily runs both, and we'll
point this project specifically at 3.12.

## 1. Install Python 3.12

Download from **python.org/downloads/windows** — scroll to "Python 3.12.x"
(not the big button at the top, which is the newest version). Get the
"Windows installer (64-bit)".

During install:
- Tick **"Add python.exe to PATH"**
- Tick **"py launcher"** (usually on by default) — lets us select 3.12
  specifically even with 3.14 also installed

Verify: open a new terminal and run `py -3.12 --version` → should print
`Python 3.12.x`.

## 2. Install Git (if you don't have it)

git-scm.com/downloads → default options are fine.

## 3. Open the project in VS Code

1. Unzip the project somewhere, e.g. `Documents\ev-fleet-intelligence`
2. Open VS Code → File → Open Folder → select that folder
3. If prompted, install the **Python extension** (Extensions icon →
   search "Python" → Install — Microsoft's official one)
4. Open a terminal inside VS Code: **Terminal → New Terminal**

## 4. Create the virtual environment — pinned to 3.12

```
py -3.12 -m venv venv
```
VS Code will likely pop up "Select Python Interpreter" — choose
`.\venv\Scripts\python.exe`. If it doesn't ask: `Ctrl+Shift+P` → "Python:
Select Interpreter" → pick it manually.

Activate it manually this one time:
```
venv\Scripts\Activate.ps1
```
You should see `(venv)` at the start of your terminal prompt.

> **"Running scripts is disabled" error?** Run once:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then retry.

## 5. Install dependencies

```
pip install -r backend/requirements.txt
```
Installs fastapi, lightgbm, scikit-learn, pandas, tensorflow-cpu, pulp,
and the rest. Takes a few minutes (TensorFlow is the big one, ~200MB).

## 6. Battery agent — 19 real NASA batteries

```
python data/download_battery_data.py
```
*Expected:* fetches B0005/6/7/18 automatically (~5 seconds).

For the other 15 batteries (G3, G4, G7, G9 condition groups), download
**"Li-ion Battery Dataset from NASA PCoE"** from Kaggle, unzip it, then
copy these 15 files from `Battery_DataSet\Battery_DataSet\` into
`data/raw/nasa_battery\`:
```
B0029, B0030, B0031, B0032, B0033, B0034, B0036,
B0045, B0046, B0047, B0048, B0053, B0054, B0055, B0056  (.mat files)
```
Then parse them all in one command:
```
python data/parse_new_batteries.py
```
*Expected:* 15 lines like "B0029: parsed 40 discharge cycles", ending
"Done: 15 parsed, 0 missing." If any show as missing, the copy step above
didn't land in the right folder.

Train:
```
python train/train_battery_model.py
```
*Expected:* "Loading 19 batteries across 5 condition groups", 5 per-battery
MAE/RMSE/R2 lines (2 will say "R2=undefined" — that's correct, not an
error, see the printed explanation), then "Pooled R2 (...): 0.740",
ending "Saved model to .../battery_rul_model.pkl".

> Skipping the 15-battery download? The battery agent still works with
> just B0005/6/7/18 — you'll just be back to the smaller, earlier result
> set rather than the full 19-battery one.

## 7. Quality agent — casting defect images

You already have this dataset. Place it so the folder structure looks
like:
```
data/raw/casting_defect/train/def_front/   ← defective casting images
data/raw/casting_defect/train/ok_front/    ← OK casting images
data/raw/casting_defect/test/def_front/
data/raw/casting_defect/test/ok_front/
```
Train:
```
python train/train_quality_model.py
```
*Expected:* downloads real ImageNet-pretrained ResNet50 weights (~90MB,
one time), then **20-40 minutes on CPU** for 6 epochs over ~6,600 images
— that's normal, let it run. Ends with a classification report and
"Saved model to .../quality_model.keras".

*Faster first check:* temporarily change `EPOCHS = 6` to `EPOCHS = 2` near
the top of the script — confirms everything works in a few minutes before
committing to the full run.

## 8. Supply Chain Risk agent — synthetic data, no download needed

```
python train/train_risk_model.py
```
*Expected:* "1000 shipments... 81 true injected anomalies across 5
archetypes", Precision/Recall/F1 (~0.6-0.7 range), ending "Saved model to
.../risk_model.pkl".

Optional but worth running once — the lead-time analysis (generates a
plot for your submission doc):
```
python train/analyze_risk_lead_time.py
```
*Expected:* ends with "WATCH-level lead time: 4 shipments (~28 days)...",
saves `docs/risk_lead_time_analysis.png`.

## 9. Fleet Electrification Readiness agent — no download needed

```
python train/train_fleet_readiness.py
```
*Expected:* "Scoring against 3 real EV models...", "60 vehicles scored,
~77% ready...", ends "Saved to .../fleet_readiness.csv".

## 10. Net Zero Carbon Tracker — no training step

Nothing to run — the real CEA grid emission data and vehicle comparison
figures are built directly into `backend/app/agents/carbon_agent.py`, so
this agent works as soon as the backend starts. Optionally place your own
copy of the CEA source file at `data/raw/cea/CEA_CO2_Baseline_Database.xlsx`
for your own records — the running app doesn't read it live.

## 11. Maintenance Optimiser — no training step

Nothing to run — this is a live scheduling optimizer (PuLP), not a
trained model. Works as soon as the backend starts and pulls current
priorities from the Battery + Compound Risk agents live.

## 12. Run the API

```
cd backend
uvicorn app.main:app --reload --port 8000
```
Browser → `http://127.0.0.1:8000/docs` → all 7 agents listed, each
individually testable.

## 12b. Run the dashboard (separate terminal, keep the API running)

Open a **second** terminal (VS Code: click `+` in the terminal panel),
activate the venv again (`venv\Scripts\Activate.ps1`), then:
```
streamlit run dashboard/app.py
```
*Expected:* opens your browser to `http://localhost:8501`, landing on
**Fleet Command Center**. Sidebar has 8 sections. Both terminals need to
stay running.

## 13. Push to GitHub

```
git init
git add .
git commit -m "EV Fleet Intelligence: all 7 agents"
```
Create an empty repo on github.com (green "New" button, don't add a
README), then run the two commands it shows you:
```
git remote add origin https://github.com/SuppaTessta/ev-fleet-intelligence.git
git branch -M main
git push -u origin main
```

## Troubleshooting

- **`ModuleNotFoundError`** → venv isn't activated, or step 5 didn't finish. Re-run both.
- **VS Code terminal doesn't show `(venv)`** → `Ctrl+Shift+P` → "Python: Select Interpreter" → pick `.\venv\Scripts\python.exe`, open a fresh terminal.
- **`py -3.12` not recognized** → py launcher wasn't installed; reinstall Python 3.12 with that box ticked.
- **A dashboard section shows "Run `python train/...` first"** → that agent's training step (6-9 above) hasn't been run yet in this environment.
- **Anything else** → paste the exact error back in chat.

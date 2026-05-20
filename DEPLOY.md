# Deployment Guide

## Architecture

```
GitHub Pages (React)  ──→  Render (Flask API)
web/build/                  api/
```

---

## 1. Train and save the model locally

```bash
cd api
python3 train.py        # produces api/saved/model.pt and api/saved/vocab.json
```

The saved files are committed to the repo so Render can load them without retraining.

---

## 2. Deploy the Flask API to Render

1. Push the repo to GitHub (the `api/saved/` directory **must** be committed).
2. On [render.com](https://render.com), create a **New Web Service**.
3. Connect your GitHub repo.
4. Set:
   - **Root directory**: `api`
   - **Build command**: `pip install -r requirements.txt`
   - **Start command**: `gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 120 app:app`
   - **Environment**: Python 3.11
5. Deploy. Copy the URL (e.g. `https://lstm-api-xxxx.onrender.com`).

> **Note**: Render free-tier containers spin down after inactivity. The first request after sleep takes ~30s. Upgrade to a paid instance to avoid cold starts.

---

## 3. Deploy the React front-end to GitHub Pages

```bash
cd web

# Set your Render API URL
echo "REACT_APP_API_URL=https://your-api.onrender.com" > .env.local

# Add homepage to package.json (replace with your GitHub username/repo)
# "homepage": "https://sassom2112.github.io/lstm-text-prediction"

npm run build
npm install -g gh-pages   # if not already installed
npx gh-pages -d build
```

Your app will be live at `https://sassom2112.github.io/lstm-text-prediction`.

---

## 4. Local development

```bash
# Terminal 1 — API
cd api
python3 app.py            # runs on http://localhost:5000

# Terminal 2 — React
cd web
echo "REACT_APP_API_URL=http://localhost:5000" > .env.local
npm start                 # runs on http://localhost:3000
```

# 🚀 Deployment Guide - Flower Classification App

## Local Development Setup

1. **Clone/navigate to project**:
   ```bash
   cd flower_app
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # macOS/Linux
   pip install -r requirements.txt
   ```

2. **Run tests**:
   ```bash
   python test_app.py
   ```

3. **Run locally**:
   ```bash
   streamlit run app.py
   ```

---

## GitHub Setup

### Step 1: Initialize Git

```bash
git init
git add .
git commit -m "Initial commit: Flower Classification App"
```

### Step 2: Create GitHub Repository

1. Go to [GitHub.com](https://github.com/new)
2. Create repository: `flower-app`
3. Choose Public repository
4. Do NOT initialize with README

### Step 3: Push to GitHub

```bash
git remote add origin https://github.com/YOUR-USERNAME/flower-app.git
git branch -M main
git push -u origin main
```

### Step 4: Handle Large Model Files

If model files are too large (>100MB):

```bash
# Install Git LFS
git lfs install

# Track model files
git lfs track "*.keras" "*.pkl"
git add .gitattributes
git commit -m "Configure Git LFS for model files"
git push origin main
```

---

## Streamlit Cloud Deployment

### Step 1: Connect to Streamlit Cloud

1. Visit [Streamlit Cloud](https://share.streamlit.io)
2. Sign up with GitHub (if needed)
3. Click "New app"

### Step 2: Configure App

- **Repository**: YOUR-USERNAME/flower-app
- **Branch**: main
- **Main file path**: app.py

### Step 3: Deploy

1. Click "Deploy"
2. Wait 2-5 minutes for deployment
3. Access app at: `flower-app-username.streamlit.app`

---

## Updating the App

1. Make changes locally
2. Test: `streamlit run app.py`
3. Commit: `git add . && git commit -m "message"`
4. Push: `git push origin main`
5. Streamlit Cloud auto-redeploys

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Models not found | Verify model files in repo, check config.py paths |
| Installation fails | `pip install --upgrade pip`, `pip cache purge` |
| App crashes | Check requirements.txt, test locally first |
| Slow inference | Enable model caching, optimize image size |
| Memory errors | Reduce batch size, compress images |
| GitHub auth fails | Set up SSH keys or personal access token |

---

## Requirements Met

✅ Local development ready
✅ GitHub repository configured
✅ Streamlit Cloud deployment instructions
✅ CI/CD workflow in `.github/workflows/deploy.yml`
✅ Model files included
✅ All dependencies in requirements.txt
✅ Complete README with setup guide

---

**Deployment Ready! 🚀**

Visit your Streamlit Cloud dashboard to monitor app performance and logs.
# 🚀 DocVision AI - Deployment & Infrastructure Guide

This guide provides step-by-step instructions for deploying **DocVision AI** across local containers and major cloud hosting platforms (**Docker**, **Render**, **HuggingFace Spaces**, **Streamlit Cloud**, and **GitHub Actions**).

---

## 🐳 1. Docker & Docker Compose Deployment

### Local Docker Build & Run
```bash
# 1. Build production Docker image
docker build -t docvision-ai:latest .

# 2. Run Streamlit Studio (Port 8501)
docker run -d -p 8501:8501 --name docvision_studio docvision-ai:latest

# 3. Access in browser: http://localhost:8501
```

### Docker Compose Multi-Container Setup
Deploy both the **FastAPI Microservice (Port 8000)** and **Streamlit Studio (Port 8501)** simultaneously:
```bash
# Launch containers in detached mode
docker-compose up -d --build

# View container logs
docker-compose logs -f

# Stop containers
docker-compose down
```

---

## 🌐 2. Render Cloud Deployment

Render supports zero-downtime deployment using the included [`render.yaml`](render.yaml) configuration file.

### Step-by-Step Instructions:
1. Push your repository to **GitHub / GitLab**.
2. Log into [Render Dashboard](https://dashboard.render.com/).
3. Click **New +** -> **Blueprint**.
4. Connect your `DocVision-AI` repository. Render automatically reads `render.yaml` and provisions:
   - `docvision-studio` (Streamlit Studio on port 8501)
   - `docvision-api` (FastAPI REST service on port 8000)
5. Click **Apply**. Your app will build and deploy automatically!

---

## 🤗 3. HuggingFace Spaces Deployment

HuggingFace Spaces supports custom Docker SDK spaces.

### Step-by-Step Instructions:
1. Create a new Space on [HuggingFace Spaces](https://huggingface.co/new-space).
2. Select **Space SDK**: `Docker` -> **Blank Docker**.
3. Clone your HuggingFace Space repository locally:
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/docvision-ai
   cd docvision-ai
   ```
4. Add the following YAML frontmatter at the very top of `README.md`:
   ```yaml
   ---
   title: DocVision AI - Document Intelligence & Fraud Detection
   emoji: 🛡️
   colorFrom: indigo
   colorTo: purple
   sdk: docker
   app_port: 8501
   pinned: false
   ---
   ```
5. Copy your project files (`Dockerfile`, `requirements.txt`, `app.py`, `src/`) into the HuggingFace repo folder and push:
   ```bash
   git add .
   git commit -m "Deploy DocVision AI Docker space"
   git push
   ```
6. HuggingFace will build the Docker container and host your app live!

---

## ☁️ 4. Streamlit Community Cloud Deployment

Deploy directly from GitHub with zero server maintenance:

### Step-by-Step Instructions:
1. Push your code to a public/private GitHub repository.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **New app**.
4. Select your repository, branch (`main`), and set **Main file path** to `app.py`.
5. Click **Deploy!**

*(Theme configuration is automatically configured via [`.streamlit/config.toml`](.streamlit/config.toml)).*

---

## ⚙️ 5. Automated GitHub Actions CI/CD Pipeline

The included GitHub Actions deployment workflow ([`.github/workflows/deploy.yml`](.github/workflows/deploy.yml)) automatically builds and publishes production Docker images to **GitHub Container Registry (`ghcr.io`)** whenever a version tag (e.g. `v1.0.0`) is pushed:

```bash
# Push release tag to trigger automated Docker build & publish
git tag v1.0.0
git push origin v1.0.0
```

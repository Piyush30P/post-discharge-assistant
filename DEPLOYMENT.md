# 🚀 Deployment Guide

## 📦 GitHub Repository Setup

### Prerequisites

- GitHub account
- Git installed locally
- Docker Desktop (for containerized deployment)

### 1. Create GitHub Repository

1. Go to [GitHub](https://github.com) and create a new repository
2. Name it: `post-discharge-assistant`
3. Make it public (or private if you prefer)
4. Don't initialize with README (we already have one)

### 2. Push to GitHub

```bash
# Initialize git repository (if not already done)
git init

# Add all files
git add .

# Commit files
git commit -m "Initial commit: Post-Discharge Medical AI Assistant"

# Add remote origin (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/post-discharge-assistant.git

# Push to GitHub
git push -u origin main
```

### 3. Set Up Repository Secrets (for API Keys)

If you plan to use GitHub Actions for deployment, add these secrets:

1. Go to your repository on GitHub
2. Click `Settings` → `Secrets and variables` → `Actions`
3. Add these repository secrets:
   - `GOOGLE_API_KEY`: Your Google Gemini API key
   - `PINECONE_API_KEY`: Your Pinecone API key
   - `TAVILY_API_KEY`: Your Tavily API key

## 🐳 Docker Deployment Options

### Option 1: Local Docker

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/post-discharge-assistant.git
cd post-discharge-assistant

# Set up environment
cp .env.docker .env
# Edit .env with your API keys

# Run with Docker Compose
docker-compose up --build
```

### Option 2: Docker Hub

```bash
# Build and tag image
docker build -t your-username/post-discharge-assistant .

# Push to Docker Hub
docker login
docker push your-username/post-discharge-assistant

# Others can pull and run
docker pull your-username/post-discharge-assistant
docker run -p 8501:8501 --env-file .env your-username/post-discharge-assistant
```

### Option 3: GitHub Container Registry

```bash
# Login to GitHub Container Registry
echo $GITHUB_TOKEN | docker login ghcr.io -u YOUR_USERNAME --password-stdin

# Build and tag for GitHub Container Registry
docker build -t ghcr.io/your-username/post-discharge-assistant .

# Push to GitHub Container Registry
docker push ghcr.io/your-username/post-discharge-assistant
```

## ☁️ Cloud Deployment Options

### 1. Heroku

1. Install Heroku CLI
2. Create `heroku.yml`:

```yaml
build:
  docker:
    web: Dockerfile
```

3. Deploy:

```bash
heroku create your-app-name
heroku stack:set container
heroku config:set GOOGLE_API_KEY=your_key
heroku config:set PINECONE_API_KEY=your_key
heroku config:set TAVILY_API_KEY=your_key
git push heroku main
```

### 2. Railway

1. Connect GitHub repository to Railway
2. Add environment variables in Railway dashboard
3. Deploy automatically from GitHub

### 3. Render

1. Connect GitHub repository to Render
2. Choose "Web Service"
3. Use Docker runtime
4. Add environment variables

### 4. DigitalOcean App Platform

1. Connect GitHub repository
2. Use Dockerfile for deployment
3. Configure environment variables

## 🔐 Security Best Practices

### Environment Variables

- ✅ Use `.env.docker` as template
- ✅ Never commit real API keys to Git
- ✅ Use different keys for development/production
- ✅ Set up GitHub repository secrets

### Docker Security

- ✅ Use non-root user in production
- ✅ Scan images for vulnerabilities
- ✅ Keep base images updated
- ✅ Use multi-stage builds for smaller images

## 🌐 Production Configuration

### Environment-Specific Settings

```bash
# Production
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
LOG_LEVEL=WARNING
PYTHONUNBUFFERED=1

# Development
LOG_LEVEL=DEBUG
STREAMLIT_DEBUG=true
```

## 📊 Monitoring and Logs

### Docker Logs

```bash
# View logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f post-discharge-assistant
```

### Health Checks

The application includes health checks at:

- `http://your-app-url/_stcore/health`

## 🔄 CI/CD with GitHub Actions

See `.github/workflows/` directory for automated:

- Testing
- Building
- Deployment
- Security scanning

## 📝 Deployment Checklist

- [ ] Repository created on GitHub
- [ ] All files committed and pushed
- [ ] `.env` file configured with real API keys
- [ ] Repository secrets set up (if using CI/CD)
- [ ] Docker images build successfully
- [ ] Application accessible at deployed URL
- [ ] Health checks passing
- [ ] API integrations working
- [ ] Database connections functional

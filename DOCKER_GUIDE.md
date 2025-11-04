# 🐳 Docker Setup Guide for Post-Discharge Assistant

This guide will help you quickly dockerize and run your Post-Discharge Assistant project.

## 📁 Files Created

The following Docker-related files have been created in your project:

### Core Docker Files

- `Dockerfile` - Production container configuration
- `Dockerfile.dev` - Development container with hot reload
- `docker-compose.yml` - Production orchestration
- `docker-compose.dev.yml` - Development orchestration
- `.dockerignore` - Files to exclude from Docker build
- `.env.docker` - Environment template for Docker

### Helper Scripts

- `run-docker.bat` - Windows batch script for easy setup
- `run-docker.sh` - Unix/Linux shell script for easy setup
- `DOCKER_GUIDE.md` - This guide

## 🚀 Quick Start

### Option 1: Using Docker Compose (Recommended)

1. **Setup environment variables:**

   ```bash
   # Copy template
   cp .env.docker .env

   # Edit with your API keys
   # GOOGLE_API_KEY=your_actual_key
   # PINECONE_API_KEY=your_actual_key
   # TAVILY_API_KEY=your_actual_key
   ```

2. **Run production setup:**

   ```bash
   docker-compose up --build
   ```

3. **Run development setup (with hot reload):**
   ```bash
   docker-compose -f docker-compose.dev.yml up --build
   ```

### Option 2: Using Helper Scripts

**Windows:**

```cmd
run-docker.bat
```

**Linux/Mac:**

```bash
chmod +x run-docker.sh
./run-docker.sh
```

### Option 3: Manual Docker Commands

1. **Build the image:**

   ```bash
   docker build -t post-discharge-assistant .
   ```

2. **Run the container:**
   ```bash
   docker run -p 8501:8501 --env-file .env post-discharge-assistant
   ```

## 🌐 Access Your Application

Once running, access your application at:

- **URL:** http://localhost:8501
- **Health Check:** http://localhost:8501/\_stcore/health

## 📊 Container Features

### Production Container (`Dockerfile`)

- Optimized Python 3.11 slim image
- Security hardened
- Health checks included
- Minimal attack surface
- Production-ready

### Development Container (`Dockerfile.dev`)

- Hot reload enabled
- Source code mounting
- Development optimizations
- Faster iteration cycles

## 🔧 Configuration

### Environment Variables

All configuration is handled through `.env` file:

```bash
# API Keys (Required)
GOOGLE_API_KEY=your_key_here
PINECONE_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here

# Pinecone Settings
PINECONE_ENVIRONMENT=us-west1-gcp
PINECONE_INDEX_NAME=nephrology-knowledge

# Application Settings
LOG_LEVEL=INFO
TEMPERATURE=0.3
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

### Volume Mounts

The following directories are persisted:

- `./data` - Patient data and databases
- `./logs` - Application logs
- `./vectorstore` - Vector database storage

## 🛠️ Development Workflow

### For Development with Hot Reload:

```bash
# Start development environment
docker-compose -f docker-compose.dev.yml up --build

# Make changes to your code in src/
# Changes will automatically reload in the container
```

### For Production Testing:

```bash
# Start production environment
docker-compose up --build

# Test in production-like environment
```

## 📝 Common Commands

```bash
# View running containers
docker ps

# View logs
docker-compose logs -f

# Stop containers
docker-compose down

# Remove containers and images
docker-compose down --rmi all

# Access container shell
docker-compose exec post-discharge-assistant bash

# Rebuild after major changes
docker-compose up --build --force-recreate
```

## 🐛 Troubleshooting

### Port Already in Use

Change port in `docker-compose.yml`:

```yaml
ports:
  - "8502:8501" # Changed from 8501:8501
```

### Permission Issues (Linux/Mac)

```bash
sudo docker-compose up --build
```

### Memory Issues

Increase Docker Desktop memory:

- Docker Desktop → Settings → Resources → Memory
- Increase to at least 4GB

### API Key Issues

1. Verify `.env` file exists and has correct keys
2. Check file is in project root directory
3. Ensure no extra spaces around `=` in env file

### Container Won't Start

1. Check Docker Desktop is running
2. Verify all required files are present
3. Check logs: `docker-compose logs`

## 🔄 Updates and Maintenance

### Updating Dependencies

1. Update `requirements.txt`
2. Rebuild containers: `docker-compose up --build`

### Updating Configuration

1. Modify `.env` file
2. Restart containers: `docker-compose restart`

### Database Migrations

Volumes persist data, so databases survive container restarts.

## 🎯 Next Steps

1. **Setup API Keys:** Get your free API keys and update `.env`
2. **Run Setup:** Use one of the quick start methods above
3. **Test Application:** Visit http://localhost:8501
4. **Customize:** Modify configuration as needed
5. **Deploy:** Use production Docker setup for deployment

## 📞 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review container logs: `docker-compose logs`
3. Ensure all prerequisites are installed
4. Verify API keys are correct

Happy containerizing! 🐳

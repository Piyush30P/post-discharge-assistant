# 🏥 Post-Discharge Medical AI Assistant

**A multi-agent AI system for post-discharge patient care with RAG-powered medical knowledge base**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32.0-FF4B4B.svg)](https://streamlit.io)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.0.35-green.svg)](https://github.com/langchain-ai/langgraph)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Docker Setup](#docker-setup)
- [GitHub Setup](#github-setup)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [API Keys Setup](#api-keys-setup)
- [Deployment](#deployment)
- [Demo](#demo)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

The **Post-Discharge Medical AI Assistant** is an intelligent multi-agent system designed to help patients after hospital discharge. It provides:

- **Personalized Care**: Retrieves patient-specific discharge information
- **Medical Guidance**: Answers questions using a comprehensive nephrology knowledge base
- **Current Research**: Searches the web for latest medical information
- **Smart Routing**: Automatically routes queries to specialized AI agents
- **Evidence-Based**: All medical advice includes citations and disclaimers

### 🎥 Demo Video

[Watch the 5-minute demo](https://youtu.be/YOUR_VIDEO_LINK) _(Coming soon)_

---

## ✨ Features

### 🤖 Multi-Agent System

- **Receptionist Agent**: Greets patients, retrieves discharge reports, handles basic queries
- **Clinical Agent**: Provides medical information using RAG and web search
- **Smart Handoff**: Automatically routes complex medical queries to Clinical Agent

### 🧠 RAG Implementation

- **Vector Database**: Pinecone cloud-based storage (~4000 medical text chunks)
- **Embeddings**: Sentence-Transformers (all-MiniLM-L6-v2)
- **Source Material**: Comprehensive Clinical Nephrology 7th Edition
- **Intelligent Chunking**: Sentence-aware splitting for better semantic meaning

### 🌐 Web Search Integration

- **Provider**: Tavily API for reliable, rate-limit-free searches
- **Use Case**: Latest research, current guidelines, recent medical news
- **Smart Fallback**: Uses web search when knowledge base doesn't have current info

### 💾 Patient Data Management

- **Database**: SQLite with 30+ dummy patient records
- **Data**: Discharge reports, medications, dietary restrictions, follow-ups
- **Privacy**: All data is dummy/synthetic for demonstration purposes

### 📊 Comprehensive Logging

- **All Operations**: Database queries, tool calls, agent actions, errors
- **Debugging**: Detailed logs for troubleshooting
- **Analytics**: Track user interactions and system performance

### 🎨 User Interface

- **Framework**: Streamlit for rapid prototyping
- **Features**: Chat interface, conversation history, session management
- **Responsive**: Works on desktop and mobile browsers

---

## 🏗️ Architecture

# Architecture Overview

<details>
<summary>📊 Click to view System Architecture</summary>

![System Architecture](Architecture%20diagram/architecture_diagram.jpg)

</details>

<details>
<summary>🔄 Click to view Data Flow</summary>

![Data flow Flowchart](Architecture%20diagram/data_flow_diagram.jpg)

</details>

<details>
<summary>🔄 Click to view Workflow</summary>

![Workflow Flowchart](Architecture%20diagram/workflow_flowchart.jpg)

</details>

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit Frontend                       │
│                    (src/app.py)                             │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              LangGraph Workflow Orchestrator                │
│                 (src/workflow/graph.py)                     │
│  ┌──────────────────────┐    ┌──────────────────────────┐  │
│  │ Receptionist Agent   │───▶│   Clinical Agent         │  │
│  │ (Basic Info)         │    │   (Medical Queries)      │  │
│  └──────────────────────┘    └──────────────────────────┘  │
└────────────────────┬────────────────────┬───────────────────┘
                     │                     │
                     ▼                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    MCP Tools Layer                          │
│                   (src/mcp/tools.py)                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐ │
│  │   Database   │  │     RAG      │  │   Web Search     │ │
│  │   (SQLite)   │  │  (Pinecone)  │  │    (Tavily)      │ │
│  └──────────────┘  └──────────────┘  └──────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Key Components:

1. **Frontend Layer**: Streamlit UI for user interactions
2. **Orchestration Layer**: LangGraph manages multi-agent workflow
3. **Agent Layer**: Specialized agents for different tasks
4. **Tool Layer**: Database, RAG, and web search capabilities
5. **Logging Layer**: Comprehensive system monitoring

---

## 🛠️ Technology Stack

| Category           | Technology              | Purpose                         |
| ------------------ | ----------------------- | ------------------------------- |
| **LLM**            | Google Gemini 2.0 Flash | Fast, cost-effective reasoning  |
| **Framework**      | LangChain + LangGraph   | Agent orchestration             |
| **Vector DB**      | Pinecone                | Semantic search (serverless)    |
| **Embeddings**     | Sentence-Transformers   | Text vectorization              |
| **Web Search**     | Tavily API              | Current information retrieval   |
| **Database**       | SQLite                  | Patient data storage            |
| **Frontend**       | Streamlit               | Rapid UI development            |
| **PDF Processing** | PyMuPDF                 | Extract text from medical books |
| **Logging**        | Python logging          | System monitoring               |

---

## 📦 Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Git

### Step 1: Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/post-discharge-assistant.git
cd post-discharge-assistant
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set Up Environment Variables

Copy `.env.example` to `.env` and add your API keys:

```bash
cp .env.example .env
```

Edit `.env`:

```bash
# Google Gemini API Key
GOOGLE_API_KEY=your_gemini_api_key_here

# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=us-west1-gcp
PINECONE_INDEX_NAME=nephrology-knowledge

# Tavily API Key (for web search)
TAVILY_API_KEY=your_tavily_api_key_here
```

---

## 🐳 Docker Setup

### Prerequisites for Docker

- Docker Desktop installed on your system
- Docker Compose (included with Docker Desktop)

### Quick Start with Docker

1. **Clone the repository:**

```bash
git clone https://github.com/YOUR_USERNAME/post-discharge-assistant.git
cd post-discharge-assistant
```

2. **Set up environment variables:**

```bash
# Copy the Docker environment template
cp .env.docker .env

# Edit .env with your actual API keys
# GOOGLE_API_KEY=your_actual_key_here
# PINECONE_API_KEY=your_actual_key_here
# TAVILY_API_KEY=your_actual_key_here
```

3. **Build and run with Docker Compose:**

```bash
# Production setup
docker-compose up --build

# Development setup with hot reload
docker-compose -f docker-compose.dev.yml up --build
```

4. **Access the application:**
   - Open your browser and go to: `http://localhost:8501`

### Docker Commands Reference

```bash
# Build the Docker image
docker build -t post-discharge-assistant .

# Run the container directly
docker run -p 8501:8501 --env-file .env post-discharge-assistant

# Stop all containers
docker-compose down

# View logs
docker-compose logs -f

# Rebuild after changes
docker-compose up --build

# Run in detached mode
docker-compose up -d
```

### Development with Docker

For development with hot reload:

```bash
# Use the development docker-compose file
docker-compose -f docker-compose.dev.yml up --build
```

This setup:

- Mounts your source code for instant changes
- Enables file watching for auto-reload
- Persists data in `./data`, `./logs`, and `./vectorstore` directories

### Docker Environment Variables

The Docker setup uses the same environment variables as the local setup. Make sure to update `.env` with your actual API keys:

```bash
# Required API Keys
GOOGLE_API_KEY=your_gemini_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here

# Optional configurations
LOG_LEVEL=INFO
TEMPERATURE=0.3
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

### Troubleshooting Docker

- **Port already in use**: Change the port mapping in `docker-compose.yml` from `"8501:8501"` to `"8502:8501"`
- **Permission issues**: On Linux/Mac, you might need to run with `sudo`
- **API key issues**: Ensure your `.env` file has the correct API keys and is in the project root
- **Memory issues**: Increase Docker Desktop memory allocation in settings

---

## 🐙 GitHub Setup

### Quick Setup with Helper Script

**Windows:**

```cmd
setup-github.bat
```

**Linux/Mac:**

```bash
chmod +x setup-github.sh
./setup-github.sh
```

### Manual Setup

1. **Create GitHub Repository:**

   - Go to [GitHub](https://github.com) and create a new repository
   - Name: `post-discharge-assistant`
   - Make it public or private
   - Don't initialize with README

2. **Push to GitHub:**

```bash
# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: Post-Discharge Medical AI Assistant"

# Add remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/post-discharge-assistant.git

# Push to GitHub
git push -u origin main
```

3. **Set Repository Secrets (for deployment):**
   - Go to repository → Settings → Secrets and variables → Actions
   - Add secrets:
     - `GOOGLE_API_KEY`
     - `PINECONE_API_KEY`
     - `TAVILY_API_KEY`

### GitHub Features Included

- ✅ **Automated CI/CD** with GitHub Actions
- ✅ **Docker image building** and security scanning
- ✅ **Multi-platform deployment** workflows
- ✅ **Comprehensive documentation**

---

## 🔑 API Keys Setup

### 1. Google Gemini API Key (FREE)

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Click "Get API Key"
3. Copy the key to `.env`

### 2. Pinecone API Key (FREE)

1. Go to [Pinecone](https://app.pinecone.io/)
2. Sign up for free account
3. Create a new project
4. Copy API key to `.env`

### 3. Tavily API Key (FREE)

1. Go to [Tavily](https://tavily.com/)
2. Sign up for free account (1000 searches/month)
3. Copy API key to `.env`

---

## ⚙️ Configuration

### Phase 1: Data Setup

Generate patient data and set up vector database:

```bash
python setup_phase1.py
```

This will:

- ✅ Generate 30 dummy patient records
- ✅ Populate SQLite database
- ✅ Process PDF (if available)
- ✅ Upload vectors to Pinecone

**Note**: PDF processing requires a nephrology textbook PDF at `data/nephrology_book.pdf`

### Phase 1.5: MCP Server Setup (Optional)

Set up Model Context Protocol server for enhanced web search:

```bash
python setup_mcp.py
```

This will:

- ✅ Test MCP web search server
- ✅ Verify medical news search
- ✅ Test integration with clinical agent
- ✅ Enable advanced medical source prioritization

**Benefits of MCP Setup:**

- 🔍 Dedicated web search server
- 🏥 Medical source prioritization
- 📰 Specialized medical news search
- ⚡ Improved search performance
- 🔄 Automatic fallback to direct search

### Phase 2: Agent Setup

Install and verify multi-agent system:

```bash
python setup_phase2.py
```

This will:

- ✅ Install Phase 2 dependencies
- ✅ Verify agent initialization
- ✅ Test workflow orchestration

---

## 🚀 Usage

### Running the Application

```bash
streamlit run src/app.py
```

The app will open in your browser at `http://localhost:8501`

### Example Conversations

**1. Patient Greeting**

```
User: Hello, my name is Ashley King
Assistant: Hi Ashley King! I found your discharge report from
August 16, 2025, for Kidney Stones. How are you feeling today?
```

**2. Medication Query**

```
User: What are my medications?
Assistant: Your medications are:
- Tamsulosin 0.4mg daily
- Ketorolac 10mg PRN for pain
- Potassium citrate 10mEq twice daily
```

**3. Medical Question (RAG)**

```
User: What causes kidney stones?
Assistant: According to Comprehensive Clinical Nephrology (page 456),
kidney stones form when urine contains high levels of crystal-forming
substances... [detailed answer with citations]
```

**4. Current Research (Web Search)**

```
User: What's the latest research on SGLT2 inhibitors?
Assistant: According to recent medical literature from Northwestern
University and the National Kidney Foundation, SGLT2 inhibitors have
shown promising results... [current information with sources]
```

---

## 📁 Project Structure

```
post-discharge-assistant/
│
├── data/                          # Data directory
│   ├── patient_reports.json       # 30 dummy patient records
│   ├── patients.db                # SQLite database
│   |── nephrology_book.pdf        # Source material (not included)
|   |── chunks_preview.txt
│
├── logs/                          # System logs
│   └── system_logs.txt            # Comprehensive logging
│
├── src/                           # Source code
│   ├── agents/                    # AI Agents
│   │   ├── receptionist_agent.py  # Patient greeting & basic queries
│   │   ├── clinical_agent.py      # Medical queries with RAG
│   │   └── prompts.py             # Agent system prompts
│   │
│   ├── mcp/                       # MCP Tools
│   │   └── tools.py               # Database, RAG, web search tools
│   │
│   ├── workflow/                  # LangGraph Workflow
│   │   ├── graph.py               # Multi-agent orchestration
│   │   └── state.py               # Workflow state management
│   │
│   ├── utils/                     # Utilities
│   │   └── logger.py              # Logging system
│   │
│   ├── app.py                     # Streamlit UI
│   ├── config.py                  # Configuration management
│   ├── database.py                # SQLite database manager
│   ├── pinecone_manager.py        # Pinecone vector DB manager
│   ├── pdf_processor_enhanced.py  # PDF processing with chunking
│   └── generate_dummy_data.py     # Patient data generator
│
├── setup_phase1.py                # Phase 1 setup script
├── setup_phase2.py                # Phase 2 setup script
├── verify_phase1.py               # Phase 1 verification
├── verify_phase2.py               # Phase 2 verification
│
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment variables template
├── .gitignore                     # Git ignore rules
├── README.md                      # This file
├── REPORT.md                      # Architecture justification report
└── LICENSE                        # MIT License
```

---

## 🧪 Testing

### Verify Phase 1 (Data Setup)

```bash
python verify_phase1.py
```

Checks:

- ✅ Patient data generated
- ✅ Database populated
- ✅ Pinecone index created
- ✅ Search functionality working

### Verify Phase 2 (Multi-Agent System)

```bash
python verify_phase2.py
```

Checks:

- ✅ All modules import correctly
- ✅ MCP tools working
- ✅ Agents initialized
- ✅ Workflow functional

### Manual Testing

Test the complete system with these queries:

1. **Patient greeting**: "Hello, my name is [Patient Name]"
2. **Basic query**: "What are my medications?"
3. **Medical question**: "What causes kidney stones?"
4. **Symptom**: "I'm having leg swelling, should I be worried?"
5. **Current research**: "Latest research on kidney transplants?"

---

## 📊 System Metrics

| Metric              | Value             |
| ------------------- | ----------------- |
| Patient Records     | 30                |
| Vector DB Size      | ~4,000 chunks     |
| Avg Chunk Size      | ~1,000 characters |
| Database Size       | ~100 KB           |
| Embedding Dimension | 384               |
| Response Time (RAG) | 2-3 seconds       |
| Response Time (Web) | 3-5 seconds       |

---

## 🎓 Architecture Justification

See [REPORT.md](REPORT.md) for detailed architecture justification including:

- LLM Selection (Google Gemini)
- Vector Database (Pinecone)
- Multi-Agent Framework (LangGraph)
- Web Search Integration (Tavily)
- Logging Implementation

---

## 🐛 Troubleshooting

### Issue: "API Key not found"

**Solution**: Ensure all API keys are set in `.env` file

```bash
# Check if .env exists
cat .env

# Verify keys are set
python -c "from src.config import validate_config; validate_config()"
```

### Issue: "Pinecone connection failed"

**Solution**: Check Pinecone API key and index name

```bash
python verify_phase1.py
```

### Issue: "Web search failed"

**Solution**: Verify Tavily API key

```bash
# Test Tavily
python -c "from src.mcp.tools import MCPTools; tools = MCPTools(); print(tools.web_search('test'))"
```

### Issue: "Slow responses"

**Possible causes**:

- First query loads models (5-7 seconds)
- Pinecone cold start
- Large PDF chunking

---

## 🔒 Privacy & Security

⚠️ **Important**: This is a **demonstration project** with dummy data only.

**For production use**:

- [ ] Implement proper authentication
- [ ] Use encrypted database
- [ ] HIPAA compliance measures
- [ ] Secure API key management
- [ ] Input sanitization
- [ ] Rate limiting

---

## 📝 Medical Disclaimer

```
⚕️ IMPORTANT MEDICAL DISCLAIMER

This is an AI assistant for EDUCATIONAL PURPOSES ONLY.

- NOT a substitute for professional medical advice
- NOT for use in medical emergencies (call 911)
- All medical decisions should be made with healthcare providers
- Information may not be complete or up-to-date
- Always verify information with qualified medical professionals
```

---

---

## � Deployment

### Cloud Deployment Options

#### 1. Heroku (Easy)

```bash
heroku create your-app-name
heroku stack:set container
heroku config:set GOOGLE_API_KEY=your_key
heroku config:set PINECONE_API_KEY=your_key
heroku config:set TAVILY_API_KEY=your_key
git push heroku main
```

#### 2. Railway (Recommended)

1. Connect GitHub repository to Railway
2. Add environment variables in dashboard
3. Deploy automatically from GitHub

#### 3. Render

1. Connect GitHub repository
2. Choose "Web Service"
3. Use Docker runtime
4. Add environment variables

#### 4. DigitalOcean App Platform

1. Connect GitHub repository
2. Use Dockerfile for deployment
3. Configure environment variables

### Self-Hosted Deployment

```bash
# On your server
git clone https://github.com/YOUR_USERNAME/post-discharge-assistant.git
cd post-discharge-assistant
cp .env.docker .env
# Edit .env with your API keys
docker-compose up -d
```

For detailed deployment instructions, see [`DEPLOYMENT.md`](DEPLOYMENT.md).

---

## �📜 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Your Name**

- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)
- Email: your.email@example.com

---

## 🙏 Acknowledgments

- **Anthropic Claude** for coding assistance
- **Google Gemini** for LLM capabilities
- **Pinecone** for vector database
- **Tavily** for web search API
- **LangChain Team** for agent framework
- **Comprehensive Clinical Nephrology 7th Edition** for medical knowledge

---

## 📚 References

1. Comprehensive Clinical Nephrology, 7th Edition
2. [LangChain Documentation](https://python.langchain.com/)
3. [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
4. [Pinecone Documentation](https://docs.pinecone.io/)
5. [Tavily API Documentation](https://docs.tavily.com/)
6. [Streamlit Documentation](https://docs.streamlit.io/)

---

## 📈 Future Enhancements

- [ ] Voice interface for hands-free operation
- [ ] Multi-language support (Spanish, Chinese, etc.)
- [ ] Mobile app (React Native)
- [ ] Integration with EHR systems
- [ ] Appointment scheduling
- [ ] Medication reminders
- [ ] Symptom tracking over time
- [ ] Family member access
- [ ] Telemedicine integration

---

## 📞 Support

If you have questions or need help:

1. Check the [Troubleshooting](#troubleshooting) section
2. Open an [Issue](https://github.com/yourusername/post-discharge-assistant/issues)
3. Email: support@example.com

---

**Made with ❤️ for better patient care**

---

_Last Updated: October 2025_

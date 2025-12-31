# 🏠 Local LLM Setup Guide

## Why Use Local LLMs?

**Problem:** Gemini free tier has strict quotas (20 requests/day)

**Solution:** Use local open-source LLMs for simple tasks, reserve Gemini for complex medical reasoning

### Benefits:
- ✅ **No API quota limits** - Unlimited query transformations & summaries
- ✅ **Faster** - No network latency for simple tasks
- ✅ **Privacy** - Data stays on your machine
- ✅ **Cost savings** - Reserve paid API for complex tasks only

---

## 🎯 Hybrid LLM Strategy

| Task | LLM Used | Why |
|------|----------|-----|
| **Query Expansion** | Local (Mistral/Llama) | Simple text generation, no medical expertise needed |
| **Query Decomposition** | Local (Mistral/Llama) | Pattern matching, no medical knowledge required |
| **Keyword Extraction** | Local (Mistral/Llama) | Simple NLP task |
| **Summary Generation** | Local (Mistral/Llama) | Text summarization, straightforward |
| **Medical Reasoning** | Cloud (Gemini) | Complex medical knowledge, diagnosis support |
| **Clinical Responses** | Cloud (Gemini) | Accurate medical information critical |

---

## 📥 Step 1: Install Ollama

### Windows

```powershell
# Download Ollama for Windows
# Visit: https://ollama.com/download/windows
# Run the installer

# Verify installation
ollama --version
```

### Linux/Mac

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Verify installation
ollama --version
```

---

## 📦 Step 2: Pull a Model

Choose one of these models:

### Option A: Mistral 7B (Recommended - Fast & Good Quality)

```powershell
ollama pull mistral:7b-instruct
```

**Stats:**
- Size: ~4.1 GB
- Speed: Fast on CPU
- Quality: Excellent for query transformation

### Option B: Llama 3.2 3B (Lighter, Faster)

```powershell
ollama pull llama3.2:3b
```

**Stats:**
- Size: ~2 GB
- Speed: Very fast
- Quality: Good for simple tasks

### Option C: Llama 3.2 1B (Smallest)

```powershell
ollama pull llama3.2:1b
```

**Stats:**
- Size: ~1.3 GB
- Speed: Extremely fast
- Quality: Decent for basic tasks

---

## ⚙️ Step 3: Configure Environment

Add to your `.env` file:

```env
# Local LLM Configuration (Ollama)
USE_LOCAL_FOR_SIMPLE_TASKS=true
LOCAL_LLM_MODEL=mistral:7b-instruct
OLLAMA_BASE_URL=http://localhost:11434

# Cloud LLM (Gemini) - for complex tasks only
GOOGLE_API_KEY=your_gemini_api_key_here
CLOUD_LLM_MODEL=gemini-2.5-flash
```

---

## 🧪 Step 4: Test Local LLM

```powershell
# Test Ollama is running
python -c "from src.llm_config import HybridLLMConfig; print('✅ Local LLM available!' if HybridLLMConfig.is_local_available() else '❌ Local LLM not available')"

# Test query transformation with local LLM
python << 'EOF'
from src.llm_config import get_query_transformation_llm

llm = get_query_transformation_llm()
result = llm.invoke("Expand this medical query: kidney disease symptoms")
print("✅ Local LLM working!")
print(f"Response: {result[:100]}...")
EOF
```

---

## 🚀 Step 5: Run Advanced RAG Setup

Now run the setup - it will use local LLM for summaries:

```powershell
# Install langchain-community
pip install langchain-community

# Run setup (will use local LLM, no Gemini quota!)
python setup_advanced_rag.py
```

You should see:
```
Using local LLM for query transformation
Using local LLM for summary generation
✓ Summary index built successfully (no quota limits!)
```

---

## 🔧 Model Comparison

| Model | Size | RAM Needed | Speed | Quality | Best For |
|-------|------|------------|-------|---------|----------|
| **mistral:7b-instruct** | 4.1 GB | 8 GB | Fast | ⭐⭐⭐⭐⭐ | General use (recommended) |
| **llama3.2:3b** | 2 GB | 4 GB | Very Fast | ⭐⭐⭐⭐ | Low-end machines |
| **llama3.2:1b** | 1.3 GB | 2 GB | Extremely Fast | ⭐⭐⭐ | Minimal resources |
| **llama3.1:8b** | 4.7 GB | 8 GB | Fast | ⭐⭐⭐⭐⭐ | Best quality |

---

## 📊 Performance Comparison

### Before (Cloud Only):
```
Query Transformation: 20/day limit ❌
Summary Generation: 20/day limit ❌
Total Cost: Free tier exhausted quickly
```

### After (Hybrid):
```
Query Transformation: Unlimited ✅ (local LLM)
Summary Generation: Unlimited ✅ (local LLM)
Medical Responses: 20/day (Gemini) ✅
Total Cost: Better resource utilization
```

---

## 🛠️ Troubleshooting

### "Ollama not found"

**Solution:**
```powershell
# Check if Ollama is running
ollama list

# If not, start it
ollama serve
```

### "Connection refused to localhost:11434"

**Solution:**
```powershell
# Start Ollama service
ollama serve

# In another terminal, test
ollama run mistral:7b-instruct "Hello"
```

### "Model not found"

**Solution:**
```powershell
# List available models
ollama list

# Pull the model
ollama pull mistral:7b-instruct
```

### "Out of memory"

**Solution:**
```powershell
# Switch to smaller model
ollama pull llama3.2:3b

# Update .env
LOCAL_LLM_MODEL=llama3.2:3b
```

### "Still using Gemini for simple tasks"

**Check .env:**
```env
USE_LOCAL_FOR_SIMPLE_TASKS=true  # Must be "true"
```

---

## 🔄 Switching Between Local and Cloud

### Force Local LLM

```python
from src.llm_config import HybridLLMConfig

# Get local LLM only
llm = HybridLLMConfig.get_local_llm()
```

### Force Cloud LLM

```python
from src.llm_config import HybridLLMConfig

# Get cloud LLM only
llm = HybridLLMConfig.get_cloud_llm()
```

### Auto (Hybrid - Recommended)

```python
from src.llm_config import get_query_transformation_llm

# Auto-selects local if available, falls back to cloud
llm = get_query_transformation_llm()
```

---

## 💡 Advanced Configuration

### Use Different Models for Different Tasks

Edit `.env`:

```env
# Fast model for queries
LOCAL_LLM_MODEL=llama3.2:3b

# Better model for summaries (if you have RAM)
# (implement custom selection in llm_config.py)
```

### Adjust Temperature

```python
from src.llm_config import get_query_transformation_llm

# More creative
llm = get_query_transformation_llm(temperature=0.9)

# More deterministic
llm = get_query_transformation_llm(temperature=0.1)
```

---

## 📈 Resource Usage

### Expected Resource Usage:

| Model | Disk Space | RAM (Idle) | RAM (Active) | CPU Usage |
|-------|------------|------------|--------------|-----------|
| mistral:7b | 4.1 GB | ~100 MB | ~6 GB | Medium |
| llama3.2:3b | 2 GB | ~50 MB | ~3 GB | Low |
| llama3.2:1b | 1.3 GB | ~30 MB | ~1.5 GB | Very Low |

### Minimum System Requirements:

- **For mistral:7b-instruct**: 8 GB RAM, 5 GB free disk
- **For llama3.2:3b**: 4 GB RAM, 3 GB free disk
- **For llama3.2:1b**: 2 GB RAM, 2 GB free disk

---

## 🎯 Recommended Setup

**For most users:**

```powershell
# 1. Install Ollama
# Download from https://ollama.com/download/windows

# 2. Pull Mistral (best balance)
ollama pull mistral:7b-instruct

# 3. Update .env
USE_LOCAL_FOR_SIMPLE_TASKS=true
LOCAL_LLM_MODEL=mistral:7b-instruct

# 4. Install dependency
pip install langchain-community

# 5. Run setup
python setup_advanced_rag.py
```

**For low-end machines:**

```powershell
# Use smaller model
ollama pull llama3.2:3b

# Update .env
LOCAL_LLM_MODEL=llama3.2:3b
```

---

## ✅ Verification Checklist

- [ ] Ollama installed (`ollama --version`)
- [ ] Model downloaded (`ollama list`)
- [ ] Ollama running (`ollama serve`)
- [ ] `.env` configured
- [ ] `langchain-community` installed
- [ ] Local LLM test passed
- [ ] Advanced RAG setup completed

---

## 🆘 Getting Help

**Check logs:**
```powershell
# See what LLM is being used
python -c "from src.llm_config import HybridLLMConfig; print('Local available:', HybridLLMConfig.is_local_available())"
```

**Test connection:**
```powershell
# Test Ollama directly
ollama run mistral:7b-instruct "Hello, how are you?"
```

**Fallback to cloud:**
```env
# Temporarily disable local LLM
USE_LOCAL_FOR_SIMPLE_TASKS=false
```

---

## 🎉 Success!

Once set up, you can:
- ✅ Generate unlimited query transformations
- ✅ Build summary index without quota limits
- ✅ Reserve Gemini for important medical responses
- ✅ Run Advanced RAG system efficiently

**Enjoy unlimited Advanced RAG capabilities!** 🚀

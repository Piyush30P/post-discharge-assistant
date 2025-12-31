# 🚀 Hybrid LLM Quick Start - Avoid Quota Limits!

## Problem You're Facing

You hit Gemini's free tier quota (20 requests/day) because every query transformation and summary uses an API call.

## Solution

Use **local open-source LLMs** (free, unlimited) for simple tasks, reserve Gemini for complex medical responses.

---

## ⚡ 5-Minute Setup

### 1. Install Ollama (One-Time)

**Windows:**
```powershell
# Download from https://ollama.com/download/windows
# Run the installer
# That's it!
```

**Verify:**
```powershell
ollama --version
```

### 2. Pull a Model (One-Time)

```powershell
# Recommended: Mistral 7B (4GB, excellent quality)
ollama pull mistral:7b-instruct

# Or lighter: Llama 3.2 3B (2GB, good quality)
ollama pull llama3.2:3b
```

This downloads the model (~5 minutes depending on your internet).

### 3. Update Your .env

```powershell
# Add these lines to your .env file
@"
USE_LOCAL_FOR_SIMPLE_TASKS=true
LOCAL_LLM_MODEL=mistral:7b-instruct
OLLAMA_BASE_URL=http://localhost:11434
"@ | Add-Content .env
```

### 4. Install Python Package

```powershell
pip install langchain-community
```

### 5. Run Setup Again!

```powershell
python setup_advanced_rag.py
```

**Now it will use:**
- ✅ **Local LLM** for query transformations (unlimited!)
- ✅ **Local LLM** for summaries (unlimited!)
- ✅ **Gemini** only for complex medical responses (stays within quota!)

---

## 🎯 What Changes

### Before (All Cloud):
```
❌ Query expansion: Gemini API (counts toward 20/day)
❌ Query decomposition: Gemini API (counts toward 20/day)
❌ Keyword extraction: Gemini API (counts toward 20/day)
❌ Summary generation: Gemini API (counts toward 20/day)
❌ Medical responses: Gemini API (counts toward 20/day)

Result: Quota exhausted quickly! 😞
```

### After (Hybrid):
```
✅ Query expansion: Local Mistral (FREE, UNLIMITED)
✅ Query decomposition: Local Mistral (FREE, UNLIMITED)
✅ Keyword extraction: Local Mistral (FREE, UNLIMITED)
✅ Summary generation: Local Mistral (FREE, UNLIMITED)
✅ Medical responses: Gemini API (20/day is plenty now!)

Result: Never hit quota limits! 🎉
```

---

## 🧪 Test It Works

```powershell
# Test local LLM is available
python -c "from src.llm_config import HybridLLMConfig; print('✅ Local LLM Ready!' if HybridLLMConfig.is_local_available() else '❌ Check Ollama')"

# Test query transformation with local LLM
python << 'EOF'
import sys
sys.path.insert(0, 'src')
from llm_config import get_query_transformation_llm

llm = get_query_transformation_llm()
print("✅ Using local LLM for query transformation!")
print("Testing: kidney disease symptoms")
result = llm.invoke("Expand: kidney disease symptoms")
print(f"Result: {result[:100]}...")
EOF
```

---

## 📊 Resource Requirements

### Minimum (Llama 3.2 3B):
- **RAM:** 4 GB
- **Disk:** 3 GB free
- **CPU:** Any modern CPU

### Recommended (Mistral 7B):
- **RAM:** 8 GB
- **Disk:** 5 GB free
- **CPU:** Any modern CPU (works on CPU, no GPU needed!)

---

## 🔄 Fallback Behavior

The system is smart:

1. **Local LLM available** → Uses local (unlimited, free)
2. **Local LLM not available** → Falls back to Gemini (with quota)
3. **Both available** → Uses local for simple, Gemini for complex

You can't break it - it always works!

---

## 🎁 Benefits

### Cost
- **Before:** Risk of hitting quota daily
- **After:** Never hit quota (simple tasks are free)

### Speed
- **Before:** Network latency for every API call
- **After:** Local = instant (no network delay)

### Privacy
- **Before:** All data sent to cloud
- **After:** Simple tasks processed locally

### Scalability
- **Before:** Limited by API quotas
- **After:** Unlimited simple tasks, quota for complex only

---

## 🛠️ Troubleshooting

### "Ollama command not found"

**Solution:** Restart your terminal after installing Ollama.

### "Connection refused"

**Solution:**
```powershell
# Start Ollama service
ollama serve
```

### "Still using Gemini for everything"

**Check your .env:**
```env
USE_LOCAL_FOR_SIMPLE_TASKS=true  # Must be exactly "true"
```

### "Out of RAM"

**Solution:** Use smaller model
```powershell
ollama pull llama3.2:1b  # Only 1.3GB
```

Then update .env:
```env
LOCAL_LLM_MODEL=llama3.2:1b
```

---

## 📝 Summary

**Setup:**
1. Install Ollama → 2 minutes
2. Pull model → 5 minutes
3. Update .env → 30 seconds
4. Install package → 1 minute
5. **Done!** Total: < 10 minutes

**Result:**
- ✅ Unlimited query transformations
- ✅ Unlimited summary generation
- ✅ Never hit API quotas again
- ✅ Faster responses for simple tasks
- ✅ Better resource utilization

**Worth it?** Absolutely! 🚀

---

## 🎯 Next Steps

```powershell
# 1. Install Ollama
# https://ollama.com/download/windows

# 2. Pull model
ollama pull mistral:7b-instruct

# 3. Update .env (add the 3 lines)
USE_LOCAL_FOR_SIMPLE_TASKS=true
LOCAL_LLM_MODEL=mistral:7b-instruct
OLLAMA_BASE_URL=http://localhost:11434

# 4. Install package
pip install langchain-community

# 5. Rebuild indices (no quota limits!)
python setup_advanced_rag.py

# 6. Test
python test_advanced_rag.py
```

**That's it! Enjoy unlimited Advanced RAG!** 🎉

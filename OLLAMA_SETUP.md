# Ollama Setup Guide for Query Transformation

This guide explains how to set up Ollama for local query transformation, avoiding API rate limits while keeping main agents on cloud-based Gemini.

## Architecture

```
User Query
    ↓
Router Agent (Gemini) ──→ Receptionist Agent (Gemini)
    ↓                            ↓
Clinical Agent (Gemini)    Query Transformer (Ollama - LOCAL)
    ↓                            ↓
Retrieval Agent          [Llama/Mistral - NO RATE LIMITS]
```

## Why This Setup?

- **Main Agents (Gemini)**: Better for conversation, medical accuracy, tool calling
- **Query Transformation (Ollama)**:
  - Unlimited queries (runs locally)
  - No rate limits
  - Fast query decomposition
  - No API costs

## Installation Steps

### 1. Install Ollama

**Windows:**

```powershell
# Download from https://ollama.ai/download
# Or use winget
winget install Ollama.Ollama
```

**Mac:**

```bash
brew install ollama
```

**Linux:**

```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### 2. Start Ollama Service

```powershell
# Windows (runs automatically as service)
# Check if running:
curl http://localhost:11434

# Should return: "Ollama is running"
```

### 3. Pull Required Models

```powershell
# Option 1: Llama 3.2 (Recommended - 2GB, Fast)
ollama pull llama3.2

# Option 2: Mistral (Alternative - 4GB)
ollama pull mistral

# Option 3: Phi-3 (Lightweight - 1.5GB)
ollama pull phi3
```

### 4. Verify Installation

```powershell
# Test the model
ollama run llama3.2 "What is 2+2?"

# List installed models
ollama list
```

### 5. Install Python Package

```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install langchain-ollama
pip install langchain-ollama
```

## Configuration

The `QueryTransformer` class automatically:

1. Tries to connect to Ollama at `http://localhost:11434`
2. Falls back to simple rule-based transformation if Ollama unavailable
3. Uses Llama 3.2 by default

### Change Model

```python
# In your code
from src.query_transformer import QueryTransformer

# Use different model
transformer = QueryTransformer(model="mistral")  # or "phi3"
```

## Testing

```powershell
# Test query transformation
python test_query_transform.py
```

Expected output:

```
✓ QueryTransformer initialized with Ollama model: llama3.2
[QUERY_TRANSFORM] Decomposing: What medications should I take...
[QUERY_TRANSFORM] Decomposed into 3 sub-queries
```

## Model Comparison

| Model    | Size  | Speed     | Quality | Recommended For    |
| -------- | ----- | --------- | ------- | ------------------ |
| llama3.2 | 2GB   | Fast      | Good    | **Default choice** |
| mistral  | 4GB   | Medium    | Better  | Complex queries    |
| phi3     | 1.5GB | Very Fast | Good    | Limited resources  |

## Troubleshooting

### Ollama Not Running

```powershell
# Windows: Start Ollama Desktop app
# Or restart service:
Restart-Service Ollama
```

### Model Not Found

```powershell
ollama list  # Check installed models
ollama pull llama3.2  # Pull missing model
```

### Port Conflict

```powershell
# Check if port 11434 is in use
netstat -ano | findstr 11434

# Change Ollama port (in environment)
$env:OLLAMA_HOST="0.0.0.0:11435"
ollama serve
```

### Connection Refused

```powershell
# Verify Ollama is running
curl http://localhost:11434/api/version

# Should return version JSON
```

## Fallback Mode

If Ollama is not available, the system automatically uses **rule-based fallbacks**:

- **Decomposition**: Splits on "and", "or", commas
- **Multi-query**: Adds "What", "How", "When" variations
- **Rewrite**: Expands abbreviations, adds question marks

This ensures the system always works, even without Ollama!

## Performance Benefits

### Without Ollama (Using Gemini)

- Rate limits: 15 queries/minute
- Cost: $0.001 per query
- **Problem**: Hits limits with 5+ complex queries

### With Ollama (Local)

- Rate limits: **UNLIMITED** ✓
- Cost: **FREE** ✓
- Speed: **Faster** (local processing) ✓
- Privacy: **Data stays local** ✓

## Integration Status

✅ Query Transformer uses Ollama (llama3.2)  
✅ Receptionist Agent uses Gemini (cloud)  
✅ Clinical Agent uses Gemini (cloud)  
✅ Router Agent uses Gemini (cloud)  
✅ Automatic fallback if Ollama unavailable

## Next Steps

1. ✅ Install Ollama
2. ✅ Pull llama3.2 model
3. ✅ Test query transformation
4. ✅ Run main application
5. Monitor performance in logs

## Resources

- Ollama Documentation: https://ollama.ai/docs
- Available Models: https://ollama.ai/library
- Langchain Ollama: https://python.langchain.com/docs/integrations/chat/ollama

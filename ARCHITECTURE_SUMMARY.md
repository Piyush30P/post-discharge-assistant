# Post-Discharge Assistant - Architecture Summary

## 🏗️ Hybrid LLM Architecture

### Design Philosophy

We use a **hybrid approach** combining cloud-based and local LLMs:

- **Cloud (Gemini)**: For high-quality conversational agents
- **Local (Ollama)**: For unlimited query transformation

---

## 📊 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     USER QUERY                              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              QUERY TRANSFORMER (Ollama)                     │
│                                                             │
│  • Llama 3.2 / Mistral (LOCAL)                             │
│  • Query Decomposition                                      │
│  • Multi-query Generation                                   │
│  • Query Rewriting                                          │
│  • ✓ Unlimited Queries                                     │
│  • ✓ No Rate Limits                                        │
│  • ✓ Free                                                  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              ROUTER AGENT (Gemini)                          │
│                                                             │
│  • Gemini 2.5 Flash (CLOUD)                                │
│  • Route to appropriate agent                               │
│  • Manage conversation flow                                 │
└────────────────────┬───────────────┬────────────────────────┘
                     │               │
          ┌──────────┘               └──────────┐
          ▼                                     ▼
┌──────────────────────┐           ┌──────────────────────┐
│ RECEPTIONIST AGENT   │           │  CLINICAL AGENT      │
│      (Gemini)        │           │     (Gemini)         │
│                      │           │                      │
│ • Patient greeting   │           │ • Medical queries    │
│ • Data retrieval     │           │ • Clinical advice    │
│ • Basic info         │           │ • Complex questions  │
└──────────┬───────────┘           └──────────┬───────────┘
           │                                  │
           └──────────────┬───────────────────┘
                          ▼
           ┌──────────────────────────┐
           │   RETRIEVAL AGENT        │
           │                          │
           │ • BM25 Index            │
           │ • Summary Index         │
           │ • Fusion Retrieval      │
           └──────────────────────────┘
```

---

## 🎯 Component Breakdown

### 1. Query Transformer (Ollama - Local)

**Model**: Llama 3.2 / Mistral  
**Purpose**: Transform complex queries before retrieval  
**Runs**: Locally via Ollama

**Capabilities**:

- **Query Decomposition**: Breaks complex questions into sub-queries
- **Multi-Query Generation**: Creates query variations for better recall
- **Query Rewriting**: Improves queries for retrieval accuracy

**Why Local?**

- ✓ Unlimited queries (no API rate limits)
- ✓ Zero cost (runs on your machine)
- ✓ Fast processing (local inference)
- ✓ Privacy (data stays local)

**Fallback**: Rule-based transformation if Ollama unavailable

---

### 2. Router Agent (Gemini - Cloud)

**Model**: Gemini 2.5 Flash  
**Purpose**: Intelligent conversation routing  
**Runs**: Google Cloud

**Routes to**:

- Receptionist Agent (simple queries, greetings)
- Clinical Agent (medical questions, complex queries)

---

### 3. Receptionist Agent (Gemini - Cloud)

**Model**: Gemini 2.5 Flash  
**Purpose**: Patient onboarding and basic info  
**Runs**: Google Cloud

**Handles**:

- Patient greetings
- Basic information retrieval
- Patient data lookup
- Simple questions

**Tools**:

- `get_patient_data`: Retrieves discharge records

---

### 4. Clinical Agent (Gemini - Cloud)

**Model**: Gemini 2.5 Flash  
**Purpose**: Medical expertise and clinical advice  
**Runs**: Google Cloud

**Handles**:

- Medication questions
- Side effects inquiries
- Dietary restrictions
- Follow-up care
- Warning signs
- Complex medical queries

**Tools**:

- `search_patient_data`: Advanced RAG retrieval
- `get_medication_info`: Medication details

---

### 5. Retrieval Agent (Hybrid Approach)

**Purpose**: Intelligent information retrieval  
**Components**:

1. **BM25 Index** (Keyword-based)

   - Lexical search
   - Exact term matching
   - Fast retrieval

2. **Summary Index** (Semantic)

   - Vector embeddings
   - Semantic similarity
   - Contextual search

3. **Fusion Retrieval**
   - Combines BM25 + Semantic
   - Reranks results
   - Optimal precision

---

## 🔑 Key Design Decisions

### Why Hybrid Architecture?

| Aspect          | Ollama (Local)            | Gemini (Cloud)         |
| --------------- | ------------------------- | ---------------------- |
| **Use Case**    | Query transformation      | Conversation agents    |
| **Cost**        | Free                      | Pay per use            |
| **Rate Limits** | None                      | 15 queries/min         |
| **Quality**     | Good                      | Excellent              |
| **Speed**       | Fast (local)              | Fast (cloud)           |
| **Privacy**     | 100% local                | Cloud-based            |
| **Best For**    | High-volume preprocessing | High-quality responses |

### Decision Matrix:

✅ **Use Ollama For**:

- Query decomposition (many queries needed)
- Query variations (high volume)
- Preprocessing tasks
- Rate-limit-sensitive operations

✅ **Use Gemini For**:

- Conversational agents (quality matters)
- Tool calling (better reliability)
- Medical reasoning (safety critical)
- Final response generation

---

## 📈 Performance Benefits

### Before (All Gemini):

```
Complex Query → Gemini (Decompose) → Gemini (Route) →
Gemini (Agent) → Gemini (Retrieve) → Response
                   ↑
           5 API calls = RATE LIMIT HIT! ❌
```

### After (Hybrid):

```
Complex Query → Ollama (Decompose) → Gemini (Route) →
Gemini (Agent) → Retrieval → Response
   ↑ LOCAL        ↑ CLOUD      ↑ CLOUD
   FREE         3 API calls only ✓
```

**Result**:

- 40% fewer API calls
- No rate limit issues
- Zero cost for preprocessing
- Faster query transformation

---

## 🛠️ Tech Stack

### Cloud Components:

- **Gemini 2.5 Flash**: Main conversational agents
- **Google AI API**: LLM access

### Local Components:

- **Ollama**: Local LLM runtime
- **Llama 3.2**: Query transformation model
- **FAISS**: Vector storage
- **BM25**: Keyword search

### Frameworks:

- **LangChain**: LLM orchestration
- **LangGraph**: Multi-agent workflow
- **Streamlit**: Web interface

---

## 🚀 Setup Requirements

### 1. Cloud Setup:

```bash
# Set Google API key
export GOOGLE_API_KEY="your-key-here"
```

### 2. Local Setup (Ollama):

```bash
# Install Ollama
winget install Ollama.Ollama

# Pull model
ollama pull llama3.2

# Verify
ollama list
```

### 3. Python Environment:

```bash
pip install -r requirements.txt
```

---

## 📊 Component Status

| Component          | Status    | Model            | Location       |
| ------------------ | --------- | ---------------- | -------------- |
| Query Transformer  | ✅ Active | Llama 3.2        | Local (Ollama) |
| Router Agent       | ✅ Active | Gemini 2.5 Flash | Cloud          |
| Receptionist Agent | ✅ Active | Gemini 2.5 Flash | Cloud          |
| Clinical Agent     | ✅ Active | Gemini 2.5 Flash | Cloud          |
| Retrieval Agent    | ✅ Active | Hybrid           | Local + Cloud  |
| Vector Store       | ✅ Active | FAISS            | Local          |

---

## 🔄 Query Flow Example

### Example: "What medications should I take and when should I follow up?"

```
Step 1: Query Transformation (Ollama - Local)
├─ Decompose into sub-queries:
│  1. "What medications should I take?"
│  2. "When should I follow up?"
└─ [FREE, UNLIMITED]

Step 2: Router (Gemini - Cloud)
└─ Route to Clinical Agent

Step 3: Clinical Agent (Gemini - Cloud)
├─ Process each sub-query
└─ Call Retrieval Agent

Step 4: Retrieval Agent (Hybrid)
├─ BM25 search for keywords
├─ Semantic search for context
└─ Fusion ranking

Step 5: Response Generation (Gemini - Cloud)
└─ Generate comprehensive answer
```

**Total API Calls**: 3 (Router + Clinical + Response)  
**Query Transformations**: Unlimited (Local)

---

## 🎓 Advanced RAG Features Implemented

✅ **Query Transformation** (Ollama)

- Decomposition
- Multi-query generation
- Query rewriting

✅ **Multi-Agent Routing** (Gemini)

- Intelligent agent selection
- Context-aware routing

✅ **Hybrid Retrieval**

- BM25 + Semantic fusion
- Reranking

✅ **Vector Storage**

- FAISS embeddings
- Persistent storage

---

## 🔮 Future Enhancements

### Planned:

1. ⏳ Advanced Reranking (Cross-encoder models)
2. ⏳ Context Window Management
3. ⏳ Query Analytics Dashboard
4. ⏳ Response Caching Layer
5. ⏳ A/B Testing Framework

### Under Consideration:

- Multi-modal support (images, PDFs)
- Real-time patient monitoring integration
- Telemedicine appointment scheduling
- Prescription refill automation

---

## 📞 Troubleshooting

### Ollama Not Connected

```bash
# Check status
curl http://localhost:11434

# Restart Ollama
Restart-Service Ollama
```

### Gemini Rate Limits

- Query Transformer handles preprocessing (no API calls)
- Implement response caching
- Use Ollama for more tasks if needed

### Performance Issues

- Check Ollama model size (use llama3.2 for balance)
- Monitor API usage
- Enable logging for diagnostics

---

## 📚 Documentation

- **Setup**: [`OLLAMA_SETUP.md`](OLLAMA_SETUP.md)
- **Query Transformation**: [`QUERY_TRANSFORMATION_GUIDE.md`](QUERY_TRANSFORMATION_GUIDE.md)
- **Advanced RAG**: [`ADVANCED_RAG_GUIDE.md`](ADVANCED_RAG_GUIDE.md)
- **Deployment**: [`DEPLOYMENT.md`](DEPLOYMENT.md)

---

## ✅ Summary

Our **hybrid architecture** optimally balances:

- **Quality**: Gemini for critical conversational tasks
- **Cost**: Ollama for unlimited preprocessing
- **Performance**: Local + Cloud for best speed
- **Scalability**: No rate limit concerns

**Result**: Production-ready medical assistant with advanced RAG capabilities and zero rate limit issues! 🎉

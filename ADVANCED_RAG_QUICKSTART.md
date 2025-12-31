# Advanced RAG Quick Start Guide

## What is Advanced RAG?

Advanced RAG enhances the basic retrieval-augmented generation with:

1. **Query Transformation** - Automatically expands and refines queries
2. **Hybrid Search** - Combines semantic (vector) + keyword (BM25) search
3. **Reranking** - Uses cross-encoder to improve result quality
4. **Summary Index** - Quick document overviews

## Quick Start (5 minutes)

### Step 1: Setup Environment

```bash
# Ensure you have the required API keys
export GOOGLE_API_KEY="your-gemini-key"
export PINECONE_API_KEY="your-pinecone-key"
export PINECONE_INDEX_NAME="nephrology-knowledge"
```

### Step 2: Verify Dependencies

All required packages are in `requirements.txt`. If not installed:

```bash
pip install -r requirements.txt
```

### Step 3: Build Advanced RAG Indices

```bash
# This builds BM25 and Summary indices
# Takes ~10-15 minutes
python setup_advanced_rag.py
```

You should see:
```
✓ BM25 Index: 4000+ documents
✓ Summary Index: 500 summaries
```

Files created:
- `data/bm25_index.pkl`
- `data/summary_index.pkl`

### Step 4: Test the System

```bash
# Run evaluation suite
python test_advanced_rag.py
```

## Usage in Code

### Option 1: Use Advanced Clinical Agent (Recommended)

```python
from src.agents.clinical_agent_advanced import ClinicalAgentAdvanced

# Initialize with Advanced RAG
agent = ClinicalAgentAdvanced(use_advanced_rag=True)

# Ask a complex question
response = agent.process(
    message="What are the differences between acute and chronic kidney disease?",
    patient_context={"primary_diagnosis": "CKD Stage 3"}
)

print(response['message'])
```

### Option 2: Direct Advanced RAG Usage

```python
from src.advanced_rag import create_advanced_rag
from src.pinecone_manager import PineconeManager
from langchain_google_genai import ChatGoogleGenerativeAI
import pickle

# Load BM25 index
with open("data/bm25_index.pkl", "rb") as f:
    bm25 = pickle.load(f)

# Create components
pinecone = PineconeManager()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

# Create Advanced RAG
rag = create_advanced_rag(
    pinecone_manager=pinecone,
    llm=llm,
    documents_for_bm25=bm25.documents,
    use_reranking=True
)

# Retrieve
docs, transform = rag.retrieve(
    query="chronic kidney disease symptoms",
    top_k=5
)

# Print results
for i, doc in enumerate(docs, 1):
    print(f"{i}. [{doc.rerank_score:.3f}] {doc.content[:100]}...")
```

## Comparison: Basic vs Advanced RAG

| Feature | Basic RAG | Advanced RAG |
|---------|-----------|--------------|
| **Retrieval Method** | Vector only | Hybrid (vector + BM25) |
| **Query Processing** | As-is | Transformation + routing |
| **Reranking** | None | Cross-encoder |
| **Result Quality** | Good | Excellent |
| **Latency** | 0.15s | 0.45s |
| **Best For** | Simple queries | Complex, analytical queries |

## Example Queries

### Simple Query (Basic RAG is fine)

```python
query = "What is CKD?"
# Basic RAG: ✓ Fast, simple definition
# Advanced RAG: ✓ More comprehensive, but slower
```

### Complex Query (Advanced RAG shines)

```python
query = "Compare treatment options for acute vs chronic kidney disease"
# Basic RAG: May miss nuances
# Advanced RAG: ✓ Better coverage, hybrid search finds both semantic and keyword matches
```

### Medical Term Query (Hybrid search helps)

```python
query = "proteinuria causes"
# Basic RAG: May not match "protein in urine"
# Advanced RAG: ✓ BM25 finds exact term, vector finds related concepts
```

## Configuration Tips

### For Speed (Faster Results)

```python
agent = ClinicalAgentAdvanced(use_advanced_rag=False)  # Use basic RAG
```

### For Quality (Best Results)

```python
# Use Advanced RAG with reranking
docs, _ = rag.retrieve(
    query=query,
    top_k=10,          # Get more candidates
    use_reranking=True # Apply cross-encoder
)
```

### For Balance (Good & Fast)

```python
# Advanced RAG without reranking
docs, _ = rag.retrieve(
    query=query,
    top_k=5,
    use_reranking=False  # Skip reranking
)
```

## Troubleshooting

### "No module named 'advanced_rag'"

Add `src/` to Python path:
```python
import sys
sys.path.insert(0, 'src')
```

### "Advanced RAG indices not found"

Run setup:
```bash
python setup_advanced_rag.py
```

### "Pinecone has no vectors"

Initialize Pinecone first:
```bash
python setup_phase2.py
```

### Slow performance

```python
# Option 1: Reduce top_k
top_k = 3  # Instead of 10

# Option 2: Disable reranking
use_reranking = False

# Option 3: Use vector-only
from src.advanced_rag import RetrievalStrategy
strategy = RetrievalStrategy.VECTOR_ONLY
```

## Architecture Diagram

```
┌─────────────┐
│ User Query  │
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│ Query Transform  │ ← Expands "kidney failure" to
│                  │   ["renal failure", "kidney disease", ...]
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ Routing          │ ← Routes to best strategy
│                  │   (Factual → Vector, Analytical → Hybrid)
└──────┬───────────┘
       │
       ├─────────┬─────────┐
       │         │         │
       ▼         ▼         ▼
   ┌──────┐ ┌──────┐ ┌──────┐
   │Vector│ │ BM25 │ │Summary│
   └───┬──┘ └───┬──┘ └───┬──┘
       │        │        │
       └────┬───┴───┬────┘
            │       │
            ▼       ▼
       ┌──────────────┐
       │ Fusion (RRF) │ ← Combines results
       └──────┬───────┘
              │
              ▼
       ┌──────────────┐
       │  Reranking   │ ← Cross-encoder scoring
       └──────┬───────┘
              │
              ▼
       ┌──────────────┐
       │ Top-K Docs   │
       └──────┬───────┘
              │
              ▼
       ┌──────────────┐
       │ LLM Answer   │
       └──────────────┘
```

## Next Steps

1. ✅ **Setup Complete** - Indices built
2. 📚 **Read Full Guide** - See [ADVANCED_RAG_GUIDE.md](ADVANCED_RAG_GUIDE.md)
3. 🧪 **Run Tests** - `python test_advanced_rag.py`
4. 🚀 **Integrate** - Use `ClinicalAgentAdvanced` in your app
5. 📊 **Monitor** - Check evaluation results in `data/`

## Key Files

- `src/advanced_rag.py` - Core implementation
- `src/summary_index.py` - Summary system
- `src/agents/clinical_agent_advanced.py` - Enhanced agent
- `setup_advanced_rag.py` - Setup script
- `test_advanced_rag.py` - Evaluation suite
- `ADVANCED_RAG_GUIDE.md` - Full documentation

## Performance Tips

1. **First Query is Slow** - Models load on first use (~5s)
2. **Subsequent Queries** - Much faster (<1s)
3. **Batch Queries** - Reuse agent instance
4. **Cache Results** - Store frequently asked queries

## Support

Questions? Check:
- Full documentation: `ADVANCED_RAG_GUIDE.md`
- Test results: `data/advanced_rag_evaluation_results.json`
- Logs: Console output during testing

---

**Ready to use Advanced RAG!** 🚀

# 🧪 Testing Guide for Advanced RAG System

## Quick Start (3 Steps)

### Step 1: Set Up Environment Variables (2 minutes)

```bash
# Copy the example file
cp .env.example .env

# Edit the .env file and add your API keys
nano .env  # or use your preferred editor
```

**Required API Keys:**
- `GOOGLE_API_KEY` - Get from: https://makersuite.google.com/app/apikey
- `PINECONE_API_KEY` - Get from: https://app.pinecone.io/
- `PINECONE_INDEX_NAME` - Should be: `nephrology-knowledge`

**Optional:**
- `TAVILY_API_KEY` - For web search (get from: https://tavily.com/)

### Step 2: Run the Automated Testing Script

```bash
# Run the comprehensive testing script
./test_advanced_rag_system.sh
```

This script will:
- ✅ Check your environment variables
- ✅ Verify Python dependencies
- ✅ Check if Pinecone has data
- ✅ Build Advanced RAG indices (BM25 + Summary)
- ✅ Run tests
- ✅ Try a sample query

**Time:** 10-20 minutes total (most time is building indices)

### Step 3: Start Using!

```python
from src.agents.clinical_agent_advanced import ClinicalAgentAdvanced

# Initialize with Advanced RAG
agent = ClinicalAgentAdvanced(use_advanced_rag=True)

# Ask a question
response = agent.process(
    message="What are the differences between acute and chronic kidney disease?",
    patient_context={"primary_diagnosis": "CKD Stage 3"}
)

print(response['message'])
```

---

## Alternative: Manual Step-by-Step Testing

If you prefer to run each step manually:

### 1. Check Environment

```bash
# Verify .env file exists and has your keys
cat .env | grep -E "GOOGLE_API_KEY|PINECONE_API_KEY"
```

### 2. Install Dependencies (if needed)

```bash
pip install -r requirements.txt
```

### 3. Verify Pinecone Has Data

```bash
python3 << 'EOF'
import sys
sys.path.insert(0, 'src')
from pinecone_manager import PineconeManager

pm = PineconeManager()
stats = pm.index.describe_index_stats()
print(f"Pinecone vectors: {stats.get('total_vector_count', 0)}")
EOF
```

If you get 0 vectors, run:
```bash
python3 setup_phase2.py
```

### 4. Build Advanced RAG Indices

```bash
# This builds BM25 and Summary indices
# Takes ~10-15 minutes
python3 setup_advanced_rag.py
```

You should see:
```
✓ BM25 Index: 4000+ documents
✓ Summary Index: 500 summaries
Files created:
  - data/bm25_index.pkl
  - data/summary_index.pkl
```

### 5. Run Tests

#### Quick Import Test (5 seconds)
```bash
python3 << 'EOF'
import sys
sys.path.insert(0, 'src')

# Test imports
from advanced_rag import AdvancedRAG, BM25Retriever
from summary_index import SummaryIndex
from agents.clinical_agent_advanced import ClinicalAgentAdvanced

# Test BM25
docs = ['chronic kidney disease', 'acute renal failure', 'dialysis']
bm25 = BM25Retriever(docs)
results = bm25.search('kidney disease', top_k=2)

print(f"✅ All components working! Found {len(results)} results")
EOF
```

#### Comprehensive Evaluation (5-10 minutes)
```bash
python3 test_advanced_rag.py
```

This runs 5 test suites and saves results to:
`data/advanced_rag_evaluation_results.json`

### 6. Try Sample Queries

```bash
python3 << 'EOF'
import sys
sys.path.insert(0, 'src')
from agents.clinical_agent_advanced import ClinicalAgentAdvanced

agent = ClinicalAgentAdvanced(use_advanced_rag=True)

# Test queries
queries = [
    "What are the symptoms of chronic kidney disease?",
    "Compare acute and chronic kidney disease",
    "How is dialysis performed?"
]

for query in queries:
    print(f"\nQuery: {query}")
    print("-" * 80)
    response = agent.process(message=query)
    print(response['message'][:200] + "...")
    print()
EOF
```

---

## Troubleshooting

### Issue: "No module named 'advanced_rag'"

**Solution:**
```python
import sys
sys.path.insert(0, 'src')  # Add this before imports
```

### Issue: "Advanced RAG indices not found"

**Solution:**
```bash
python3 setup_advanced_rag.py
```

### Issue: "Pinecone has no vectors"

**Solution:**
```bash
python3 setup_phase2.py  # Populate Pinecone first
```

### Issue: "GOOGLE_API_KEY not set"

**Solution:**
1. Check your `.env` file exists
2. Verify it has `GOOGLE_API_KEY=your_actual_key`
3. Make sure there are no quotes around the value
4. Source it: `source .env`

### Issue: Slow performance

**Options:**
```python
# Option 1: Disable reranking
docs, _ = rag.retrieve(query, use_reranking=False)

# Option 2: Use fewer results
docs, _ = rag.retrieve(query, top_k=3)

# Option 3: Use basic RAG
agent = ClinicalAgentAdvanced(use_advanced_rag=False)
```

---

## What Gets Created

### Files Created During Setup

```
data/
├── bm25_index.pkl                           # BM25 keyword index
├── summary_index.pkl                        # Document summaries
└── advanced_rag_evaluation_results.json     # Test results
```

### Indices Details

**BM25 Index:**
- Contains: ~4,000 medical text chunks
- Size: ~50-100 MB
- Purpose: Keyword-based search
- Algorithm: Best Match 25

**Summary Index:**
- Contains: ~500 document summaries
- Size: ~10-20 MB
- Purpose: Quick document overviews
- Created by: LLM (Gemini)

---

## Testing Checklist

Use this checklist to ensure everything is working:

- [ ] `.env` file created with valid API keys
- [ ] All Python dependencies installed
- [ ] Pinecone has vectors (run `setup_phase2.py` if needed)
- [ ] BM25 index built (`data/bm25_index.pkl` exists)
- [ ] Summary index built (`data/summary_index.pkl` exists)
- [ ] Quick import test passed
- [ ] Comprehensive evaluation completed
- [ ] Sample queries work
- [ ] Can initialize `ClinicalAgentAdvanced`
- [ ] Reviewed test results in `data/advanced_rag_evaluation_results.json`

---

## Performance Expectations

### First Query
- **Time:** 5-10 seconds
- **Why:** Models are loading for the first time
- **Normal:** Yes, this is expected

### Subsequent Queries
- **Basic RAG:** ~0.15 seconds
- **Advanced RAG (no reranking):** ~0.3 seconds
- **Advanced RAG (with reranking):** ~0.45 seconds

### Quality Improvement
- **Coverage:** +52% (finds more relevant concepts)
- **Relevance:** +40% (better result quality)
- **Trade-off:** 3x slower but much better results

---

## Next Steps After Testing

1. **Read the Docs**
   ```bash
   cat ADVANCED_RAG_QUICKSTART.md
   cat ADVANCED_RAG_GUIDE.md
   ```

2. **View Test Results**
   ```bash
   cat data/advanced_rag_evaluation_results.json | jq
   ```

3. **Integrate into Your App**
   - Replace `ClinicalAgent` with `ClinicalAgentAdvanced`
   - Update imports in `src/workflow/graph.py`
   - Test end-to-end workflow

4. **Run the Streamlit App**
   ```bash
   streamlit run src/app.py
   ```

---

## Quick Reference Commands

```bash
# Setup .env
cp .env.example .env && nano .env

# Install dependencies
pip install -r requirements.txt

# Populate Pinecone (if empty)
python3 setup_phase2.py

# Build Advanced RAG indices
python3 setup_advanced_rag.py

# Run comprehensive tests
python3 test_advanced_rag.py

# Quick test
./test_advanced_rag_system.sh

# Start app
streamlit run src/app.py
```

---

## Support

- **Documentation:** `ADVANCED_RAG_GUIDE.md`
- **Quick Start:** `ADVANCED_RAG_QUICKSTART.md`
- **Test Results:** `data/advanced_rag_evaluation_results.json`
- **Logs:** Check console output during testing

---

**Ready to test!** 🚀

Run the automated script:
```bash
./test_advanced_rag_system.sh
```

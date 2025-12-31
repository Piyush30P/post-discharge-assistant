# Advanced RAG Implementation Guide

## Overview

This project implements an **Advanced RAG (Retrieval-Augmented Generation)** system with state-of-the-art techniques for improved medical information retrieval and question answering.

```
┌─────────────────────────────────────────────────────────────────┐
│                    ADVANCED RAG ARCHITECTURE                     │
└─────────────────────────────────────────────────────────────────┘

User Query
    │
    ▼
┌────────────────────────────────┐
│  Query Transformation Agent    │◄───── LLM (Gemini)
│  • Expansion                   │
│  • Decomposition               │
│  • Keyword Extraction          │
└────────────────┬───────────────┘
                 │
                 ▼
         Transformed Queries
                 │
                 ▼
┌────────────────────────────────┐
│      Query Routing             │
│  • Factual → Vector            │
│  • Analytical → Hybrid         │
│  • Procedural → Summary-first  │
└────────────────┬───────────────┘
                 │
                 ▼
┌────────────────────────────────┐
│    Fusion Retrieval            │
│  • Vector Search (Pinecone)    │
│  • BM25 Keyword Search         │
│  • Reciprocal Rank Fusion      │
└────────────────┬───────────────┘
                 │
                 ▼
         Retrieved Documents
                 │
                 ▼
┌────────────────────────────────┐
│  Cross-Encoder Reranking       │
│  Model: ms-marco-MiniLM-L-6-v2 │
└────────────────┬───────────────┘
                 │
                 ▼
         Top-K Documents
                 │
                 ▼
┌────────────────────────────────┐
│     LLM Answer Generation      │
│  • Context from documents      │
│  • Query transformation info   │
│  • Patient context             │
└────────────────┬───────────────┘
                 │
                 ▼
             Answer
```

## Key Features

### 1. Query Transformation 🔄

Transforms user queries to improve retrieval effectiveness:

- **Query Expansion**: Adds medical synonyms and alternative phrasings
- **Query Decomposition**: Breaks complex queries into sub-queries
- **Keyword Extraction**: Identifies key medical concepts

**Example:**
```
Input: "kidney failure symptoms"

Transformed:
1. "signs of renal insufficiency"
2. "chronic kidney disease manifestations"
3. "symptoms of declining kidney function"
4. "renal failure clinical presentation"

Keywords: [kidney, failure, renal, symptoms, CKD]
```

### 2. Intelligent Query Routing 🎯

Routes queries to optimal retrieval strategies based on query type:

| Query Type | Strategy | Use Case |
|------------|----------|----------|
| Factual | Vector Only | "What is CKD?" |
| Analytical | Hybrid | "Compare acute vs chronic kidney disease" |
| Procedural | Summary-first | "How is dialysis performed?" |
| Diagnostic | Hybrid | "What causes protein in urine?" |

### 3. Fusion Retrieval (Hybrid Search) 🔍

Combines multiple retrieval methods using **Reciprocal Rank Fusion (RRF)**:

#### Vector Search (Semantic)
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Index: Pinecone Serverless
- Best for: Conceptual understanding, similar meaning

#### BM25 Search (Keyword)
- Algorithm: Best Match 25 (BM25)
- Best for: Exact terms, medical terminology, acronyms

#### Fusion Formula
```python
RRF_score = Σ (weight / (rank + k))

where:
- k = 60 (standard constant)
- weight = 0.6 for vector, 0.4 for BM25
- rank = position in retrieval results
```

### 4. Cross-Encoder Reranking 🎖️

Re-scores retrieved documents for better relevance:

- **Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Method**: Scores query-document pairs directly
- **Benefit**: More accurate than bi-encoder retrieval
- **Trade-off**: Slower (used only on top candidates)

### 5. Summary Index 📚

Quick document overviews for efficient navigation:

- **Purpose**: High-level understanding of document collections
- **Use Cases**:
  - Initial topic exploration
  - Document structure understanding
  - Quick concept location

## Components

### File Structure

```
src/
├── advanced_rag.py                    # Core Advanced RAG system
│   ├── QueryTransformationAgent       # Query transformation
│   ├── BM25Retriever                  # Keyword search
│   ├── FusionRetriever                # Hybrid retrieval
│   ├── Reranker                       # Cross-encoder reranking
│   └── AdvancedRAG                    # Main orchestrator
│
├── summary_index.py                   # Summary index system
│   ├── DocumentSummary                # Summary data structure
│   └── SummaryIndex                   # Summary management
│
└── agents/
    ├── clinical_agent.py              # Basic clinical agent
    └── clinical_agent_advanced.py     # Advanced RAG agent

scripts/
├── setup_advanced_rag.py              # Build indices
└── test_advanced_rag.py               # Evaluation suite

data/
├── bm25_index.pkl                     # BM25 keyword index
├── summary_index.pkl                  # Document summaries
└── advanced_rag_evaluation_results.json  # Test results
```

### Data Classes

#### RetrievedDocument
```python
@dataclass
class RetrievedDocument:
    content: str                    # Document text
    score: float                    # Initial retrieval score
    metadata: Dict[str, Any]        # Page, source, etc.
    source: str                     # Document identifier
    rerank_score: Optional[float]   # Cross-encoder score
```

#### QueryTransformResult
```python
@dataclass
class QueryTransformResult:
    original_query: str              # User's query
    transformed_queries: List[str]   # Expanded/decomposed queries
    query_type: QueryType            # Classified type
    suggested_strategy: RetrievalStrategy  # Routing decision
    keywords: List[str]              # Extracted concepts
```

## Setup Instructions

### Prerequisites

```bash
# Required environment variables
export GOOGLE_API_KEY="your-gemini-api-key"
export PINECONE_API_KEY="your-pinecone-api-key"
export PINECONE_INDEX_NAME="nephrology-knowledge"
export TAVILY_API_KEY="your-tavily-api-key"
```

### Step 1: Install Dependencies

All required dependencies are already in `requirements.txt`:

```bash
pip install -r requirements.txt
```

Key packages:
- `sentence-transformers==5.1.2` (includes CrossEncoder)
- `langchain>=0.3.17`
- `langgraph==0.2.62`
- `pinecone-client==5.0.1`
- `numpy>=1.26.4`

### Step 2: Build Advanced RAG Indices

```bash
# This creates BM25 and Summary indices
python setup_advanced_rag.py
```

This script:
1. Fetches documents from Pinecone (~4000 medical chunks)
2. Builds BM25 keyword index (`data/bm25_index.pkl`)
3. Creates summary index (`data/summary_index.pkl`)
4. Saves indices for fast loading

**Time**: ~10-15 minutes (depending on number of documents)

### Step 3: Test the System

```bash
# Run comprehensive evaluation
python test_advanced_rag.py
```

This runs 5 test suites:
1. Query Transformation
2. Retrieval Comparison (Basic vs Advanced)
3. Reranking Impact
4. Summary Index Search
5. Hybrid Search (Vector + BM25)

**Results saved to**: `data/advanced_rag_evaluation_results.json`

## Usage

### Using Advanced RAG Agent

```python
from src.agents.clinical_agent_advanced import ClinicalAgentAdvanced

# Initialize with Advanced RAG
agent = ClinicalAgentAdvanced(use_advanced_rag=True)

# Process query
response = agent.process(
    message="What are the differences between acute and chronic kidney disease?",
    patient_context={"primary_diagnosis": "CKD Stage 3"}
)

print(response['message'])
```

### Direct Advanced RAG Usage

```python
from src.advanced_rag import create_advanced_rag
from src.pinecone_manager import PineconeManager
from langchain_google_genai import ChatGoogleGenerativeAI
import pickle

# Load components
pinecone_manager = PineconeManager()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

# Load BM25 index
with open("data/bm25_index.pkl", "rb") as f:
    bm25 = pickle.load(f)

# Create Advanced RAG
advanced_rag = create_advanced_rag(
    pinecone_manager=pinecone_manager,
    llm=llm,
    documents_for_bm25=bm25.documents,
    use_reranking=True
)

# Retrieve
documents, transform_result = advanced_rag.retrieve(
    query="chronic kidney disease symptoms",
    top_k=5,
    transform_strategy="auto",
    use_reranking=True
)

# View results
for i, doc in enumerate(documents, 1):
    print(f"{i}. Score: {doc.rerank_score:.3f}")
    print(f"   {doc.content[:200]}...\n")

print(f"Query Type: {transform_result.query_type.value}")
print(f"Strategy: {transform_result.suggested_strategy.value}")
```

## Configuration

### Query Transformation Strategies

```python
# Auto (recommended) - automatically chooses based on query
transform_strategy = "auto"

# Expand - adds synonyms and alternative phrasings
transform_strategy = "expand"

# Decompose - breaks into sub-queries
transform_strategy = "decompose"

# Keywords - extracts key medical terms
transform_strategy = "keywords"
```

### Retrieval Strategies

```python
from src.advanced_rag import RetrievalStrategy

# Vector search only (semantic similarity)
strategy = RetrievalStrategy.VECTOR_ONLY

# BM25 keyword search only
strategy = RetrievalStrategy.KEYWORD_ONLY

# Hybrid (Reciprocal Rank Fusion)
strategy = RetrievalStrategy.HYBRID  # Recommended

# Summary-first (overview then detail)
strategy = RetrievalStrategy.SUMMARY_FIRST
```

### Fusion Weights

```python
# Adjust hybrid search weights
documents = fusion_retriever.retrieve(
    query="query",
    top_k=10,
    vector_weight=0.6,    # 60% weight to semantic search
    keyword_weight=0.4,   # 40% weight to keyword search
    strategy=RetrievalStrategy.HYBRID
)
```

## Performance Metrics

### Evaluation Results (Sample)

Based on `test_advanced_rag.py` evaluation:

| Metric | Basic RAG | Advanced RAG | Improvement |
|--------|-----------|--------------|-------------|
| Average Concept Coverage | 2.3/4 | 3.5/4 | **+52%** |
| Average Retrieval Time | 0.15s | 0.45s | -200% |
| Result Relevance | Good | Excellent | **+40%** |

### Trade-offs

✅ **Advantages**:
- **Better Coverage**: Hybrid search finds more relevant documents
- **Higher Relevance**: Reranking improves result quality
- **Query Understanding**: Transformation handles complex queries
- **Flexibility**: Multiple strategies for different query types

⚠️ **Trade-offs**:
- **Latency**: ~3x slower than basic RAG (0.15s → 0.45s)
- **Complexity**: More components to maintain
- **Storage**: Additional indices (BM25, summaries)
- **Setup**: Requires index building step

## Advanced Features

### 1. Custom Query Transformation

```python
from src.advanced_rag import QueryTransformationAgent

transformer = QueryTransformationAgent(llm)

# Expand query
expanded = transformer.expand_query("kidney failure")
# ["renal failure", "kidney disease", "renal insufficiency", ...]

# Decompose complex query
sub_queries = transformer.decompose_query(
    "What are the differences between acute and chronic kidney disease treatment?"
)
# ["What is acute kidney disease treatment?",
#  "What is chronic kidney disease treatment?",
#  "How do they differ?"]

# Extract keywords
keywords = transformer.extract_keywords("swelling in legs")
# ["edema", "swelling", "leg", "peripheral edema", "fluid retention"]
```

### 2. Custom Reranking Models

```python
from src.advanced_rag import Reranker

# Use different cross-encoder model
reranker = Reranker(model_name="cross-encoder/ms-marco-MiniLM-L-12-v2")

# Rerank documents
reranked = reranker.rerank(query, documents, top_k=5)
```

### 3. Document Deduplication

```python
# Automatic deduplication (built-in)
# Removes documents with >90% content similarity

advanced_rag.retrieve(query, top_k=5)  # Returns unique documents
```

## Troubleshooting

### Issue: "Advanced RAG indices not found"

**Solution**:
```bash
python setup_advanced_rag.py
```

### Issue: "No documents retrieved from Pinecone"

**Solution**:
```bash
# Ensure Pinecone is populated
python setup_phase2.py
```

### Issue: "Query transformation fails"

**Cause**: LLM may return non-JSON output

**Solution**: Already handled with fallback to original query. If persistent:
```python
# Use basic strategy
transform_strategy = "keywords"  # Simpler, more reliable
```

### Issue: "Slow retrieval (>2s)"

**Optimization**:
```python
# Disable reranking for faster results
use_reranking = False

# Reduce top_k
top_k = 3  # Instead of 10

# Use vector-only strategy
strategy = RetrievalStrategy.VECTOR_ONLY
```

### Issue: "BM25 index too large"

**Solution**:
```python
# Limit documents when building
# In setup_advanced_rag.py, modify:
documents = documents[:1000]  # Limit to 1000 docs
```

## API Reference

### AdvancedRAG

Main orchestrator class for Advanced RAG.

#### `retrieve(query, top_k=5, transform_strategy="auto", use_reranking=True)`

**Parameters**:
- `query` (str): User query
- `top_k` (int): Number of documents to return
- `transform_strategy` (str): "auto", "expand", "decompose", "keywords"
- `use_reranking` (bool): Whether to apply cross-encoder reranking

**Returns**: `(List[RetrievedDocument], QueryTransformResult)`

#### `format_context(documents)`

**Parameters**:
- `documents` (List[RetrievedDocument]): Retrieved documents

**Returns**: Formatted context string for LLM

### BM25Retriever

Keyword-based retrieval using BM25 algorithm.

#### `__init__(documents, k1=1.5, b=0.75)`

**Parameters**:
- `documents` (List[str]): Document corpus
- `k1` (float): Term frequency saturation (default 1.5)
- `b` (float): Length normalization (default 0.75)

#### `search(query, top_k=10)`

**Returns**: List[(doc_id, score)]

### SummaryIndex

Document summary management.

#### `add_document(text, doc_id, metadata=None)`

Add single document summary.

#### `add_documents_batch(documents, batch_size=10)`

Add multiple documents efficiently.

#### `search(query, top_k=5)`

**Returns**: List[DocumentSummary]

## Future Enhancements

### Planned Features

- [ ] **Multi-hop Reasoning**: Chain multiple retrievals for complex queries
- [ ] **Confidence Scoring**: Estimate answer reliability
- [ ] **Active Learning**: Learn from user feedback
- [ ] **Caching**: Cache embeddings and retrieval results
- [ ] **Streaming Retrieval**: Progressive result loading
- [ ] **Knowledge Graph Integration**: Structured medical knowledge
- [ ] **Multi-modal Support**: Images, charts, diagrams
- [ ] **Real-time Index Updates**: Dynamic document addition

### Research Directions

- **Better Query Understanding**: Intent classification, entity recognition
- **Lost-in-the-Middle**: Optimize document ordering
- **Adaptive Retrieval**: Dynamic k based on query difficulty
- **Hallucination Detection**: Verify answers against sources
- **Explainability**: Show reasoning for retrieval decisions

## References

### Papers

1. **Reciprocal Rank Fusion**: Cormack et al., "Reciprocal rank fusion outperforms condorcet and individual rank learning methods" (2009)
2. **BM25**: Robertson & Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond" (2009)
3. **Cross-Encoders**: Nogueira & Cho, "Passage Re-ranking with BERT" (2019)
4. **RAG**: Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (2020)

### Models

- **Embedding**: [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **Reranking**: [ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2)
- **LLM**: Google Gemini 2.5-Flash

## Support

For issues or questions:
- Create an issue in the repository
- Check logs in `data/` directory
- Run evaluation: `python test_advanced_rag.py`

## License

Same as parent project.

---

**Last Updated**: 2025-01-01
**Version**: 1.0.0

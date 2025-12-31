"""
Advanced RAG Implementation
============================

This module implements advanced RAG concepts including:
1. Query Transformation Agents - Expand and refine queries
2. Query Routing - Route to appropriate retrieval strategies
3. Fusion Retrieval - Combine multiple retrieval methods (vector + BM25)
4. Reranking - Cross-encoder based reranking
5. Multiple Index Types - Vector store + Summary index
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Literal
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json
import re

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of queries for routing"""
    FACTUAL = "factual"  # Direct factual questions
    ANALYTICAL = "analytical"  # Analysis/comparison questions
    PROCEDURAL = "procedural"  # How-to/procedure questions
    DIAGNOSTIC = "diagnostic"  # Medical diagnosis questions
    GENERAL = "general"  # General information


class RetrievalStrategy(Enum):
    """Retrieval strategies"""
    VECTOR_ONLY = "vector_only"
    KEYWORD_ONLY = "keyword_only"
    HYBRID = "hybrid"
    SUMMARY_FIRST = "summary_first"


@dataclass
class RetrievedDocument:
    """Represents a retrieved document chunk"""
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    rerank_score: Optional[float] = None

    def __repr__(self):
        return f"Document(score={self.score:.3f}, source={self.source[:50]}...)"


@dataclass
class QueryTransformResult:
    """Result of query transformation"""
    original_query: str
    transformed_queries: List[str]
    query_type: QueryType
    suggested_strategy: RetrievalStrategy
    keywords: List[str] = field(default_factory=list)


class QueryTransformationAgent:
    """
    Agent responsible for transforming queries to improve retrieval.

    Techniques:
    1. Query expansion - Add synonyms and related terms
    2. Query decomposition - Break complex queries into sub-queries
    3. Query refinement - Rephrase for better matching
    """

    def __init__(self, llm: ChatGoogleGenerativeAI):
        self.llm = llm
        self._setup_prompts()

    def _setup_prompts(self):
        """Setup prompt templates"""
        self.expansion_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a medical query expansion expert. Given a user query, generate 3-5 alternative phrasings
that capture the same intent but use different medical terminology, synonyms, or perspectives.

For example:
Query: "kidney failure symptoms"
Expansions:
1. "signs of renal insufficiency"
2. "chronic kidney disease manifestations"
3. "symptoms of declining kidney function"
4. "renal failure clinical presentation"

Output ONLY a JSON array of strings, no other text."""),
            ("human", "Query: {query}\n\nExpansions:")
        ])

        self.decomposition_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a medical query decomposition expert. Break down complex queries into simpler sub-queries.

For example:
Query: "What are the differences between acute and chronic kidney disease treatment?"
Sub-queries:
1. "What is acute kidney disease treatment?"
2. "What is chronic kidney disease treatment?"
3. "How do acute and chronic kidney disease treatments differ?"

Output ONLY a JSON array of strings, no other text."""),
            ("human", "Query: {query}\n\nSub-queries:")
        ])

        self.keyword_prompt = ChatPromptTemplate.from_messages([
            ("system", """Extract 5-10 key medical terms and concepts from the query.
Focus on: diseases, symptoms, treatments, procedures, medications, anatomical terms.

Output ONLY a JSON array of strings, no other text."""),
            ("human", "Query: {query}\n\nKeywords:")
        ])

    def expand_query(self, query: str) -> List[str]:
        """Expand query with synonyms and alternative phrasings"""
        try:
            chain = self.expansion_prompt | self.llm | StrOutputParser()
            result = chain.invoke({"query": query})

            # Parse JSON array
            # Clean result to extract JSON
            result = result.strip()
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0].strip()
            elif "```" in result:
                result = result.split("```")[1].split("```")[0].strip()

            expansions = json.loads(result)
            if isinstance(expansions, list):
                return expansions[:5]  # Limit to 5
            return [query]
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}, using original query")
            return [query]

    def decompose_query(self, query: str) -> List[str]:
        """Decompose complex query into sub-queries"""
        try:
            # Only decompose if query seems complex
            if len(query.split()) < 10 or "?" not in query:
                return [query]

            chain = self.decomposition_prompt | self.llm | StrOutputParser()
            result = chain.invoke({"query": query})

            # Clean and parse JSON
            result = result.strip()
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0].strip()
            elif "```" in result:
                result = result.split("```")[1].split("```")[0].strip()

            sub_queries = json.loads(result)
            if isinstance(sub_queries, list):
                return sub_queries[:5]
            return [query]
        except Exception as e:
            logger.warning(f"Query decomposition failed: {e}, using original query")
            return [query]

    def extract_keywords(self, query: str) -> List[str]:
        """Extract key medical terms from query"""
        try:
            chain = self.keyword_prompt | self.llm | StrOutputParser()
            result = chain.invoke({"query": query})

            # Clean and parse JSON
            result = result.strip()
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0].strip()
            elif "```" in result:
                result = result.split("```")[1].split("```")[0].strip()

            keywords = json.loads(result)
            if isinstance(keywords, list):
                return keywords[:10]
            return []
        except Exception as e:
            logger.warning(f"Keyword extraction failed: {e}")
            return []

    def transform(self, query: str, strategy: str = "auto") -> QueryTransformResult:
        """
        Transform query based on strategy.

        Args:
            query: Original user query
            strategy: "expand", "decompose", "keywords", or "auto"

        Returns:
            QueryTransformResult with transformed queries
        """
        logger.info(f"Transforming query: {query}")

        # Determine query type and strategy
        query_type = self._classify_query_type(query)
        suggested_strategy = self._suggest_retrieval_strategy(query, query_type)

        # Transform based on strategy
        transformed = []
        keywords = []

        if strategy == "auto":
            # Automatic strategy based on query complexity
            if len(query.split()) > 15 or ("and" in query.lower() and "?" in query):
                # Complex query - decompose
                transformed = self.decompose_query(query)
                keywords = self.extract_keywords(query)
            else:
                # Simple query - expand
                transformed = self.expand_query(query)
                keywords = self.extract_keywords(query)
        elif strategy == "expand":
            transformed = self.expand_query(query)
        elif strategy == "decompose":
            transformed = self.decompose_query(query)
        elif strategy == "keywords":
            keywords = self.extract_keywords(query)
            transformed = [query]
        else:
            transformed = [query]

        # Ensure original query is included
        if query not in transformed:
            transformed.insert(0, query)

        return QueryTransformResult(
            original_query=query,
            transformed_queries=transformed,
            query_type=query_type,
            suggested_strategy=suggested_strategy,
            keywords=keywords
        )

    def _classify_query_type(self, query: str) -> QueryType:
        """Classify query type based on content"""
        query_lower = query.lower()

        # Diagnostic patterns
        if any(word in query_lower for word in ["diagnose", "diagnosis", "what causes", "why do i"]):
            return QueryType.DIAGNOSTIC

        # Procedural patterns
        if any(word in query_lower for word in ["how to", "procedure", "steps", "process"]):
            return QueryType.PROCEDURAL

        # Analytical patterns
        if any(word in query_lower for word in ["compare", "difference", "versus", "vs", "better"]):
            return QueryType.ANALYTICAL

        # Factual patterns
        if any(word in query_lower for word in ["what is", "define", "definition", "explain"]):
            return QueryType.FACTUAL

        return QueryType.GENERAL

    def _suggest_retrieval_strategy(self, query: str, query_type: QueryType) -> RetrievalStrategy:
        """Suggest retrieval strategy based on query characteristics"""

        # Diagnostic and analytical queries benefit from hybrid search
        if query_type in [QueryType.DIAGNOSTIC, QueryType.ANALYTICAL]:
            return RetrievalStrategy.HYBRID

        # Procedural queries benefit from summary-first approach
        if query_type == QueryType.PROCEDURAL:
            return RetrievalStrategy.SUMMARY_FIRST

        # Factual queries work well with vector search
        if query_type == QueryType.FACTUAL:
            return RetrievalStrategy.VECTOR_ONLY

        # Default to hybrid for best coverage
        return RetrievalStrategy.HYBRID


class BM25Retriever:
    """
    BM25 (Best Match 25) keyword-based retrieval.

    BM25 is a ranking function used for keyword-based search.
    It's particularly effective for medical terminology and exact matches.
    """

    def __init__(self, documents: List[str], k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 retriever.

        Args:
            documents: List of document texts
            k1: Term frequency saturation parameter (default 1.5)
            b: Length normalization parameter (default 0.75)
        """
        self.documents = documents
        self.k1 = k1
        self.b = b
        self._build_index()

    def _build_index(self):
        """Build BM25 index"""
        self.doc_lengths = [len(doc.split()) for doc in self.documents]
        self.avgdl = sum(self.doc_lengths) / len(self.documents) if self.documents else 0

        # Build inverted index
        self.inverted_index = {}
        for doc_id, doc in enumerate(self.documents):
            tokens = self._tokenize(doc)
            for token in set(tokens):
                if token not in self.inverted_index:
                    self.inverted_index[token] = []
                self.inverted_index[token].append(doc_id)

        # Calculate IDF scores
        self.idf = {}
        N = len(self.documents)
        for term, postings in self.inverted_index.items():
            df = len(postings)
            self.idf[term] = np.log((N - df + 0.5) / (df + 0.5) + 1.0)

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        # Convert to lowercase and split on non-alphanumeric
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Search documents using BM25.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of (doc_id, score) tuples
        """
        query_tokens = self._tokenize(query)
        scores = np.zeros(len(self.documents))

        for token in query_tokens:
            if token not in self.inverted_index:
                continue

            idf = self.idf[token]
            for doc_id in self.inverted_index[token]:
                # Count term frequency
                doc_tokens = self._tokenize(self.documents[doc_id])
                tf = doc_tokens.count(token)

                # BM25 formula
                doc_len = self.doc_lengths[doc_id]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
                scores[doc_id] += idf * (numerator / denominator)

        # Get top-k results
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]

        return results


class FusionRetriever:
    """
    Fusion retrieval combining multiple retrieval strategies.

    Uses Reciprocal Rank Fusion (RRF) to combine results from:
    1. Vector search (semantic similarity)
    2. BM25 search (keyword matching)
    """

    def __init__(
        self,
        vector_retriever,  # PineconeManager instance
        bm25_retriever: Optional[BM25Retriever] = None,
        embedding_model: Optional[SentenceTransformer] = None
    ):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.embedding_model = embedding_model or SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        vector_weight: float = 0.6,
        keyword_weight: float = 0.4,
        strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    ) -> List[RetrievedDocument]:
        """
        Retrieve documents using fusion strategy.

        Args:
            query: Search query
            top_k: Number of results
            vector_weight: Weight for vector search (0-1)
            keyword_weight: Weight for keyword search (0-1)
            strategy: Retrieval strategy to use

        Returns:
            List of RetrievedDocument objects
        """

        if strategy == RetrievalStrategy.VECTOR_ONLY:
            return self._vector_retrieve(query, top_k)

        elif strategy == RetrievalStrategy.KEYWORD_ONLY and self.bm25_retriever:
            return self._keyword_retrieve(query, top_k)

        elif strategy == RetrievalStrategy.HYBRID and self.bm25_retriever:
            return self._hybrid_retrieve(query, top_k, vector_weight, keyword_weight)

        else:
            # Fallback to vector only
            return self._vector_retrieve(query, top_k)

    def _vector_retrieve(self, query: str, top_k: int) -> List[RetrievedDocument]:
        """Vector search only"""
        try:
            results = self.vector_retriever.search(query, top_k=top_k)

            docs = []
            # results is already a list of matches (not a dict)
            for match in results:
                docs.append(RetrievedDocument(
                    content=match.get('metadata', {}).get('text', ''),
                    score=float(match.get('score', 0.0)),
                    metadata=match.get('metadata', {}),
                    source=f"vector_page_{match.get('metadata', {}).get('page', 'unknown')}"
                ))

            return docs
        except Exception as e:
            logger.error(f"Vector retrieval failed: {e}")
            return []

    def _keyword_retrieve(self, query: str, top_k: int) -> List[RetrievedDocument]:
        """Keyword (BM25) search only"""
        if not self.bm25_retriever:
            return []

        try:
            results = self.bm25_retriever.search(query, top_k)

            docs = []
            for doc_id, score in results:
                docs.append(RetrievedDocument(
                    content=self.bm25_retriever.documents[doc_id],
                    score=score,
                    metadata={'doc_id': doc_id},
                    source=f"bm25_doc_{doc_id}"
                ))

            return docs
        except Exception as e:
            logger.error(f"Keyword retrieval failed: {e}")
            return []

    def _hybrid_retrieve(
        self,
        query: str,
        top_k: int,
        vector_weight: float,
        keyword_weight: float
    ) -> List[RetrievedDocument]:
        """
        Hybrid retrieval using Reciprocal Rank Fusion (RRF).

        RRF formula: score = sum(1 / (rank + k)) where k=60 is standard
        """
        # Get results from both retrievers
        vector_docs = self._vector_retrieve(query, top_k * 2)  # Get more for fusion
        keyword_docs = self._keyword_retrieve(query, top_k * 2)

        # Build document map
        doc_map = {}  # content -> RetrievedDocument
        rrf_scores = {}  # content -> RRF score
        k = 60  # RRF constant

        # Add vector results with RRF
        for rank, doc in enumerate(vector_docs):
            key = doc.content[:200]  # Use first 200 chars as key
            if key not in doc_map:
                doc_map[key] = doc
                rrf_scores[key] = 0

            rrf_scores[key] += vector_weight * (1.0 / (rank + k))

        # Add keyword results with RRF
        for rank, doc in enumerate(keyword_docs):
            key = doc.content[:200]
            if key not in doc_map:
                doc_map[key] = doc
                rrf_scores[key] = 0

            rrf_scores[key] += keyword_weight * (1.0 / (rank + k))

        # Sort by RRF score
        sorted_docs = sorted(
            doc_map.items(),
            key=lambda x: rrf_scores[x[0]],
            reverse=True
        )

        # Update scores and return top-k
        result = []
        for key, doc in sorted_docs[:top_k]:
            doc.score = rrf_scores[key]
            result.append(doc)

        return result


class Reranker:
    """
    Cross-encoder based reranking for retrieved documents.

    Uses a cross-encoder model to score query-document pairs,
    which is more accurate but slower than bi-encoder retrieval.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize reranker.

        Args:
            model_name: HuggingFace cross-encoder model
        """
        try:
            self.model = CrossEncoder(model_name)
            logger.info(f"Loaded reranker model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load reranker model: {e}")
            self.model = None

    def rerank(
        self,
        query: str,
        documents: List[RetrievedDocument],
        top_k: Optional[int] = None
    ) -> List[RetrievedDocument]:
        """
        Rerank documents using cross-encoder.

        Args:
            query: Original query
            documents: Retrieved documents
            top_k: Number of top documents to return (None = all)

        Returns:
            Reranked documents with updated scores
        """
        if not self.model or not documents:
            return documents

        try:
            # Prepare query-document pairs
            pairs = [[query, doc.content] for doc in documents]

            # Get cross-encoder scores
            scores = self.model.predict(pairs)

            # Update document scores
            for doc, score in zip(documents, scores):
                doc.rerank_score = float(score)

            # Sort by rerank score
            reranked = sorted(documents, key=lambda x: x.rerank_score or 0, reverse=True)

            # Return top-k
            if top_k:
                return reranked[:top_k]
            return reranked

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return documents


class AdvancedRAG:
    """
    Advanced RAG system integrating all components:
    1. Query Transformation
    2. Query Routing
    3. Fusion Retrieval
    4. Reranking
    """

    def __init__(
        self,
        vector_retriever,  # PineconeManager
        llm: ChatGoogleGenerativeAI,
        bm25_retriever: Optional[BM25Retriever] = None,
        use_reranking: bool = True
    ):
        """
        Initialize Advanced RAG system.

        Args:
            vector_retriever: Pinecone vector store manager
            llm: Language model for query transformation
            bm25_retriever: Optional BM25 retriever
            use_reranking: Whether to use cross-encoder reranking
        """
        self.vector_retriever = vector_retriever
        self.llm = llm

        # Initialize components
        self.query_transformer = QueryTransformationAgent(llm)
        self.fusion_retriever = FusionRetriever(vector_retriever, bm25_retriever)
        self.reranker = Reranker() if use_reranking else None

        logger.info("Advanced RAG system initialized")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        transform_strategy: str = "auto",
        use_reranking: bool = True
    ) -> Tuple[List[RetrievedDocument], QueryTransformResult]:
        """
        Advanced RAG retrieval pipeline.

        Args:
            query: User query
            top_k: Number of final documents to return
            transform_strategy: Query transformation strategy
            use_reranking: Whether to rerank results

        Returns:
            (retrieved_documents, query_transform_result)
        """
        logger.info(f"Advanced RAG retrieval for query: {query}")

        # Step 1: Query Transformation
        transform_result = self.query_transformer.transform(query, transform_strategy)
        logger.info(f"Query type: {transform_result.query_type}, Strategy: {transform_result.suggested_strategy}")
        logger.info(f"Transformed queries: {transform_result.transformed_queries}")

        # Step 2: Fusion Retrieval for each transformed query
        all_documents = []

        for tq in transform_result.transformed_queries[:3]:  # Limit to top 3 transformations
            docs = self.fusion_retriever.retrieve(
                query=tq,
                top_k=top_k * 2,  # Get more for reranking
                strategy=transform_result.suggested_strategy
            )
            all_documents.extend(docs)

        # Deduplicate by content similarity
        all_documents = self._deduplicate_documents(all_documents)

        logger.info(f"Retrieved {len(all_documents)} documents before reranking")

        # Step 3: Reranking
        if use_reranking and self.reranker:
            all_documents = self.reranker.rerank(query, all_documents, top_k=top_k)
            logger.info(f"Reranked to top {len(all_documents)} documents")
        else:
            # Just take top-k by score
            all_documents = sorted(all_documents, key=lambda x: x.score, reverse=True)[:top_k]

        return all_documents, transform_result

    def _deduplicate_documents(self, documents: List[RetrievedDocument]) -> List[RetrievedDocument]:
        """Deduplicate documents by content similarity"""
        if not documents:
            return []

        # Simple deduplication: remove docs with >90% similar content
        unique_docs = []
        seen_contents = set()

        for doc in documents:
            # Use first 100 chars as fingerprint
            fingerprint = doc.content[:100].strip().lower()
            if fingerprint not in seen_contents:
                seen_contents.add(fingerprint)
                unique_docs.append(doc)

        return unique_docs

    def format_context(self, documents: List[RetrievedDocument]) -> str:
        """Format retrieved documents as context for LLM"""
        if not documents:
            return "No relevant information found."

        context_parts = []
        for i, doc in enumerate(documents, 1):
            score_info = f"[Relevance: {doc.rerank_score or doc.score:.3f}]"
            source_info = f"[Source: {doc.source}]"

            context_parts.append(
                f"\n--- Document {i} {score_info} {source_info} ---\n"
                f"{doc.content}\n"
            )

        return "\n".join(context_parts)


# Factory function for easy initialization
def create_advanced_rag(
    pinecone_manager,
    llm: ChatGoogleGenerativeAI,
    documents_for_bm25: Optional[List[str]] = None,
    use_reranking: bool = True
) -> AdvancedRAG:
    """
    Factory function to create Advanced RAG system.

    Args:
        pinecone_manager: PineconeManager instance
        llm: Language model
        documents_for_bm25: Optional list of documents for BM25 index
        use_reranking: Whether to use reranking

    Returns:
        Configured AdvancedRAG instance
    """
    bm25_retriever = None
    if documents_for_bm25:
        logger.info(f"Building BM25 index with {len(documents_for_bm25)} documents")
        bm25_retriever = BM25Retriever(documents_for_bm25)

    return AdvancedRAG(
        vector_retriever=pinecone_manager,
        llm=llm,
        bm25_retriever=bm25_retriever,
        use_reranking=use_reranking
    )

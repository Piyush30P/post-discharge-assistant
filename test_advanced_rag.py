#!/usr/bin/env python3
"""
Advanced RAG Testing and Evaluation Script
===========================================

This script tests and evaluates the Advanced RAG system with:
1. Query transformation evaluation
2. Retrieval quality assessment
3. Comparison with basic RAG
4. Performance metrics
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import List, Dict, Any
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import GOOGLE_API_KEY, PINECONE_API_KEY
from pinecone_manager import PineconeManager
from advanced_rag import create_advanced_rag, BM25Retriever
from summary_index import SummaryIndex
from langchain_google_genai import ChatGoogleGenerativeAI
import pickle

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AdvancedRAGEvaluator:
    """Evaluates Advanced RAG system performance"""

    def __init__(self):
        """Initialize evaluator"""
        logger.info("Initializing Advanced RAG Evaluator...")

        # Initialize components
        self.pinecone_manager = PineconeManager()

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.3
        )

        # Load BM25 index
        bm25_path = Path("data/bm25_index.pkl")
        self.bm25_retriever = None

        if bm25_path.exists():
            logger.info("Loading BM25 index...")
            with open(bm25_path, 'rb') as f:
                self.bm25_retriever = pickle.load(f)
            logger.info(f"BM25 index loaded with {len(self.bm25_retriever.documents)} documents")
        else:
            logger.warning("BM25 index not found. Some tests will be skipped.")

        # Create Advanced RAG system
        self.advanced_rag = create_advanced_rag(
            pinecone_manager=self.pinecone_manager,
            llm=self.llm,
            documents_for_bm25=self.bm25_retriever.documents if self.bm25_retriever else None,
            use_reranking=True
        )

        # Load summary index
        summary_path = Path("data/summary_index.pkl")
        self.summary_index = None
        if summary_path.exists():
            self.summary_index = SummaryIndex(
                llm=self.llm,
                storage_path=str(summary_path)
            )
            logger.info(f"Summary index loaded with {len(self.summary_index)} summaries")

        logger.info("✓ Evaluator initialized")

    def get_test_queries(self) -> List[Dict[str, Any]]:
        """Get test queries for evaluation"""
        return [
            {
                "query": "What are the symptoms of chronic kidney disease?",
                "type": "factual",
                "expected_concepts": ["symptoms", "CKD", "kidney", "renal"]
            },
            {
                "query": "What are the differences between acute and chronic kidney disease?",
                "type": "analytical",
                "expected_concepts": ["acute", "chronic", "kidney disease", "differences"]
            },
            {
                "query": "How is dialysis performed?",
                "type": "procedural",
                "expected_concepts": ["dialysis", "procedure", "hemodialysis"]
            },
            {
                "query": "What causes protein in urine?",
                "type": "diagnostic",
                "expected_concepts": ["proteinuria", "protein", "urine", "causes"]
            },
            {
                "query": "kidney failure treatment options",
                "type": "general",
                "expected_concepts": ["kidney failure", "treatment", "dialysis", "transplant"]
            },
            {
                "query": "What medications are used for high blood pressure in kidney patients?",
                "type": "factual",
                "expected_concepts": ["hypertension", "medication", "blood pressure", "ACE inhibitors"]
            },
            {
                "query": "Compare hemodialysis and peritoneal dialysis",
                "type": "analytical",
                "expected_concepts": ["hemodialysis", "peritoneal dialysis", "comparison"]
            },
            {
                "query": "What is glomerular filtration rate?",
                "type": "factual",
                "expected_concepts": ["GFR", "glomerular", "filtration", "kidney function"]
            }
        ]

    def test_query_transformation(self):
        """Test query transformation capabilities"""
        logger.info("\n" + "="*80)
        logger.info("TEST 1: Query Transformation")
        logger.info("="*80)

        test_queries = [
            "kidney failure symptoms",
            "What are the differences between acute and chronic kidney disease?",
            "How is dialysis performed?"
        ]

        results = []

        for query in test_queries:
            logger.info(f"\nOriginal Query: {query}")
            logger.info("-" * 60)

            transform_result = self.advanced_rag.query_transformer.transform(query, strategy="auto")

            logger.info(f"Query Type: {transform_result.query_type.value}")
            logger.info(f"Suggested Strategy: {transform_result.suggested_strategy.value}")
            logger.info(f"\nTransformed Queries:")
            for i, tq in enumerate(transform_result.transformed_queries, 1):
                logger.info(f"  {i}. {tq}")

            if transform_result.keywords:
                logger.info(f"\nExtracted Keywords: {', '.join(transform_result.keywords)}")

            results.append({
                "original": query,
                "type": transform_result.query_type.value,
                "transformed": transform_result.transformed_queries,
                "keywords": transform_result.keywords
            })

        logger.info("\n✓ Query transformation test complete")
        return results

    def test_retrieval_comparison(self):
        """Compare basic vs advanced retrieval"""
        logger.info("\n" + "="*80)
        logger.info("TEST 2: Retrieval Comparison (Basic vs Advanced)")
        logger.info("="*80)

        test_queries = self.get_test_queries()[:4]  # Use first 4 queries
        results = []

        for test in test_queries:
            query = test["query"]
            logger.info(f"\nQuery: {query}")
            logger.info(f"Type: {test['type']}")
            logger.info("-" * 60)

            # Basic retrieval (vector only)
            start_time = time.time()
            basic_results = self.pinecone_manager.search(query, top_k=5)
            basic_time = time.time() - start_time

            basic_docs = basic_results.get('matches', [])
            logger.info(f"\n[BASIC] Retrieved {len(basic_docs)} documents in {basic_time:.3f}s")

            if basic_docs:
                logger.info(f"  Top result score: {basic_docs[0].get('score', 0):.3f}")
                logger.info(f"  Preview: {basic_docs[0].get('metadata', {}).get('text', '')[:100]}...")

            # Advanced retrieval (with transformation, fusion, reranking)
            start_time = time.time()
            advanced_docs, transform_result = self.advanced_rag.retrieve(
                query=query,
                top_k=5,
                transform_strategy="auto",
                use_reranking=True
            )
            advanced_time = time.time() - start_time

            logger.info(f"\n[ADVANCED] Retrieved {len(advanced_docs)} documents in {advanced_time:.3f}s")
            logger.info(f"  Query type: {transform_result.query_type.value}")
            logger.info(f"  Strategy: {transform_result.suggested_strategy.value}")

            if advanced_docs:
                top_doc = advanced_docs[0]
                logger.info(f"  Top result rerank score: {top_doc.rerank_score or top_doc.score:.3f}")
                logger.info(f"  Preview: {top_doc.content[:100]}...")

            # Calculate coverage of expected concepts
            basic_text = " ".join([d.get('metadata', {}).get('text', '') for d in basic_docs[:3]])
            advanced_text = " ".join([d.content for d in advanced_docs[:3]])

            basic_coverage = sum(1 for c in test["expected_concepts"] if c.lower() in basic_text.lower())
            advanced_coverage = sum(1 for c in test["expected_concepts"] if c.lower() in advanced_text.lower())

            logger.info(f"\n[COVERAGE] Expected concepts: {test['expected_concepts']}")
            logger.info(f"  Basic: {basic_coverage}/{len(test['expected_concepts'])}")
            logger.info(f"  Advanced: {advanced_coverage}/{len(test['expected_concepts'])}")

            results.append({
                "query": query,
                "type": test["type"],
                "basic": {
                    "num_results": len(basic_docs),
                    "time": basic_time,
                    "coverage": basic_coverage
                },
                "advanced": {
                    "num_results": len(advanced_docs),
                    "time": advanced_time,
                    "coverage": advanced_coverage,
                    "query_type": transform_result.query_type.value,
                    "strategy": transform_result.suggested_strategy.value
                }
            })

        logger.info("\n✓ Retrieval comparison test complete")
        return results

    def test_reranking_impact(self):
        """Test impact of reranking"""
        logger.info("\n" + "="*80)
        logger.info("TEST 3: Reranking Impact")
        logger.info("="*80)

        query = "What are the symptoms of chronic kidney disease?"
        logger.info(f"\nQuery: {query}")
        logger.info("-" * 60)

        # Retrieve without reranking
        docs_no_rerank, _ = self.advanced_rag.retrieve(
            query=query,
            top_k=5,
            use_reranking=False
        )

        logger.info("\n[WITHOUT RERANKING]")
        for i, doc in enumerate(docs_no_rerank, 1):
            logger.info(f"{i}. Score: {doc.score:.3f}")
            logger.info(f"   Preview: {doc.content[:80]}...")

        # Retrieve with reranking
        docs_with_rerank, _ = self.advanced_rag.retrieve(
            query=query,
            top_k=5,
            use_reranking=True
        )

        logger.info("\n[WITH RERANKING]")
        for i, doc in enumerate(docs_with_rerank, 1):
            logger.info(f"{i}. Rerank Score: {doc.rerank_score or doc.score:.3f}")
            logger.info(f"   Preview: {doc.content[:80]}...")

        logger.info("\n✓ Reranking impact test complete")

        return {
            "without_rerank": [{"score": d.score, "preview": d.content[:100]} for d in docs_no_rerank],
            "with_rerank": [{"score": d.rerank_score or d.score, "preview": d.content[:100]} for d in docs_with_rerank]
        }

    def test_summary_index(self):
        """Test summary index"""
        logger.info("\n" + "="*80)
        logger.info("TEST 4: Summary Index")
        logger.info("="*80)

        if not self.summary_index:
            logger.warning("Summary index not available, skipping test")
            return None

        test_queries = [
            "kidney disease",
            "dialysis",
            "blood pressure"
        ]

        results = []

        for query in test_queries:
            logger.info(f"\nQuery: {query}")
            logger.info("-" * 60)

            summaries = self.summary_index.search(query, top_k=3)

            logger.info(f"Found {len(summaries)} summaries")

            for i, summary in enumerate(summaries, 1):
                logger.info(f"\n{i}. {summary.doc_id}")
                logger.info(f"   Summary: {summary.summary}")
                logger.info(f"   Key Concepts: {', '.join(summary.key_concepts)}")

            results.append({
                "query": query,
                "num_summaries": len(summaries),
                "summaries": [
                    {
                        "doc_id": s.doc_id,
                        "summary": s.summary,
                        "concepts": s.key_concepts
                    }
                    for s in summaries
                ]
            })

        logger.info("\n✓ Summary index test complete")
        return results

    def test_hybrid_search(self):
        """Test hybrid search (vector + BM25)"""
        logger.info("\n" + "="*80)
        logger.info("TEST 5: Hybrid Search (Vector + BM25)")
        logger.info("="*80)

        if not self.bm25_retriever:
            logger.warning("BM25 index not available, skipping test")
            return None

        query = "chronic kidney disease symptoms treatment"
        logger.info(f"\nQuery: {query}")
        logger.info("-" * 60)

        # Vector only
        from src.advanced_rag import RetrievalStrategy

        vector_docs = self.advanced_rag.fusion_retriever.retrieve(
            query=query,
            top_k=5,
            strategy=RetrievalStrategy.VECTOR_ONLY
        )

        logger.info(f"\n[VECTOR ONLY] Retrieved {len(vector_docs)} documents")
        for i, doc in enumerate(vector_docs[:3], 1):
            logger.info(f"{i}. Score: {doc.score:.3f}")
            logger.info(f"   Preview: {doc.content[:80]}...")

        # BM25 only
        keyword_docs = self.advanced_rag.fusion_retriever.retrieve(
            query=query,
            top_k=5,
            strategy=RetrievalStrategy.KEYWORD_ONLY
        )

        logger.info(f"\n[BM25 ONLY] Retrieved {len(keyword_docs)} documents")
        for i, doc in enumerate(keyword_docs[:3], 1):
            logger.info(f"{i}. Score: {doc.score:.3f}")
            logger.info(f"   Preview: {doc.content[:80]}...")

        # Hybrid (fusion)
        hybrid_docs = self.advanced_rag.fusion_retriever.retrieve(
            query=query,
            top_k=5,
            strategy=RetrievalStrategy.HYBRID
        )

        logger.info(f"\n[HYBRID (FUSION)] Retrieved {len(hybrid_docs)} documents")
        for i, doc in enumerate(hybrid_docs[:3], 1):
            logger.info(f"{i}. Fusion Score: {doc.score:.3f}")
            logger.info(f"   Preview: {doc.content[:80]}...")

        logger.info("\n✓ Hybrid search test complete")

        return {
            "vector": [{"score": d.score, "preview": d.content[:100]} for d in vector_docs],
            "bm25": [{"score": d.score, "preview": d.content[:100]} for d in keyword_docs],
            "hybrid": [{"score": d.score, "preview": d.content[:100]} for d in hybrid_docs]
        }

    def run_all_tests(self):
        """Run all tests"""
        logger.info("\n" + "="*80)
        logger.info("ADVANCED RAG EVALUATION SUITE")
        logger.info("="*80)

        all_results = {}

        # Test 1: Query Transformation
        all_results["query_transformation"] = self.test_query_transformation()

        # Test 2: Retrieval Comparison
        all_results["retrieval_comparison"] = self.test_retrieval_comparison()

        # Test 3: Reranking Impact
        all_results["reranking"] = self.test_reranking_impact()

        # Test 4: Summary Index
        all_results["summary_index"] = self.test_summary_index()

        # Test 5: Hybrid Search
        all_results["hybrid_search"] = self.test_hybrid_search()

        # Save results
        results_path = Path("data/advanced_rag_evaluation_results.json")
        results_path.parent.mkdir(parents=True, exist_ok=True)

        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

        logger.info("\n" + "="*80)
        logger.info("ALL TESTS COMPLETE")
        logger.info("="*80)
        logger.info(f"\nResults saved to: {results_path}")

        # Summary statistics
        logger.info("\n" + "="*80)
        logger.info("SUMMARY STATISTICS")
        logger.info("="*80)

        if "retrieval_comparison" in all_results:
            comparison = all_results["retrieval_comparison"]

            avg_basic_coverage = sum(r["basic"]["coverage"] for r in comparison) / len(comparison)
            avg_advanced_coverage = sum(r["advanced"]["coverage"] for r in comparison) / len(comparison)

            avg_basic_time = sum(r["basic"]["time"] for r in comparison) / len(comparison)
            avg_advanced_time = sum(r["advanced"]["time"] for r in comparison) / len(comparison)

            logger.info(f"\nAverage Concept Coverage:")
            logger.info(f"  Basic RAG: {avg_basic_coverage:.2f}")
            logger.info(f"  Advanced RAG: {avg_advanced_coverage:.2f}")
            logger.info(f"  Improvement: {((avg_advanced_coverage - avg_basic_coverage) / avg_basic_coverage * 100):.1f}%")

            logger.info(f"\nAverage Retrieval Time:")
            logger.info(f"  Basic RAG: {avg_basic_time:.3f}s")
            logger.info(f"  Advanced RAG: {avg_advanced_time:.3f}s")

        return all_results


def main():
    """Main function"""
    logger.info("Starting Advanced RAG Evaluation...")

    # Check prerequisites
    if not GOOGLE_API_KEY:
        logger.error("GOOGLE_API_KEY not set")
        sys.exit(1)

    if not PINECONE_API_KEY:
        logger.error("PINECONE_API_KEY not set")
        sys.exit(1)

    # Check if indices exist
    bm25_path = Path("data/bm25_index.pkl")
    summary_path = Path("data/summary_index.pkl")

    if not bm25_path.exists() or not summary_path.exists():
        logger.warning("\n" + "="*80)
        logger.warning("WARNING: Advanced RAG indices not found!")
        logger.warning("="*80)
        logger.warning("\nPlease run: python setup_advanced_rag.py")
        logger.warning("\nSome tests will be skipped without the indices.")
        logger.warning("")

        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            logger.info("Exiting...")
            sys.exit(0)

    # Run evaluation
    evaluator = AdvancedRAGEvaluator()
    results = evaluator.run_all_tests()

    logger.info("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()

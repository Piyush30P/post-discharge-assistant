#!/usr/bin/env python3
"""
Advanced RAG Setup Script
==========================

This script initializes the Advanced RAG system by:
1. Fetching all documents from Pinecone
2. Building BM25 keyword index
3. Creating summary index
4. Saving indices to disk for fast loading
"""

import os
import sys
import logging
from pathlib import Path
import pickle
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    GOOGLE_API_KEY,
    LOG_LEVEL
)
from pinecone_manager import PineconeManager
from advanced_rag import BM25Retriever
from summary_index import SummaryIndex
from langchain_google_genai import ChatGoogleGenerativeAI

# Setup logging
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def fetch_all_documents_from_pinecone(pinecone_manager: PineconeManager) -> List[Dict[str, Any]]:
    """
    Fetch all documents from Pinecone vector store.

    Returns:
        List of documents with text and metadata
    """
    logger.info("Fetching all documents from Pinecone...")

    try:
        # Get index stats
        stats = pinecone_manager.index.describe_index_stats()
        total_vectors = stats.get('total_vector_count', 0)
        logger.info(f"Total vectors in Pinecone: {total_vectors}")

        if total_vectors == 0:
            logger.warning("No vectors found in Pinecone. Run setup_phase2.py first.")
            return []

        # Fetch vectors using query with dummy vector
        # This is a workaround to get all vectors since Pinecone doesn't have list_all
        documents = []

        # Strategy: Use multiple dummy queries to fetch different vectors
        import numpy as np
        from sentence_transformers import SentenceTransformer

        # Load embedding model
        embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        # Query with different medical terms to get diverse results
        search_terms = [
            "kidney disease treatment",
            "dialysis procedure",
            "chronic renal failure",
            "nephrology diagnosis",
            "urinary system",
            "blood pressure",
            "protein in urine",
            "kidney transplant",
            "acute kidney injury",
            "glomerular filtration"
        ]

        seen_ids = set()

        for term in search_terms:
            logger.info(f"Querying with term: {term}")

            # Query Pinecone
            results = pinecone_manager.search(term, top_k=1000)  # Get up to 1000 per query

            # results is already a list of matches (not a dict)
            for match in results:
                vec_id = match.get('id', '')

                if vec_id not in seen_ids:
                    seen_ids.add(vec_id)

                    text = match.get('metadata', {}).get('text', '')
                    if text:
                        documents.append({
                            'id': vec_id,
                            'text': text,
                            'metadata': match.get('metadata', {})
                        })

            logger.info(f"Total unique documents collected: {len(documents)}")

            # Stop if we have enough
            if len(documents) >= min(total_vectors, 5000):
                break

        logger.info(f"Successfully fetched {len(documents)} documents from Pinecone")
        return documents

    except Exception as e:
        logger.error(f"Error fetching documents from Pinecone: {e}")
        return []


def build_bm25_index(documents: List[Dict[str, Any]]) -> BM25Retriever:
    """
    Build BM25 index from documents.

    Args:
        documents: List of document dicts with 'text' field

    Returns:
        BM25Retriever instance
    """
    logger.info(f"Building BM25 index from {len(documents)} documents...")

    # Extract text content
    texts = [doc['text'] for doc in documents if doc.get('text')]

    logger.info(f"Building BM25 index with {len(texts)} text chunks")

    # Create BM25 retriever
    bm25 = BM25Retriever(documents=texts)

    logger.info("BM25 index built successfully")
    return bm25


def build_summary_index(documents: List[Dict[str, Any]], llm: ChatGoogleGenerativeAI) -> SummaryIndex:
    """
    Build summary index from documents.

    Args:
        documents: List of document dicts
        llm: Language model for summarization

    Returns:
        SummaryIndex instance
    """
    logger.info(f"Building summary index from {len(documents)} documents...")

    # Create summary index
    summary_index = SummaryIndex(llm=llm, storage_path="data/summary_index.pkl")

    # Clear existing summaries
    summary_index.clear()

    # Add documents in batches
    doc_batch = []
    for doc in documents[:500]:  # Limit to 500 for now to save on API calls
        doc_batch.append({
            'text': doc.get('text', ''),
            'doc_id': doc.get('id', ''),
            'metadata': doc.get('metadata', {})
        })

    summary_index.add_documents_batch(doc_batch, batch_size=10)

    logger.info("Summary index built successfully")
    return summary_index


def save_bm25_index(bm25: BM25Retriever, filepath: str = "data/bm25_index.pkl"):
    """Save BM25 index to disk"""
    logger.info(f"Saving BM25 index to {filepath}...")

    # Create directory if needed
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    # Save
    with open(filepath, 'wb') as f:
        pickle.dump(bm25, f)

    logger.info("BM25 index saved successfully")


def load_bm25_index(filepath: str = "data/bm25_index.pkl") -> BM25Retriever:
    """Load BM25 index from disk"""
    logger.info(f"Loading BM25 index from {filepath}...")

    with open(filepath, 'rb') as f:
        bm25 = pickle.load(f)

    logger.info(f"BM25 index loaded with {len(bm25.documents)} documents")
    return bm25


def main():
    """Main setup function"""
    logger.info("=" * 60)
    logger.info("Advanced RAG Setup")
    logger.info("=" * 60)

    # Check environment variables
    if not PINECONE_API_KEY:
        logger.error("PINECONE_API_KEY not set")
        sys.exit(1)

    if not GOOGLE_API_KEY:
        logger.error("GOOGLE_API_KEY not set")
        sys.exit(1)

    # Initialize Pinecone
    logger.info("\n1. Connecting to Pinecone...")
    pinecone_manager = PineconeManager()

    # Connect to the index
    try:
        pinecone_manager.connect_to_index()
        logger.info(f"✓ Connected to Pinecone index: {PINECONE_INDEX_NAME}")

        # Verify index has data
        stats = pinecone_manager.index.describe_index_stats()
        total_vectors = stats.get('total_vector_count', 0)

        if total_vectors == 0:
            logger.error("\n❌ Pinecone index is EMPTY!")
            logger.error("\nYou need to populate Pinecone first:")
            logger.error("  python setup_phase2.py")
            logger.error("\nThis will:")
            logger.error("  1. Create the Pinecone index")
            logger.error("  2. Process the medical knowledge PDF")
            logger.error("  3. Upload ~4000 document chunks to Pinecone")
            sys.exit(1)

        logger.info(f"✓ Pinecone index has {total_vectors:,} vectors")

    except Exception as e:
        logger.error(f"Failed to connect to Pinecone index: {e}")
        logger.error("\nPlease ensure:")
        logger.error("  1. Your PINECONE_API_KEY is correct in .env")
        logger.error("  2. The index exists (run setup_phase2.py first)")
        logger.error("  3. The PINECONE_INDEX_NAME is correct")
        sys.exit(1)

    # Fetch documents
    logger.info("\n2. Fetching documents from Pinecone...")
    documents = fetch_all_documents_from_pinecone(pinecone_manager)

    if not documents:
        logger.error("No documents fetched. Cannot proceed.")
        sys.exit(1)

    logger.info(f"Fetched {len(documents)} documents")

    # Build BM25 index
    logger.info("\n3. Building BM25 index...")
    bm25_index = build_bm25_index(documents)
    save_bm25_index(bm25_index, "data/bm25_index.pkl")

    # Build Summary index
    logger.info("\n4. Building Summary index...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3
    )
    summary_index = build_summary_index(documents, llm)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Advanced RAG Setup Complete!")
    logger.info("=" * 60)
    logger.info(f"BM25 Index: {len(bm25_index.documents)} documents")
    logger.info(f"Summary Index: {len(summary_index)} summaries")
    logger.info("\nFiles created:")
    logger.info("  - data/bm25_index.pkl")
    logger.info("  - data/summary_index.pkl")
    logger.info("\nYou can now use the Advanced RAG system!")


if __name__ == "__main__":
    main()

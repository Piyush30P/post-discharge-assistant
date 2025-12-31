"""
Summary Index for Advanced RAG
================================

This module creates and manages document summaries for high-level retrieval.
Summary index is useful for:
1. Quick document overview
2. Routing queries to appropriate document sections
3. Providing context before detailed retrieval
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
import pickle
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Import hybrid LLM config
try:
    from src.llm_config import get_summary_llm
    HYBRID_LLM_AVAILABLE = True
except ImportError:
    HYBRID_LLM_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class DocumentSummary:
    """Represents a summary of a document or document chunk"""
    doc_id: str
    original_text: str
    summary: str
    key_concepts: List[str]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'doc_id': self.doc_id,
            'original_text': self.original_text,
            'summary': self.summary,
            'key_concepts': self.key_concepts,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentSummary':
        """Create from dictionary"""
        return cls(
            doc_id=data['doc_id'],
            original_text=data['original_text'],
            summary=data['summary'],
            key_concepts=data['key_concepts'],
            metadata=data['metadata']
        )


class SummaryIndex:
    """
    Summary index for document collections.

    Generates and stores hierarchical summaries of documents.
    Uses local LLM (Ollama) by default to avoid API quota limits.
    """

    def __init__(self, llm=None, storage_path: str = "data/summary_index.pkl"):
        """
        Initialize summary index.

        Args:
            llm: Optional language model. If None, uses hybrid config (local preferred)
            storage_path: Path to store the index
        """
        if llm is None and HYBRID_LLM_AVAILABLE:
            # Use local LLM for summaries (no quota limits!)
            try:
                self.llm = get_summary_llm(temperature=0.3)
                logger.info("Using local LLM for summary generation")
            except Exception as e:
                logger.warning(f"Local LLM not available: {e}")
                self.llm = llm
        else:
            self.llm = llm

        self.storage_path = Path(storage_path)
        self.summaries: Dict[str, DocumentSummary] = {}
        self._setup_prompts()

        # Load existing index if available
        self.load()

    def _setup_prompts(self):
        """Setup summarization prompts"""
        self.summarize_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a medical document summarization expert.
Generate a concise 2-3 sentence summary of the following medical text.
Focus on the main topic, key findings, and clinical relevance.

Output format:
Summary: [your summary here]
Key Concepts: [concept1, concept2, concept3]

Keep it concise and medically accurate."""),
            ("human", "Text to summarize:\n\n{text}\n\nSummary:")
        ])

    def generate_summary(self, text: str, doc_id: str, metadata: Optional[Dict] = None) -> DocumentSummary:
        """
        Generate summary for a document.

        Args:
            text: Document text
            doc_id: Unique document identifier
            metadata: Optional metadata

        Returns:
            DocumentSummary object
        """
        try:
            # Generate summary using LLM
            chain = self.summarize_prompt | self.llm | StrOutputParser()
            result = chain.invoke({"text": text[:4000]})  # Limit input length

            # Parse result
            summary = ""
            key_concepts = []

            lines = result.strip().split('\n')
            for line in lines:
                if line.startswith("Summary:"):
                    summary = line.replace("Summary:", "").strip()
                elif line.startswith("Key Concepts:"):
                    concepts_str = line.replace("Key Concepts:", "").strip()
                    key_concepts = [c.strip() for c in concepts_str.split(',')]

            # Fallback if parsing failed
            if not summary:
                summary = result[:200]  # Use first 200 chars

            return DocumentSummary(
                doc_id=doc_id,
                original_text=text,
                summary=summary,
                key_concepts=key_concepts,
                metadata=metadata or {}
            )

        except Exception as e:
            logger.error(f"Failed to generate summary for {doc_id}: {e}")
            # Return basic summary on failure
            return DocumentSummary(
                doc_id=doc_id,
                original_text=text,
                summary=text[:200] + "...",
                key_concepts=[],
                metadata=metadata or {}
            )

    def add_document(self, text: str, doc_id: str, metadata: Optional[Dict] = None):
        """Add a document to the index"""
        logger.info(f"Adding document to summary index: {doc_id}")
        summary = self.generate_summary(text, doc_id, metadata)
        self.summaries[doc_id] = summary

    def add_documents_batch(self, documents: List[Dict[str, Any]], batch_size: int = 10):
        """
        Add multiple documents in batches.

        Args:
            documents: List of dicts with 'text', 'doc_id', 'metadata'
            batch_size: Number of documents to process at once
        """
        logger.info(f"Adding {len(documents)} documents to summary index")

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            for doc in batch:
                self.add_document(
                    text=doc.get('text', ''),
                    doc_id=doc.get('doc_id', f'doc_{i}'),
                    metadata=doc.get('metadata', {})
                )

            # Save periodically
            if i % 50 == 0:
                self.save()
                logger.info(f"Processed {i}/{len(documents)} documents")

        # Final save
        self.save()
        logger.info(f"Completed adding {len(documents)} documents")

    def search(self, query: str, top_k: int = 5) -> List[DocumentSummary]:
        """
        Search summaries for relevant documents.

        Simple keyword-based search on summaries and key concepts.

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            List of DocumentSummary objects
        """
        query_lower = query.lower()
        query_terms = set(query_lower.split())

        # Score each summary
        scored_summaries = []
        for doc_id, summary in self.summaries.items():
            score = 0

            # Check summary
            summary_lower = summary.summary.lower()
            for term in query_terms:
                if term in summary_lower:
                    score += 2

            # Check key concepts
            for concept in summary.key_concepts:
                concept_lower = concept.lower()
                for term in query_terms:
                    if term in concept_lower:
                        score += 3

            # Check original text (brief)
            text_lower = summary.original_text[:500].lower()
            for term in query_terms:
                if term in text_lower:
                    score += 1

            if score > 0:
                scored_summaries.append((score, summary))

        # Sort by score and return top-k
        scored_summaries.sort(key=lambda x: x[0], reverse=True)
        return [summary for score, summary in scored_summaries[:top_k]]

    def get_summary(self, doc_id: str) -> Optional[DocumentSummary]:
        """Get summary for a specific document"""
        return self.summaries.get(doc_id)

    def get_all_summaries(self) -> List[DocumentSummary]:
        """Get all summaries"""
        return list(self.summaries.values())

    def save(self):
        """Save index to disk"""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert to dict for storage
            data = {
                doc_id: summary.to_dict()
                for doc_id, summary in self.summaries.items()
            }

            with open(self.storage_path, 'wb') as f:
                pickle.dump(data, f)

            logger.info(f"Saved summary index with {len(self.summaries)} summaries to {self.storage_path}")

        except Exception as e:
            logger.error(f"Failed to save summary index: {e}")

    def load(self):
        """Load index from disk"""
        try:
            if self.storage_path.exists():
                with open(self.storage_path, 'rb') as f:
                    data = pickle.load(f)

                # Convert back to DocumentSummary objects
                self.summaries = {
                    doc_id: DocumentSummary.from_dict(summary_dict)
                    for doc_id, summary_dict in data.items()
                }

                logger.info(f"Loaded summary index with {len(self.summaries)} summaries from {self.storage_path}")
            else:
                logger.info("No existing summary index found, starting fresh")

        except Exception as e:
            logger.error(f"Failed to load summary index: {e}")
            self.summaries = {}

    def clear(self):
        """Clear all summaries"""
        self.summaries = {}
        logger.info("Cleared summary index")

    def __len__(self):
        """Number of summaries in index"""
        return len(self.summaries)

    def __repr__(self):
        return f"SummaryIndex(summaries={len(self.summaries)})"

"""
Enhanced PDF Processor with Sentence-Aware Chunking
Processes PDF efficiently with smart boundaries for better semantic chunking
"""

import re
from pathlib import Path
from typing import List, Dict, Tuple
import fitz  # PyMuPDF

from src.config import NEPHROLOGY_PDF_PATH, CHUNK_SIZE, CHUNK_OVERLAP
from src.utils.logger import get_logger

logger = get_logger()


class EnhancedPDFProcessor:
    """Process medical PDF documents with intelligent chunking"""
    
    def __init__(self, pdf_path: Path = NEPHROLOGY_PDF_PATH):
        self.pdf_path = pdf_path
        self.chunk_size = CHUNK_SIZE
        self.chunk_overlap = CHUNK_OVERLAP
        
    def extract_text_from_pdf(self) -> List[Dict[str, any]]:
        """
        Extract text from PDF with page numbers and metadata
        
        Returns:
            List of dicts with 'page', 'text', and 'metadata'
        """
        if not self.pdf_path.exists():
            raise FileNotFoundError(
                f"PDF not found at: {self.pdf_path}\n"
                f"Please place your nephrology book PDF at this location."
            )
        
        logger.info(f"Opening PDF: {self.pdf_path}")
        
        try:
            doc = fitz.open(self.pdf_path)
            total_pages = len(doc)
            logger.info(f"PDF has {total_pages} pages")
            
            pages_data = []
            
            for page_num in range(total_pages):
                page = doc[page_num]
                text = page.get_text()
                
                # Clean the extracted text
                text = self._clean_text(text)
                
                if text.strip():  # Only add non-empty pages
                    pages_data.append({
                        'page': page_num + 1,
                        'text': text,
                        'metadata': {
                            'source': 'Comprehensive Clinical Nephrology 7th Edition',
                            'page': page_num + 1,
                            'total_pages': total_pages
                        }
                    })
                
                # Progress logging
                if (page_num + 1) % 100 == 0:
                    logger.info(f"Processed {page_num + 1}/{total_pages} pages...")
            
            doc.close()
            logger.info(f"✓ Successfully extracted text from {len(pages_data)} pages")
            return pages_data
            
        except Exception as e:
            logger.log_error("EnhancedPDFProcessor", e)
            raise
    
    def _clean_text(self, text: str) -> str:
        """
        Clean extracted PDF text
        - Remove excessive whitespace
        - Fix line breaks
        - Remove page numbers and headers/footers
        """
        # Remove multiple spaces
        text = re.sub(r' +', ' ', text)
        
        # Remove multiple newlines (but keep paragraph breaks)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Remove standalone page numbers
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Fix hyphenated words at line breaks
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
        
        # Remove excessive whitespace at start/end of lines
        text = '\n'.join(line.strip() for line in text.split('\n'))
        
        return text.strip()
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using medical-aware rules
        Handles abbreviations common in medical text
        """
        # Medical abbreviations that shouldn't trigger sentence splits
        abbreviations = [
            'Dr', 'Mr', 'Mrs', 'Ms', 'Prof', 'Sr', 'Jr',
            'vs', 'etc', 'i.e', 'e.g', 'cf', 'Fig', 'vol',
            'mg', 'ml', 'cm', 'mm', 'kg', 'lb', 'oz',
            'approx', 'min', 'max', 'avg'
        ]
        
        # Temporarily replace abbreviations
        temp_text = text
        for i, abbr in enumerate(abbreviations):
            temp_text = temp_text.replace(f'{abbr}.', f'{abbr}<!ABBR{i}!>')
        
        # Split on sentence boundaries
        # Look for: period/question/exclamation followed by space and capital letter
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', temp_text)
        
        # Restore abbreviations
        restored_sentences = []
        for sentence in sentences:
            for i, abbr in enumerate(abbreviations):
                sentence = sentence.replace(f'{abbr}<!ABBR{i}!>', f'{abbr}.')
            if sentence.strip():
                restored_sentences.append(sentence.strip())
        
        return restored_sentences
    
    def _find_best_split_point(self, sentences: List[str], target_size: int) -> int:
        """
        Find the best point to split sentences to reach target size
        without cutting mid-sentence
        """
        current_size = 0
        split_index = 0
        
        for i, sentence in enumerate(sentences):
            sentence_length = len(sentence)
            
            if current_size + sentence_length > target_size:
                # If we haven't added any sentences yet, include at least one
                if i == 0:
                    return 1
                return i
            
            current_size += sentence_length + 1  # +1 for space
            split_index = i + 1
        
        return split_index
    
    def chunk_text_intelligently(self, pages_data: List[Dict]) -> List[Dict]:
        """
        Chunk text using sentence-aware splitting
        Respects sentence boundaries for better semantic meaning
        
        Args:
            pages_data: List of page dictionaries with text and metadata
            
        Returns:
            List of chunk dictionaries with text and enhanced metadata
        """
        logger.info(f"Starting intelligent chunking (target={self.chunk_size}, overlap={self.chunk_overlap})...")
        
        chunks = []
        chunk_id = 0
        
        for page_data in pages_data:
            text = page_data['text']
            page_num = page_data['page']
            
            # Split into sentences
            sentences = self._split_into_sentences(text)
            
            if not sentences:
                continue
            
            # Process sentences into chunks
            i = 0
            while i < len(sentences):
                # Determine how many sentences to include
                split_point = self._find_best_split_point(
                    sentences[i:], 
                    self.chunk_size
                )
                
                # Create chunk from sentences
                chunk_sentences = sentences[i:i + split_point]
                chunk_text = ' '.join(chunk_sentences)
                
                # Add chunk
                chunks.append({
                    'chunk_id': chunk_id,
                    'text': chunk_text,
                    'metadata': {
                        **page_data['metadata'],
                        'chunk_id': chunk_id,
                        'page': page_num,
                        'sentence_count': len(chunk_sentences),
                        'char_count': len(chunk_text)
                    }
                })
                chunk_id += 1
                
                # Calculate overlap in sentences
                overlap_chars = 0
                overlap_sentences = 0
                for sentence in reversed(chunk_sentences):
                    if overlap_chars + len(sentence) <= self.chunk_overlap:
                        overlap_chars += len(sentence) + 1
                        overlap_sentences += 1
                    else:
                        break
                
                # Move forward, accounting for overlap
                i += max(1, split_point - overlap_sentences)
        
        logger.info(f"✓ Created {len(chunks)} intelligent chunks from {len(pages_data)} pages")
        
        # Log statistics
        chunk_sizes = [len(c['text']) for c in chunks]
        avg_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
        min_size = min(chunk_sizes) if chunk_sizes else 0
        max_size = max(chunk_sizes) if chunk_sizes else 0
        
        logger.info(f"Chunk statistics:")
        logger.info(f"  Average size: {avg_size:.0f} characters")
        logger.info(f"  Min size: {min_size} characters")
        logger.info(f"  Max size: {max_size} characters")
        
        return chunks
    
    def process_pdf(self) -> List[Dict]:
        """
        Complete PDF processing pipeline with intelligent chunking
        
        Returns:
            List of processed chunks ready for embedding
        """
        logger.info("="*80)
        logger.info("Starting Enhanced PDF Processing Pipeline")
        logger.info("="*80)
        
        # Step 1: Extract text
        pages_data = self.extract_text_from_pdf()
        
        # Step 2: Intelligent chunking
        chunks = self.chunk_text_intelligently(pages_data)
        
        # Step 3: Statistics
        total_chars = sum(len(chunk['text']) for chunk in chunks)
        avg_chunk_size = total_chars / len(chunks) if chunks else 0
        
        logger.info("="*80)
        logger.info("PDF Processing Complete")
        logger.info("="*80)
        logger.info(f"Total Pages: {len(pages_data)}")
        logger.info(f"Total Chunks: {len(chunks)}")
        logger.info(f"Average Chunk Size: {avg_chunk_size:.0f} characters")
        logger.info(f"Total Characters: {total_chars:,}")
        
        return chunks
    
    def save_chunks_preview(self, chunks: List[Dict], output_file: Path = None, num_samples: int = 5):
        """Save a preview of chunks for inspection"""
        if output_file is None:
            output_file = self.pdf_path.parent / "chunks_preview.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("NEPHROLOGY BOOK CHUNKS PREVIEW (Sentence-Aware)\n")
            f.write("="*80 + "\n\n")
            
            for i, chunk in enumerate(chunks[:num_samples]):
                f.write(f"\n{'='*80}\n")
                f.write(f"CHUNK {i+1} (ID: {chunk['chunk_id']})\n")
                f.write(f"Page: {chunk['metadata']['page']}\n")
                f.write(f"Sentences: {chunk['metadata'].get('sentence_count', 'N/A')}\n")
                f.write(f"Characters: {chunk['metadata'].get('char_count', len(chunk['text']))}\n")
                f.write(f"{'='*80}\n")
                f.write(chunk['text'])
                f.write("\n")
        
        logger.info(f"✓ Saved chunks preview to: {output_file}")


def main():
    """Main function for testing enhanced PDF processing"""
    print("\n" + "="*80)
    print("Enhanced PDF Processor - Sentence-Aware Chunking")
    print("="*80 + "\n")
    
    processor = EnhancedPDFProcessor()
    
    try:
        # Process PDF
        chunks = processor.process_pdf()
        
        # Save preview
        processor.save_chunks_preview(chunks, num_samples=10)
        
        # Display sample chunk
        print("\n" + "="*80)
        print("SAMPLE INTELLIGENT CHUNK")
        print("="*80)
        sample = chunks[0]
        print(f"Chunk ID: {sample['chunk_id']}")
        print(f"Page: {sample['metadata']['page']}")
        print(f"Sentences: {sample['metadata'].get('sentence_count', 'N/A')}")
        print(f"Length: {len(sample['text'])} characters")
        print("\nText Preview:")
        print(sample['text'][:500] + "...")
        
        print("\n✓ Enhanced PDF processing complete!")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease ensure the nephrology PDF is placed at:")
        print(f"  {NEPHROLOGY_PDF_PATH}")
        print("\nThe PDF should be named 'nephrology_book.pdf'")


if __name__ == "__main__":
    main()
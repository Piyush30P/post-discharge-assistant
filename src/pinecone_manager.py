"""
Pinecone Manager - Using NEW Pinecone SDK API (v3+)
Handles all Pinecone vector database operations
"""

from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import time
from tqdm import tqdm

from src.config import (
    PINECONE_API_KEY,
    PINECONE_ENVIRONMENT,
    PINECONE_INDEX_NAME,
    EMBEDDING_MODEL
)
from src.utils.logger import get_logger

logger = get_logger()


class PineconeManager:
    """Manage Pinecone vector database operations"""
    
    def __init__(
        self,
        api_key: str = PINECONE_API_KEY,
        environment: str = PINECONE_ENVIRONMENT,
        index_name: str = PINECONE_INDEX_NAME
    ):
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        
        # Initialize Pinecone with NEW API
        logger.info("Initializing Pinecone client...")
        self.pc = Pinecone(api_key=self.api_key)
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        logger.info(f"✓ Embedding model loaded (dimension: {self.embedding_dim})")
        
        # Index will be set later
        self.index = None
    
    def create_index(self, metric: str = "cosine", cloud: str = "aws", region: str = "us-east-1"):
        """Create a new Pinecone index"""
        logger.info(f"Creating Pinecone index: {self.index_name}")
        
        # Check if index exists
        index_list = self.pc.list_indexes()
        existing_names = [idx['name'] for idx in index_list] if isinstance(index_list, list) else index_list.names()
        
        if self.index_name in existing_names:
            logger.warning(f"Index '{self.index_name}' already exists")
            response = input("Do you want to delete and recreate it? (yes/no): ")
            if response.lower() == 'yes':
                self.delete_index()
            else:
                logger.info("Using existing index")
                self.index = self.pc.Index(self.index_name)
                return
        
        # Create index
        logger.info(f"Creating serverless index...")
        logger.info(f"  Dimension: {self.embedding_dim}")
        logger.info(f"  Metric: {metric}")
        logger.info(f"  Cloud: {cloud}")
        logger.info(f"  Region: {region}")
        
        self.pc.create_index(
            name=self.index_name,
            dimension=self.embedding_dim,
            metric=metric,
            spec=ServerlessSpec(
                cloud=cloud,
                region=region
            )
        )
        
        # Wait for index to be ready
        logger.info("Waiting for index to be ready...")
        while True:
            try:
                desc = self.pc.describe_index(self.index_name)
                if desc.status['ready']:
                    break
            except:
                pass
            time.sleep(1)
        
        logger.info(f"✓ Index '{self.index_name}' created successfully")
        self.index = self.pc.Index(self.index_name)
    
    def connect_to_index(self):
        """Connect to existing Pinecone index"""
        logger.info(f"Connecting to index: {self.index_name}")
        
        try:
            self.index = self.pc.Index(self.index_name)
            stats = self.index.describe_index_stats()
            total = stats['total_vector_count']
            
            logger.info(f"✓ Connected to index '{self.index_name}'")
            logger.info(f"  Total vectors: {total}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to index: {e}")
            return False
    
    def delete_index(self):
        """Delete the Pinecone index"""
        logger.warning(f"Deleting index: {self.index_name}")
        try:
            self.pc.delete_index(self.index_name)
            logger.info(f"✓ Index '{self.index_name}' deleted")
            self.index = None
        except Exception as e:
            logger.error(f"Failed to delete index: {e}")
    
    def upsert_documents(self, chunks: List[Dict], batch_size: int = 100, namespace: str = ""):
        """Upload document chunks to Pinecone"""
        if not self.index:
            raise ValueError("Index not initialized. Call create_index() or connect_to_index() first")
        
        logger.info(f"Upserting {len(chunks)} documents to Pinecone...")
        logger.info(f"Batch size: {batch_size}")
        if namespace:
            logger.info(f"Namespace: {namespace}")
        
        for i in tqdm(range(0, len(chunks), batch_size), desc="Uploading to Pinecone"):
            batch = chunks[i:i + batch_size]
            vectors = []
            
            for chunk in batch:
                embedding = self.embedding_model.encode(
                    chunk['text'],
                    convert_to_numpy=True
                ).tolist()
                
                metadata = {
                    'text': chunk['text'][:1000],
                    'page': float(chunk['metadata']['page']),
                    'source': chunk['metadata']['source'],
                    'chunk_id': float(chunk['chunk_id']),
                }
                
                if 'sentence_count' in chunk['metadata']:
                    metadata['sentence_count'] = float(chunk['metadata']['sentence_count'])
                if 'char_count' in chunk['metadata']:
                    metadata['char_count'] = float(chunk['metadata']['char_count'])
                
                vectors.append({
                    'id': f"chunk_{chunk['chunk_id']}",
                    'values': embedding,
                    'metadata': metadata
                })
            
            self.index.upsert(vectors=vectors, namespace=namespace)
        
        time.sleep(2)
        stats = self.index.describe_index_stats()
        total = stats['total_vector_count']
        
        logger.info(f"✓ Successfully upserted documents")
        logger.info(f"  Total vectors in index: {total}")
        if namespace:
            ns_stats = stats.get('namespaces', {}).get(namespace, {})
            logger.info(f"  Vectors in namespace '{namespace}': {ns_stats.get('vector_count', 0)}")
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        namespace: str = "",
        filter: Optional[Dict] = None,
        include_metadata: bool = True
    ) -> List[Dict]:
        """Search for similar documents"""
        if not self.index:
            raise ValueError("Index not initialized")
        
        logger.info(f"Searching Pinecone for: '{query}'")
        logger.info(f"  Top K: {top_k}")
        if namespace:
            logger.info(f"  Namespace: {namespace}")
        if filter:
            logger.info(f"  Filter: {filter}")
        
        query_embedding = self.embedding_model.encode(
            query,
            convert_to_numpy=True
        ).tolist()
        
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=namespace,
            filter=filter,
            include_metadata=include_metadata
        )
        
        formatted_results = []
        for match in results['matches']:
            formatted_results.append({
                'id': match['id'],
                'score': match['score'],
                'metadata': match.get('metadata', {})
            })
        
        logger.info(f"✓ Found {len(formatted_results)} results")
        return formatted_results
    
    def get_index_stats(self) -> Dict:
        """Get statistics about the Pinecone index"""
        if not self.index:
            raise ValueError("Index not initialized")
        
        stats = self.index.describe_index_stats()
        
        return {
            'index_name': self.index_name,
            'dimension': self.embedding_dim,
            'total_vectors': stats['total_vector_count'],
            'namespaces': stats.get('namespaces', {}),
            'index_fullness': stats.get('index_fullness', 0)
        }
    
    def delete_all_vectors(self, namespace: str = ""):
        """Delete all vectors from the index"""
        if not self.index:
            raise ValueError("Index not initialized")
        
        logger.warning(f"Deleting all vectors from index '{self.index_name}'")
        if namespace:
            logger.warning(f"  In namespace: {namespace}")
        
        self.index.delete(delete_all=True, namespace=namespace)
        logger.info("✓ Vectors deleted")


def main():
    """Test Pinecone manager"""
    print("\n" + "="*80)
    print("Pinecone Manager - Connection Test")
    print("="*80 + "\n")
    
    try:
        manager = PineconeManager()
        
        # Connect to existing index
        print("Connecting to existing index...")
        if manager.connect_to_index():
            stats = manager.get_index_stats()
            print(f"\n✓ Successfully connected to Pinecone!")
            print(f"  Index: {stats['index_name']}")
            print(f"  Vectors: {stats['total_vectors']}")
            print(f"  Dimension: {stats['dimension']}")
            
            # Test search
            print("\n✓ Testing search...")
            results = manager.search("chronic kidney disease", top_k=1)
            if results:
                print(f"  Score: {results[0]['score']:.4f}")
                print(f"  Page: {results[0]['metadata'].get('page', 'N/A')}")
        else:
            print("❌ Failed to connect to index")
        
        print("\n" + "="*80)
        print("✓ Test complete!")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
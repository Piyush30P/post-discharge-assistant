"""
Phase 1 Verification Script
Verifies that all Phase 1 components are working correctly
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import DATA_DIR, DATABASE_PATH, PINECONE_INDEX_NAME
from src.utils.logger import get_logger

logger = get_logger()


def print_section(title):
    """Print section header"""
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80)


def check_patient_data():
    """Verify patient data exists"""
    print_section("CHECKING PATIENT DATA")
    
    json_file = DATA_DIR / "patient_reports.json"
    
    if not json_file.exists():
        print(f"❌ Patient data not found: {json_file}")
        return False
    
    import json
    with open(json_file, 'r') as f:
        patients = json.load(f)
    
    print(f"✅ Patient data file exists: {json_file}")
    print(f"✅ Number of patients: {len(patients)}")
    
    if len(patients) >= 25:
        print(f"✅ Meets requirement: 25+ patients")
        return True
    else:
        print(f"❌ Only {len(patients)} patients (need 25+)")
        return False


def check_database():
    """Verify database is populated"""
    print_section("CHECKING DATABASE")
    
    if not DATABASE_PATH.exists():
        print(f"❌ Database not found: {DATABASE_PATH}")
        return False
    
    try:
        from src.database import DatabaseManager
        
        db = DatabaseManager()
        stats = db.get_database_stats()
        
        print(f"✅ Database exists: {DATABASE_PATH}")
        print(f"✅ Total patients: {stats['total_patients']}")
        print(f"✅ Total interactions: {stats['total_interactions']}")
        
        if stats['total_patients'] >= 25:
            print(f"✅ Meets requirement: 25+ patients")
            
            # Show top diagnoses
            if stats['diagnoses_distribution']:
                print(f"\n   Top diagnoses:")
                for diagnosis, count in stats['diagnoses_distribution'][:3]:
                    print(f"     • {diagnosis}: {count}")
            
            return True
        else:
            print(f"❌ Only {stats['total_patients']} patients in database")
            return False
            
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False


def check_pinecone():
    """Verify Pinecone index is set up"""
    print_section("CHECKING PINECONE")
    
    try:
        from src.pinecone_manager import PineconeManager
        
        print("Connecting to Pinecone...")
        manager = PineconeManager()
        
        # Try to connect
        if not manager.connect_to_index():
            print(f"❌ Could not connect to index: {PINECONE_INDEX_NAME}")
            print(f"   Run: python setup_phase1.py")
            return False
        
        # Get stats
        stats = manager.get_index_stats()
        
        print(f"✅ Connected to Pinecone")
        print(f"✅ Index name: {stats['index_name']}")
        print(f"✅ Dimension: {stats['dimension']}")
        print(f"✅ Total vectors: {stats['total_vectors']}")
        print(f"✅ Fullness: {stats['index_fullness']*100:.1f}%")
        
        if stats['total_vectors'] > 100:
            print(f"✅ Index populated (expected ~4000 vectors)")
        else:
            print(f"⚠️  Index has only {stats['total_vectors']} vectors")
            print(f"   Expected ~4000. May need to re-run upload.")
        
        return stats['total_vectors'] > 100
        
    except Exception as e:
        print(f"❌ Pinecone error: {e}")
        print(f"\nPossible issues:")
        print(f"  • PINECONE_API_KEY not set in .env")
        print(f"  • Index not created yet")
        print(f"  • Network connection issue")
        return False


def test_pinecone_search():
    """Test Pinecone search functionality"""
    print_section("TESTING PINECONE SEARCH")
    
    try:
        from src.pinecone_manager import PineconeManager
        
        manager = PineconeManager()
        manager.connect_to_index()
        
        # Test search
        test_query = "chronic kidney disease treatment"
        print(f"Test query: '{test_query}'")
        print("Searching...")
        
        results = manager.search(test_query, top_k=3)
        
        if not results:
            print("❌ No results returned")
            return False
        
        print(f"✅ Search successful! Found {len(results)} results\n")
        
        # Show results
        for i, result in enumerate(results, 1):
            print(f"Result {i}:")
            print(f"  Score: {result['score']:.4f}")
            print(f"  Page: {result['metadata'].get('page', 'N/A')}")
            text = result['metadata'].get('text', 'N/A')
            print(f"  Text: {text[:100]}...")
            print()
        
        # Check quality
        if results[0]['score'] > 0.5:
            print("✅ Search quality: Good (score > 0.5)")
            return True
        else:
            print("⚠️  Search quality: Low scores")
            return False
            
    except Exception as e:
        print(f"❌ Search test failed: {e}")
        return False


def test_embeddings():
    """Test embedding generation"""
    print_section("TESTING EMBEDDINGS")
    
    try:
        from sentence_transformers import SentenceTransformer
        from src.config import EMBEDDING_MODEL
        
        print(f"Loading model: {EMBEDDING_MODEL}")
        model = SentenceTransformer(EMBEDDING_MODEL)
        
        print(f"✅ Model loaded")
        print(f"✅ Dimension: {model.get_sentence_embedding_dimension()}")
        
        # Test embedding
        test_text = "Chronic kidney disease treatment"
        print(f"\nGenerating embedding for: '{test_text}'")
        embedding = model.encode(test_text)
        
        print(f"✅ Embedding generated")
        print(f"   Shape: {embedding.shape}")
        print(f"   Sample values: [{embedding[0]:.4f}, {embedding[1]:.4f}, ...]")
        
        return True
        
    except Exception as e:
        print(f"❌ Embedding test failed: {e}")
        return False


def main():
    """Run all verification checks"""
    print("\n" + "="*80)
    print(" PHASE 1 VERIFICATION")
    print(" Post Discharge Medical AI Assistant")
    print("="*80)
    
    results = {}
    
    # Run all checks
    results['patient_data'] = check_patient_data()
    results['database'] = check_database()
    results['embeddings'] = test_embeddings()
    results['pinecone'] = check_pinecone()
    
    if results['pinecone']:
        results['pinecone_search'] = test_pinecone_search()
    else:
        results['pinecone_search'] = False
    
    # Summary
    print_section("VERIFICATION SUMMARY")
    
    checks = [
        ('patient_data', 'Patient Data (30 records)'),
        ('database', 'SQLite Database'),
        ('embeddings', 'Embedding Model'),
        ('pinecone', 'Pinecone Index'),
        ('pinecone_search', 'Pinecone Search')
    ]
    
    passed = 0
    total = len(checks)
    
    for key, name in checks:
        status = "✅" if results.get(key) else "❌"
        print(f"  {status} {name}")
        if results.get(key):
            passed += 1
    
    print(f"\n{'-'*80}")
    print(f"Passed: {passed}/{total} checks")
    print(f"{'-'*80}")
    
    # Final verdict
    if passed == total:
        print("\n" + "🎉"*40)
        print("\n✅ PHASE 1 VERIFICATION PASSED!")
        print("\nAll components verified:")
        print("  ✅ Patient data generated and in database")
        print("  ✅ Embeddings working")
        print("  ✅ Pinecone index created and populated")
        print("  ✅ Search functionality working")
        
        print("\n🚀 READY FOR PHASE 2!")
        print("\nNext Phase Components:")
        print("  1. MCP Server (tool management)")
        print("  2. Receptionist Agent (patient greeting)")
        print("  3. Clinical Agent (medical queries)")
        print("  4. LangGraph Workflow (orchestration)")
        
        print("\n" + "🎉"*40)
        
    elif passed >= 3:
        print("\n⚠️  PHASE 1 MOSTLY COMPLETE")
        print(f"\nPassed: {passed}/{total} checks")
        
        failed = [name for key, name in checks if not results.get(key)]
        if failed:
            print("\n⚠️  Issues to fix:")
            for item in failed:
                print(f"  ❌ {item}")
        
        if not results.get('pinecone'):
            print("\nTo fix Pinecone:")
            print("  1. Ensure PINECONE_API_KEY is set in .env")
            print("  2. Run: python setup_phase1.py")
            print("  3. Wait for upload to complete")
        
    else:
        print("\n❌ PHASE 1 VERIFICATION FAILED")
        print(f"\nOnly {passed}/{total} checks passed")
        print("\nPlease run: python setup_phase1.py")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
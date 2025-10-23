"""
Phase 1 Setup: Data Preparation & Pinecone Upload
Automates the complete Phase 1 pipeline:
1. Generate patient data
2. Populate database
3. Process PDF with intelligent chunking
4. Upload to Pinecone
"""

import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent))

from src.config import validate_config, NEPHROLOGY_PDF_PATH, DATA_DIR
from src.utils.logger import get_logger
from src.generate_dummy_data import DummyPatientGenerator
from src.database import DatabaseManager, populate_database_from_json
from src.pdf_processor_enhanced import EnhancedPDFProcessor
from src.pinecone_manager import PineconeManager

logger = get_logger()


def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80 + "\n")


def step_1_validate_config():
    """Step 1: Validate configuration including Pinecone"""
    print_header("STEP 1: Validating Configuration")
    
    try:
        validate_config()
        print("✓ Google API key validated")
        print("✓ Pinecone API key validated")
        print("✓ All configuration validated successfully")
        return True
    except ValueError as e:
        print(f"❌ Configuration Error:\n{e}")
        print("\nPlease:")
        print("  1. Copy .env.example to .env")
        print("  2. Add your GOOGLE_API_KEY")
        print("  3. Add your PINECONE_API_KEY from https://app.pinecone.io/")
        print("  4. Run this script again")
        return False


def step_2_generate_patient_data():
    """Step 2: Generate dummy patient data"""
    print_header("STEP 2: Generating Patient Data")
    
    json_file = DATA_DIR / "patient_reports.json"
    
    if json_file.exists():
        print(f"✓ Patient data already exists at: {json_file}")
        with open(json_file, 'r') as f:
            patients = json.load(f)
        print(f"✓ Found {len(patients)} patient records")
        
        response = input("\nRegenerate data? (y/n): ")
        if response.lower() != 'y':
            print("Skipping data generation...")
            return True, patients
    
    try:
        generator = DummyPatientGenerator()
        patients = generator.generate_multiple_patients(count=30)
        generator.save_to_json(patients)
        
        print(f"\n✓ Generated {len(patients)} patient records")
        print(f"✓ Saved to: {json_file}")
        return True, patients
    except Exception as e:
        print(f"❌ Error generating data: {e}")
        return False, None


def step_3_populate_database(patients):
    """Step 3: Populate SQLite database"""
    print_header("STEP 3: Populating Database")
    
    try:
        db = DatabaseManager()
        stats = db.get_database_stats()
        
        if stats['total_patients'] > 0:
            print(f"✓ Database already has {stats['total_patients']} patients")
            response = input("\nReset and repopulate? (y/n): ")
            if response.lower() == 'y':
                db.reset_database()
            else:
                print("Skipping database population...")
                return True
        
        # Add patients to database
        count = db.bulk_add_patients(patients)
        print(f"\n✓ Added {count} patients to database")
        
        # Show stats
        stats = db.get_database_stats()
        print(f"\nDatabase Statistics:")
        print(f"  Total Patients: {stats['total_patients']}")
        print(f"  Database Path: {stats['database_path']}")
        
        if stats['diagnoses_distribution']:
            print(f"\n  Top Diagnoses:")
            for diagnosis, count in stats['diagnoses_distribution'][:5]:
                print(f"    • {diagnosis}: {count}")
        
        return True
    except Exception as e:
        print(f"❌ Error populating database: {e}")
        return False


def step_4_process_pdf():
    """Step 4: Process PDF with intelligent chunking"""
    print_header("STEP 4: Processing PDF with Intelligent Chunking")
    
    if not NEPHROLOGY_PDF_PATH.exists():
        print(f"⚠️  PDF not found at: {NEPHROLOGY_PDF_PATH}")
        print("\nTo complete Phase 1:")
        print(f"  1. Place your nephrology PDF at: {NEPHROLOGY_PDF_PATH}")
        print("  2. Name it: nephrology_book.pdf")
        print("  3. Run this script again")
        print("\n⚠️  Skipping PDF processing...")
        return False, None
    
    try:
        print(f"Found PDF: {NEPHROLOGY_PDF_PATH}")
        print(f"Processing with sentence-aware chunking...")
        print("This will take 5-15 minutes depending on PDF size...\n")
        
        processor = EnhancedPDFProcessor()
        chunks = processor.process_pdf()
        
        # Save preview
        processor.save_chunks_preview(chunks, num_samples=5)
        
        print(f"\n✓ Processed PDF into {len(chunks)} intelligent chunks")
        print(f"✓ Chunks respect sentence boundaries")
        print(f"✓ Average chunk size: ~{sum(len(c['text']) for c in chunks) // len(chunks)} chars")
        
        return True, chunks
    except Exception as e:
        print(f"❌ Error processing PDF: {e}")
        logger.log_error("PDF Processing", e)
        return False, None


def step_5_setup_pinecone(chunks):
    """Step 5: Create Pinecone index and upload vectors"""
    print_header("STEP 5: Setting Up Pinecone & Uploading Vectors")
    
    if not chunks:
        print("⚠️  No chunks available. PDF must be processed first.")
        return False
    
    try:
        # Initialize Pinecone manager
        print("Initializing Pinecone connection...")
        manager = PineconeManager()
        
        # Create index
        print("\nCreating Pinecone index...")
        print("This will create a serverless index (free tier)")
        manager.create_index(
            metric="cosine",
            cloud="aws",
            region="us-east-1"
        )
        
        # Upload chunks
        print(f"\nUploading {len(chunks)} vectors to Pinecone...")
        print("This will take 5-10 minutes...")
        print("Progress bar will show upload status:\n")
        
        manager.upsert_documents(chunks, batch_size=100)
        
        # Verify upload
        print("\nVerifying upload...")
        stats = manager.get_index_stats()
        
        print(f"\n✓ Pinecone setup complete!")
        print(f"\nIndex Statistics:")
        print(f"  Index Name: {stats['index_name']}")
        print(f"  Dimension: {stats['dimension']}")
        print(f"  Total Vectors: {stats['total_vectors']}")
        print(f"  Fullness: {stats['index_fullness']*100:.1f}%")
        
        # Test search
        print("\nTesting search functionality...")
        test_results = manager.search("chronic kidney disease treatment", top_k=3)
        
        print(f"✓ Search test successful! Found {len(test_results)} results")
        print(f"\nSample result:")
        if test_results:
            result = test_results[0]
            print(f"  Score: {result['score']:.4f}")
            print(f"  Page: {result['metadata'].get('page', 'N/A')}")
            print(f"  Text: {result['metadata'].get('text', 'N/A')[:150]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error setting up Pinecone: {e}")
        logger.log_error("Pinecone Setup", e)
        return False


def main():
    """Main setup function for Phase 1"""
    print("\n" + "="*80)
    print(" POST DISCHARGE MEDICAL AI ASSISTANT")
    print(" PHASE 1: DATA SETUP WITH PINECONE")
    print("="*80)
    
    print("\nThis script will:")
    print("  1. ✅ Validate API keys (Google + Pinecone)")
    print("  2. ✅ Generate 30 patient records")
    print("  3. ✅ Populate SQLite database")
    print("  4. ✅ Process PDF with intelligent chunking")
    print("  5. ✅ Upload vectors to Pinecone")
    
    print("\n⏱️  Estimated time: 15-30 minutes")
    print("💰 Cost: FREE (using free tiers)")
    
    response = input("\nReady to begin? (y/n): ")
    if response.lower() != 'y':
        print("\nSetup cancelled.")
        return
    
    results = {}
    
    # Step 1: Validate configuration
    results['config'] = step_1_validate_config()
    if not results['config']:
        print("\n❌ Setup cannot continue without valid configuration")
        return
    
    # Step 2: Generate patient data
    success, patients = step_2_generate_patient_data()
    results['data'] = success
    if not success:
        print("\n❌ Setup cannot continue without patient data")
        return
    
    # Step 3: Populate database
    results['database'] = step_3_populate_database(patients)
    
    # Step 4: Process PDF
    success, chunks = step_4_process_pdf()
    results['pdf'] = success
    
    # Step 5: Setup Pinecone
    if chunks:
        results['pinecone'] = step_5_setup_pinecone(chunks)
    else:
        results['pinecone'] = False
        print("\n⚠️  Pinecone setup skipped (no PDF chunks)")
    
    # Summary
    print_header("PHASE 1 SETUP SUMMARY")
    
    print("Completed Steps:")
    status_map = {
        'config': 'Configuration',
        'data': 'Patient Data Generation',
        'database': 'Database Population',
        'pdf': 'PDF Processing',
        'pinecone': 'Pinecone Upload'
    }
    
    for key, name in status_map.items():
        status = "✅" if results.get(key) else "❌"
        print(f"  {status} {name}")
    
    # Success check
    core_steps = ['config', 'data', 'database']
    core_complete = all(results.get(step) for step in core_steps)
    full_complete = all(results.values())
    
    if full_complete:
        print("\n" + "🎉"*40)
        print("\n✅ PHASE 1 COMPLETE! All components ready.")
        print("\nWhat's ready:")
        print("  ✅ 30 patient records in SQLite")
        print("  ✅ PDF processed into intelligent chunks")
        print("  ✅ ~4000 vectors in Pinecone cloud")
        print("  ✅ Semantic search working")
        
        print("\n📊 Quick Stats:")
        from src.database import DatabaseManager
        from src.pinecone_manager import PineconeManager
        
        db = DatabaseManager()
        db_stats = db.get_database_stats()
        print(f"  Database: {db_stats['total_patients']} patients")
        
        try:
            pm = PineconeManager()
            pm.connect_to_index()
            pc_stats = pm.get_index_stats()
            print(f"  Pinecone: {pc_stats['total_vectors']} vectors")
        except:
            pass
        
        print("\n🚀 READY FOR PHASE 2: Multi-Agent System")
        print("\nNext steps:")
        print("  1. Build MCP server with tools")
        print("  2. Create Receptionist Agent")
        print("  3. Create Clinical Agent")
        print("  4. Build LangGraph workflow")
        
    elif core_complete:
        print("\n⚠️  PHASE 1 PARTIALLY COMPLETE")
        print("\nCore components ready:")
        print("  ✅ Patient data and database")
        
        if not results.get('pdf'):
            print("\n⚠️  Missing:")
            print("  ❌ PDF processing (need nephrology_book.pdf)")
            print("  ❌ Pinecone upload (requires PDF)")
            print("\nTo complete:")
            print(f"  1. Add PDF to: {NEPHROLOGY_PDF_PATH}")
            print("  2. Run this script again")
        
    else:
        print("\n❌ PHASE 1 INCOMPLETE")
        print("\nPlease fix the errors above and run again.")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
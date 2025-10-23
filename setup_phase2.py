"""
Phase 2 Setup: Multi-Agent System Installation
Installs dependencies and verifies Phase 2 components
"""

import sys
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.logger import get_logger

logger = get_logger()


def print_header(title):
    """Print formatted header"""
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80 + "\n")


def check_package_installed(package_name):
    """Check if a package is already installed"""
    try:
        # Get package name without version
        pkg = package_name.split('==')[0].split('>=')[0].split('<=')[0]
        __import__(pkg.replace('-', '_'))
        return True
    except ImportError:
        return False


def step_1_install_dependencies():
    """Install Phase 2 dependencies (skip if already installed)"""
    print_header("STEP 1: Checking Phase 2 Dependencies")
    
    packages = [
        ("langchain", "langchain==0.3.18"),
        ("langchain_google_genai", "langchain-google-genai==2.0.8"),
        ("langchain_core", "langchain-core==0.3.29"),
        ("langgraph", "langgraph==0.2.62"),
        ("langgraph.checkpoint", "langgraph-checkpoint==2.0.13"),
        ("duckduckgo_search", "duckduckgo-search==7.1.0"),
        ("streamlit", "streamlit==1.41.1")
    ]
    
    to_install = []
    
    # Check which packages need installation
    print("Checking installed packages...\n")
    for import_name, package_spec in packages:
        try:
            __import__(import_name)
            print(f"✓ {package_spec.split('==')[0]} (already installed)")
        except ImportError:
            print(f"⚠️  {package_spec.split('==')[0]} (needs installation)")
            to_install.append(package_spec)
    
    if not to_install:
        print("\n✓ All dependencies already installed")
        return True
    
    # Install missing packages
    print(f"\nInstalling {len(to_install)} missing package(s)...\n")
    
    try:
        for package in to_install:
            print(f"Installing {package}...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", package, "-q"],
                check=True,
                capture_output=True
            )
        
        print("\n✓ All dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Installation failed: {e}")
        print("Try running: pip install -r requirements.txt")
        return False


def step_2_verify_imports():
    """Verify all imports work"""
    print_header("STEP 2: Verifying Imports")
    
    imports_to_test = [
        ("langchain", "LangChain"),
        ("langchain_google_genai", "LangChain Google AI"),
        ("langgraph", "LangGraph"),
        ("duckduckgo_search", "DuckDuckGo Search"),
        ("streamlit", "Streamlit")
    ]
    
    all_passed = True
    
    for module, name in imports_to_test:
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError as e:
            print(f"❌ {name}: {e}")
            all_passed = False
    
    if all_passed:
        print("\n✓ All imports successful")
    
    return all_passed


def step_3_verify_phase1():
    """Verify Phase 1 is complete"""
    print_header("STEP 3: Verifying Phase 1 Prerequisites")
    
    try:
        from src.database import DatabaseManager
        from src.pinecone_manager import PineconeManager
        
        # Check database
        db = DatabaseManager()
        stats = db.get_database_stats()
        print(f"✓ Database: {stats['total_patients']} patients")
        
        # Check Pinecone
        pm = PineconeManager()
        if pm.connect_to_index():
            pc_stats = pm.get_index_stats()
            print(f"✓ Pinecone: {pc_stats['total_vectors']} vectors")
            
            if pc_stats['total_vectors'] == 0:
                print("\n⚠️  Warning: Pinecone has 0 vectors!")
                print("   Run: python setup_phase1.py to populate")
                return False
        else:
            print("❌ Pinecone connection failed")
            return False
        
        print("\n✓ Phase 1 prerequisites met")
        return True
        
    except Exception as e:
        print(f"❌ Phase 1 verification failed: {e}")
        print("\nPlease complete Phase 1 first:")
        print("  python setup_phase1.py")
        return False


def step_4_test_agents():
    """Test agent initialization"""
    print_header("STEP 4: Testing Agent System")
    
    try:
        print("Testing Receptionist Agent...")
        from src.agents.receptionist_agent import ReceptionistAgent
        receptionist = ReceptionistAgent()
        print("✓ Receptionist Agent initialized")
        
        print("\nTesting Clinical Agent...")
        from src.agents.clinical_agent import ClinicalAgent
        clinical = ClinicalAgent()
        print("✓ Clinical Agent initialized")
        
        print("\nTesting Workflow...")
        from src.workflow.graph import MultiAgentWorkflow
        workflow = MultiAgentWorkflow()
        print("✓ Workflow initialized")
        
        print("\n✓ All agents working")
        return True
        
    except Exception as e:
        print(f"\n❌ Agent test failed: {e}")
        logger.log_error("Phase2Setup", e)
        
        # Give specific error hints
        if "List" in str(e) or "Dict" in str(e):
            print("\n💡 Fix: Add missing typing imports to src/workflow/graph.py")
            print("   Change line 5 from:")
            print("   from typing import Dict, Literal")
            print("   To:")
            print("   from typing import Dict, List, Literal")
        
        return False


def main():
    """Main setup function"""
    print("\n" + "="*80)
    print(" POST DISCHARGE MEDICAL AI ASSISTANT")
    print(" PHASE 2: MULTI-AGENT SYSTEM SETUP")
    print("="*80)
    
    print("\nThis script will:")
    print("  1. ✅ Check Phase 2 dependencies (skip if installed)")
    print("  2. ✅ Verify imports")
    print("  3. ✅ Check Phase 1 prerequisites")
    print("  4. ✅ Test agent system")
    
    print("\n⏱️  Estimated time: 1-3 minutes (faster if packages installed)")
    
    response = input("\nReady to begin? (y/n): ")
    if response.lower() != 'y':
        print("\nSetup cancelled.")
        return
    
    results = {}
    
    # Step 1: Check/Install dependencies
    results['dependencies'] = step_1_install_dependencies()
    if not results['dependencies']:
        print("\n❌ Cannot continue without dependencies")
        return
    
    # Step 2: Verify imports
    results['imports'] = step_2_verify_imports()
    
    # Step 3: Verify Phase 1
    results['phase1'] = step_3_verify_phase1()
    if not results['phase1']:
        print("\n❌ Please complete Phase 1 first")
        return
    
    # Step 4: Test agents
    results['agents'] = step_4_test_agents()
    
    # Summary
    print_header("PHASE 2 SETUP SUMMARY")
    
    status_map = {
        'dependencies': 'Dependencies Check',
        'imports': 'Import Verification',
        'phase1': 'Phase 1 Prerequisites',
        'agents': 'Agent System Test'
    }
    
    for key, name in status_map.items():
        status = "✅" if results.get(key) else "❌"
        print(f"  {status} {name}")
    
    if all(results.values()):
        print("\n" + "🎉"*40)
        print("\n✅ PHASE 2 SETUP COMPLETE!")
        
        print("\n📊 What's ready:")
        print("  ✅ Receptionist Agent (patient greeting)")
        print("  ✅ Clinical Agent (medical queries)")
        print("  ✅ MCP Tools (database, RAG, web search)")
        print("  ✅ LangGraph Workflow (orchestration)")
        print("  ✅ Streamlit UI (ready to launch)")
        
        print("\n🚀 READY TO RUN!")
        print("\nLaunch the assistant:")
        print("  streamlit run src/app.py")
        print("\nOr use the CLI:")
        print("  python run_assistant.py")
        
    else:
        print("\n⚠️  PHASE 2 SETUP INCOMPLETE")
        print("\nPlease fix the errors above and run again.")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
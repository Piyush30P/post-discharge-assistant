"""
Phase 2 Verification
Comprehensive testing of multi-agent system
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.logger import get_logger

logger = get_logger()


def print_header(title):
    """Print formatted header"""
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80 + "\n")


def test_imports():
    """Test Phase 2 imports"""
    print_header("TESTING IMPORTS")
    
    try:
        from src.agents.receptionist_agent import ReceptionistAgent
        from src.agents.clinical_agent import ClinicalAgent
        from src.workflow.graph import MultiAgentWorkflow
        from src.mcp.tools import MCPTools
        print("✅ All Phase 2 modules import successfully")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False


def test_tools():
    """Test MCP tools"""
    print_header("TESTING MCP TOOLS")
    
    try:
        from src.mcp.tools import MCPTools
        tools = MCPTools()
        
        # Test patient data retrieval
        print("Test 1: Patient data retrieval...")
        result = tools.get_patient_data("John Brown")
        if result.get("success"):
            print(f"✅ Retrieved patient: {result.get('patient_name')}")
        else:
            print(f"⚠️  Patient not found (this is okay if John Brown isn't in your DB)")
        
        # Test knowledge search
        print("\nTest 2: Medical knowledge search...")
        result = tools.search_medical_knowledge("kidney disease")
        if result.get("success") and result.get("num_results", 0) > 0:
            print(f"✅ Found {result['num_results']} relevant chunks")
            print(f"   Top score: {result['results'][0]['relevance_score']}")
        else:
            print("❌ Knowledge search failed")
            return False
        
        print("\n✅ All tools working")
        return True
        
    except Exception as e:
        print(f"❌ Tool test failed: {e}")
        return False


def test_receptionist():
    """Test receptionist agent"""
    print_header("TESTING RECEPTIONIST AGENT")
    
    try:
        from src.agents.receptionist_agent import ReceptionistAgent
        
        print("Initializing agent...")
        agent = ReceptionistAgent()
        
        print("Testing greeting...")
        response = agent.process("Hello, I was recently discharged")
        
        if response and response.get("message"):
            print(f"✅ Agent response: {response['message'][:100]}...")
            return True
        else:
            print("❌ No response from agent")
            return False
            
    except Exception as e:
        print(f"❌ Receptionist test failed: {e}")
        return False


def test_clinical():
    """Test clinical agent"""
    print_header("TESTING CLINICAL AGENT")
    
    try:
        from src.agents.clinical_agent import ClinicalAgent
        
        print("Initializing agent...")
        agent = ClinicalAgent()
        
        print("Testing medical query...")
        response = agent.process("What is chronic kidney disease?")
        
        if response and response.get("message"):
            print(f"✅ Agent response: {response['message'][:100]}...")
            print(f"   Sources used: {len(response.get('sources_used', []))}")
            return True
        else:
            print("❌ No response from agent")
            return False
            
    except Exception as e:
        print(f"❌ Clinical test failed: {e}")
        return False


def test_workflow():
    """Test complete workflow"""
    print_header("TESTING WORKFLOW")
    
    try:
        from src.workflow.graph import MultiAgentWorkflow
        
        print("Initializing workflow...")
        workflow = MultiAgentWorkflow()
        
        print("Testing conversation flow...")
        response = workflow.process_message(
            "Hello, I need help with my discharge instructions",
            "test-verify"
        )
        
        if response and len(response) > 0:
            print(f"✅ Workflow response: {response[:100]}...")
            return True
        else:
            print("❌ No response from workflow")
            return False
            
    except Exception as e:
        print(f"❌ Workflow test failed: {e}")
        return False


def main():
    """Run all verification tests"""
    print("\n" + "="*80)
    print(" PHASE 2 VERIFICATION")
    print(" Post Discharge Medical AI Assistant")
    print("="*80)
    
    results = {}
    
    # Run tests
    results['imports'] = test_imports()
    results['tools'] = test_tools()
    results['receptionist'] = test_receptionist()
    results['clinical'] = test_clinical()
    results['workflow'] = test_workflow()
    
    # Summary
    print_header("VERIFICATION SUMMARY")
    
    status_map = {
        'imports': 'Module Imports',
        'tools': 'MCP Tools',
        'receptionist': 'Receptionist Agent',
        'clinical': 'Clinical Agent',
        'workflow': 'Complete Workflow'
    }
    
    for key, name in status_map.items():
        status = "✅" if results.get(key) else "❌"
        print(f"  {status} {name}")
    
    print("\n" + "-"*80)
    passed = sum(results.values())
    total = len(results)
    print(f"Passed: {passed}/{total} checks")
    print("-"*80)
    
    if all(results.values()):
        print("\n" + "🎉"*40)
        print("\n✅ PHASE 2 VERIFICATION PASSED!")
        print("\n🚀 Ready to launch:")
        print("   streamlit run src/app.py")
    else:
        print("\n⚠️  PHASE 2 VERIFICATION INCOMPLETE")
        print("\nPlease fix the failing tests above.")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
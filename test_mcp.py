"""
MCP Integration Test Script
Tests the MCP server and client functionality
"""

import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.mcp.client import MCPClient
from src.utils.logger import get_logger

logger = get_logger()


async def test_mcp_integration():
    """Test MCP server and client integration"""
    print("🧪 Testing MCP Integration for Post-Discharge Assistant")
    print("=" * 60)
    
    client = MCPClient()
    
    try:
        # Test 1: List available resources
        print("\n1️⃣ Testing: List Resources")
        resources = await client.list_resources()
        print(f"✅ Found {len(resources)} resources:")
        for resource in resources:
            print(f"   - {resource['name']}: {resource['description']}")
        
        # Test 2: List available tools
        print("\n2️⃣ Testing: List Tools")
        tools = await client.list_tools()
        print(f"✅ Found {len(tools)} tools:")
        for tool in tools:
            print(f"   - {tool['name']}: {tool['description']}")
        
        # Test 3: Test patient data retrieval
        print("\n3️⃣ Testing: Patient Data Retrieval")
        patient_result = await client.get_patient_data("Ashley King")
        if patient_result.get("success"):
            print("✅ Patient data retrieved successfully")
            print(f"   - Patient: {patient_result.get('patient_name')}")
            print(f"   - Diagnosis: {patient_result.get('diagnosis')}")
        else:
            print(f"❌ Patient data retrieval failed: {patient_result.get('error')}")
        
        # Test 4: Test medical knowledge search
        print("\n4️⃣ Testing: Medical Knowledge Search")
        knowledge_result = await client.search_medical_knowledge("kidney disease symptoms", top_k=3)
        if knowledge_result.get("success"):
            print("✅ Medical knowledge search successful")
            print(f"   - Found {knowledge_result.get('results_count', 0)} results")
        else:
            print(f"❌ Medical knowledge search failed: {knowledge_result.get('error')}")
        
        # Test 5: Test web search
        print("\n5️⃣ Testing: Web Search")
        web_result = await client.web_search_medical("latest kidney disease treatment 2024", max_results=2)
        if web_result.get("success"):
            print("✅ Web search successful")
            print(f"   - Found {len(web_result.get('results', []))} results")
        else:
            print(f"❌ Web search failed: {web_result.get('error')}")
        
        # Test 6: Test list patients
        print("\n6️⃣ Testing: List All Patients")
        patients_result = await client.list_patients()
        if patients_result.get("success"):
            print("✅ Patient list retrieved successfully")
            print(f"   - Total patients: {patients_result.get('total_patients')}")
        else:
            print(f"❌ Patient list failed: {patients_result.get('error')}")
        
        # Test 7: Test drug interactions
        print("\n7️⃣ Testing: Drug Interactions Check")
        interaction_result = await client.check_drug_interactions(["Tamsulosin", "Ketorolac"])
        if interaction_result.get("success"):
            print("✅ Drug interaction check successful")
            print(f"   - Checked medications: {interaction_result.get('medications_checked')}")
        else:
            print(f"❌ Drug interaction check failed: {interaction_result.get('error')}")
        
        print("\n" + "=" * 60)
        print("🎉 MCP Integration Test Complete!")
        
    except Exception as e:
        print(f"\n❌ MCP Integration Test Failed: {e}")
        logger.log_error("test_mcp_integration", e)
    
    finally:
        await client.close()


def test_mcp_tools_fallback():
    """Test MCP tools fallback functionality"""
    print("\n🔄 Testing MCP Tools Fallback Mode")
    print("=" * 40)
    
    try:
        # Import with MCP disabled to test fallback
        from src.mcp.tools import MCPTools
        
        # Initialize without MCP
        tools = MCPTools(use_mcp=False)
        
        # Test patient data retrieval (direct mode)
        print("\n1️⃣ Testing: Patient Data (Direct Mode)")
        result = tools.get_patient_data("Ashley King")
        if result.get("success"):
            print("✅ Direct patient data retrieval successful")
        else:
            print(f"❌ Direct patient data retrieval failed: {result.get('error')}")
        
        # Test medical knowledge search (direct mode)
        print("\n2️⃣ Testing: Medical Knowledge (Direct Mode)")
        result = tools.search_medical_knowledge("kidney disease", top_k=2)
        if result.get("success"):
            print("✅ Direct medical knowledge search successful")
        else:
            print(f"❌ Direct medical knowledge search failed: {result.get('error')}")
        
        # Test web search (direct mode)
        print("\n3️⃣ Testing: Web Search (Direct Mode)")
        result = tools.web_search("kidney disease treatment", max_results=2)
        if result.get("success"):
            print("✅ Direct web search successful")
        else:
            print(f"❌ Direct web search failed: {result.get('error')}")
        
        print("\n✅ Fallback Mode Test Complete!")
        
    except Exception as e:
        print(f"\n❌ Fallback Mode Test Failed: {e}")
        logger.log_error("test_mcp_tools_fallback", e)


async def main():
    """Main test function"""
    print("🏥 Post-Discharge Assistant MCP Test Suite")
    print("=" * 50)
    
    # Test MCP integration
    await test_mcp_integration()
    
    # Test fallback mode
    test_mcp_tools_fallback()
    
    print("\n🏁 All tests completed!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️ Tests interrupted by user")
    except Exception as e:
        print(f"\n💥 Test suite failed: {e}")
        sys.exit(1)
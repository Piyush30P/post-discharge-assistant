"""
Test web search functionality
"""

from src.workflow.graph import MultiAgentWorkflow

def test_web_search():
    print("\n" + "="*80)
    print("TESTING WEB SEARCH FUNCTIONALITY")
    print("="*80 + "\n")
    
    workflow = MultiAgentWorkflow()
    
    # Test queries that should trigger web search
    test_queries = [
        "Recent news on kidney disease",
        "What's the latest research on SGLT2 inhibitors?",
        "Current guidelines for CKD 2025",
        "New treatments for kidney disease"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}: {query}")
        print('='*80)
        
        response = workflow.process_message(query, f"test-web-{i}")
        
        # Check for clinical agent
        if "**Clinical Agent:**" in response:
            print("✅ Routed to Clinical Agent")
        else:
            print("❌ Wrong agent")
        
        # Check for web search indicators
        if any(indicator in response.lower() for indicator in ["recent", "search", "according to", "source", "2024", "2025"]):
            print("✅ Web search results present")
        else:
            print("❌ No web search results")
        
        print(f"\nResponse preview: {response[:300]}...")
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)

if __name__ == "__main__":
    test_web_search()
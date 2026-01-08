"""
Test Query Transformation System
Demonstrates advanced query transformation capabilities
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.query_transformer import QueryTransformer
from src.utils.logger import get_logger

logger = get_logger()


def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80)


def print_section(text):
    """Print a section header"""
    print(f"\n{text}")
    print("-" * len(text))


def test_all_modes():
    """Test all query transformation modes"""
    
    print_header("ADVANCED QUERY TRANSFORMATION TEST")
    
    transformer = QueryTransformer()
    
    # Test queries
    test_queries = [
        {
            "query": "What are my medications?",
            "type": "Simple query",
            "expected": "Rewrite"
        },
        {
            "query": "What medications should I take, what are the side effects, and when should I follow up with my doctor?",
            "type": "Complex query",
            "expected": "Decomposition"
        },
        {
            "query": "When do I need to go to ER?",
            "type": "Medical abbreviation",
            "expected": "Rewrite"
        },
        {
            "query": "What should I eat and what should I avoid in my diet and also when should I take my medicines?",
            "type": "Multi-part question",
            "expected": "Decomposition"
        }
    ]
    
    for i, test in enumerate(test_queries, 1):
        print_section(f"Test {i}: {test['type']}")
        print(f"Original Query: \"{test['query']}\"")
        print(f"Expected Strategy: {test['expected']}\n")
        
        # Test AUTO mode
        result = transformer.transform_query(test["query"], mode="auto")
        print(f"✓ Selected Strategy: {result['recommended_strategy'].upper()}")
        
        if result['recommended_strategy'] == "decomposition":
            decomp = result['transformations']['decomposition']
            if decomp.get('needs_decomposition'):
                print(f"\n📋 Decomposed into {len(decomp['sub_queries'])} sub-queries:")
                for j, sq in enumerate(decomp['sub_queries'], 1):
                    print(f"   {j}. {sq}")
                print(f"\n💡 Reasoning: {decomp.get('reasoning', 'N/A')}")
            else:
                print("   No decomposition needed")
                
        elif result['recommended_strategy'] == "rewrite":
            rewrite = result['transformations']['rewrite']
            print(f"\n✍️  Rewritten Query: \"{rewrite['rewritten_query']}\"")
            if rewrite.get('improvements'):
                print(f"\n📈 Improvements:")
                for imp in rewrite['improvements']:
                    print(f"   • {imp}")
            print(f"\n💡 Reasoning: {rewrite.get('reasoning', 'N/A')}")
        
        print()


def test_individual_modes():
    """Test each transformation mode individually"""
    
    print_header("INDIVIDUAL MODE TESTING")
    
    transformer = QueryTransformer()
    query = "What are the side effects of my medication and when should I take it?"
    
    print(f"Test Query: \"{query}\"\n")
    
    # Test 1: Decomposition
    print_section("1. DECOMPOSITION MODE")
    result = transformer.decompose_query(query)
    if result['success'] and result['needs_decomposition']:
        print(f"✓ Decomposed into {len(result['sub_queries'])} sub-queries:")
        for i, sq in enumerate(result['sub_queries'], 1):
            print(f"   {i}. {sq}")
    
    # Test 2: Multi-Query
    print_section("2. MULTI-QUERY MODE")
    result = transformer.generate_multi_queries(query)
    if result['success']:
        print(f"✓ Generated {len(result['variations'])} variations:")
        for i, var in enumerate(result['variations'], 1):
            print(f"   {i}. {var}")
    
    # Test 3: Rewrite
    print_section("3. REWRITE MODE")
    result = transformer.rewrite_query(query)
    if result['success']:
        print(f"✓ Rewritten: \"{result['rewritten_query']}\"")
        if result.get('improvements'):
            print("\nImprovements made:")
            for imp in result['improvements']:
                print(f"   • {imp}")
    
    # Test 4: All modes combined
    print_section("4. ALL MODES COMBINED")
    result = transformer.transform_query(query, mode="all")
    print("✓ Applied all transformations:")
    print(f"   - Decomposition: {len(result['transformations']['decomposition']['sub_queries'])} sub-queries")
    print(f"   - Multi-Query: {len(result['transformations']['multi_query']['variations'])} variations")
    print(f"   - Rewrite: Query optimized")


def test_edge_cases():
    """Test edge cases and error handling"""
    
    print_header("EDGE CASE TESTING")
    
    transformer = QueryTransformer()
    
    edge_cases = [
        ("", "Empty query"),
        ("Hi", "Single word"),
        ("?" * 5, "Only punctuation"),
        ("a" * 200, "Very long query"),
        ("What? How? Why?", "Multiple questions")
    ]
    
    for query, description in edge_cases:
        print_section(description)
        print(f"Query: \"{query[:50]}{'...' if len(query) > 50 else ''}\"")
        
        try:
            result = transformer.transform_query(query, mode="auto")
            if result.get('error'):
                print(f"⚠️  Error handled gracefully: {result['error']}")
            else:
                print(f"✓ Processed successfully ({result['recommended_strategy']})")
        except Exception as e:
            print(f"❌ Exception: {e}")


def compare_strategies():
    """Compare different strategies on the same query"""
    
    print_header("STRATEGY COMPARISON")
    
    transformer = QueryTransformer()
    query = "Tell me about my kidney disease treatment and diet restrictions"
    
    print(f"Query: \"{query}\"\n")
    
    strategies = ["decompose", "multi", "rewrite"]
    
    for strategy in strategies:
        print_section(f"{strategy.upper()} Strategy")
        result = transformer.transform_query(query, mode=strategy)
        
        if strategy == "decompose" and "decomposition" in result['transformations']:
            decomp = result['transformations']['decomposition']
            if decomp.get('needs_decomposition'):
                for i, sq in enumerate(decomp['sub_queries'], 1):
                    print(f"   {i}. {sq}")
            else:
                print("   No decomposition needed")
                
        elif strategy == "multi" and "multi_query" in result['transformations']:
            multi = result['transformations']['multi_query']
            for i, var in enumerate(multi['variations'], 1):
                print(f"   {i}. {var}")
                
        elif strategy == "rewrite" and "rewrite" in result['transformations']:
            rewrite = result['transformations']['rewrite']
            print(f"   → {rewrite['rewritten_query']}")


def main():
    """Run all tests"""
    try:
        # Run all test suites
        test_all_modes()
        test_individual_modes()
        compare_strategies()
        test_edge_cases()
        
        print_header("✅ ALL TESTS COMPLETED SUCCESSFULLY")
        
    except Exception as e:
        logger.log_error("test_query_transform.main", e)
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

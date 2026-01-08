"""
Advanced Query Transformation Module
Handles query decomposition, multi-query generation, and query rewriting
Uses open source models (Llama/Mistral) via Ollama for rate limit avoidance
"""

from typing import List, Dict, Optional
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from src.utils.logger import get_logger

logger = get_logger()


class QueryTransformer:
    """
    Advanced query transformation system that:
    1. Decomposes complex queries into sub-queries
    2. Generates multiple query variations
    3. Rewrites queries for better retrieval
    
    Uses Ollama with Llama/Mistral for unlimited local processing
    """
    
    def __init__(self, model: str = "llama3.2"):
        """
        Initialize QueryTransformer with Ollama model
        
        Args:
            model: Ollama model name (default: llama3.2)
                   Alternatives: mistral, llama3.1, phi3
        """
        try:
            self.llm = ChatOllama(
                model=model,
                temperature=0.3,
                base_url="http://localhost:11434"
            )
            logger.info(f"✓ QueryTransformer initialized with Ollama model: {model}")
        except Exception as e:
            logger.warning(f"Failed to initialize Ollama: {e}")
            logger.warning("Query transformation will use fallback mode")
            self.llm = None
        
        # Query decomposition prompt
        self.decomposition_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at breaking down complex medical queries into simpler sub-queries.
            
Your task is to decompose a complex query into 2-4 simpler, focused sub-queries that together answer the original question.

Rules:
1. Each sub-query should be self-contained and answerable independently
2. Sub-queries should cover different aspects of the original question
3. For medical queries, consider: symptoms, medications, diet, follow-up care, warning signs
4. Keep sub-queries clear and specific
5. Return ONLY valid JSON, no additional text

Output format:
{{
    "needs_decomposition": true/false,
    "sub_queries": ["query1", "query2", ...],
    "reasoning": "brief explanation"
}}"""),
            ("human", "Original query: {query}\n\nDecompose this query:")
        ])
        
        # Multi-query generation prompt
        self.multi_query_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at generating multiple variations of a query to improve retrieval.

Your task is to generate 3-5 alternative phrasings of the same query that might retrieve different relevant information.

Rules:
1. Maintain the same intent and meaning
2. Use different medical terminology, synonyms, or perspectives
3. Vary question structure (what/how/why/when)
4. Consider both patient and clinical language
5. Return ONLY valid JSON, no additional text

Output format:
{{
    "original_query": "the query",
    "variations": ["variation1", "variation2", ...],
    "reasoning": "brief explanation"
}}"""),
            ("human", "Original query: {query}\n\nGenerate query variations:")
        ])
        
        # Query rewriting prompt
        self.rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at rewriting queries for optimal medical information retrieval.

Your task is to rewrite the query to be more specific, medically accurate, and retrieval-friendly.

Rules:
1. Add relevant medical context if implicit
2. Use proper medical terminology
3. Expand abbreviations
4. Make questions more specific
5. Remove unnecessary words
6. Return ONLY valid JSON, no additional text

Output format:
{{
    "original_query": "the query",
    "rewritten_query": "improved query",
    "improvements": ["improvement1", "improvement2", ...],
    "reasoning": "brief explanation"
}}"""),
            ("human", "Original query: {query}\n\nRewrite this query:")
        ])
        
        self.parser = JsonOutputParser()
        logger.info("✓ Query Transformer initialized")
    
    def decompose_query(self, query: str) -> Dict:
        """
        Decompose a complex query into simpler sub-queries
        
        Args:
            query: Complex user query
            
        Returns:
            Dictionary with sub-queries and metadata
        """
        try:
            logger.info(f"[QUERY_TRANSFORM] Decomposing: {query[:50]}...")
            
            # Fallback if Ollama not available
            if self.llm is None:
                return self._fallback_decompose(query)
            
            chain = self.decomposition_prompt | self.llm
            response = chain.invoke({"query": query})
            
            # Parse JSON response
            result = self._parse_json_response(response.content)
            
            if result and result.get("needs_decomposition"):
                logger.info(f"[QUERY_TRANSFORM] Decomposed into {len(result['sub_queries'])} sub-queries")
                return {
                    "success": True,
                    "needs_decomposition": True,
                    "sub_queries": result["sub_queries"],
                    "reasoning": result.get("reasoning", ""),
                    "original_query": query
                }
            else:
                logger.info("[QUERY_TRANSFORM] Query doesn't need decomposition")
                return {
                    "success": True,
                    "needs_decomposition": False,
                    "sub_queries": [query],
                    "reasoning": result.get("reasoning", "Simple query"),
                    "original_query": query
                }
                
        except Exception as e:
            logger.log_error("QueryTransformer.decompose_query", e)
            return {
                "success": False,
                "needs_decomposition": False,
                "sub_queries": [query],
                "error": str(e),
                "original_query": query
            }
    
    def generate_multi_queries(self, query: str) -> Dict:
        """
        Generate multiple variations of a query
        
        Args:
            query: Original query
            
        Returns:
            Dictionary with query variations
        """
        try:
            logger.info(f"[QUERY_TRANSFORM] Generating variations: {query[:50]}...")
            
            # Fallback if Ollama not available
            if self.llm is None:
                return self._fallback_multi_query(query)
            
            chain = self.multi_query_prompt | self.llm
            response = chain.invoke({"query": query})
            
            # Parse JSON response
            result = self._parse_json_response(response.content)
            
            if result and "variations" in result:
                logger.info(f"[QUERY_TRANSFORM] Generated {len(result['variations'])} variations")
                return {
                    "success": True,
                    "original_query": query,
                    "variations": result["variations"],
                    "all_queries": [query] + result["variations"],
                    "reasoning": result.get("reasoning", "")
                }
            else:
                return {
                    "success": False,
                    "original_query": query,
                    "variations": [],
                    "all_queries": [query],
                    "error": "Failed to generate variations"
                }
                
        except Exception as e:
            logger.log_error("QueryTransformer.generate_multi_queries", e)
            return {
                "success": False,
                "original_query": query,
                "variations": [],
                "all_queries": [query],
                "error": str(e)
            }
    
    def rewrite_query(self, query: str) -> Dict:
        """
        Rewrite query for better retrieval
        
        Args:
            query: Original query
            
        Returns:
            Dictionary with rewritten query and improvements
        """
        try:
            logger.info(f"[QUERY_TRANSFORM] Rewriting: {query[:50]}...")
            
            # Fallback if Ollama not available
            if self.llm is None:
                return self._fallback_rewrite(query)
            
            chain = self.rewrite_prompt | self.llm
            response = chain.invoke({"query": query})
            
            # Parse JSON response
            result = self._parse_json_response(response.content)
            
            if result and "rewritten_query" in result:
                logger.info(f"[QUERY_TRANSFORM] Rewritten successfully")
                return {
                    "success": True,
                    "original_query": query,
                    "rewritten_query": result["rewritten_query"],
                    "improvements": result.get("improvements", []),
                    "reasoning": result.get("reasoning", "")
                }
            else:
                return {
                    "success": False,
                    "original_query": query,
                    "rewritten_query": query,
                    "improvements": [],
                    "error": "Failed to rewrite query"
                }
                
        except Exception as e:
            logger.log_error("QueryTransformer.rewrite_query", e)
            return {
                "success": False,
                "original_query": query,
                "rewritten_query": query,
                "improvements": [],
                "error": str(e)
            }
    
    def transform_query(self, query: str, mode: str = "auto") -> Dict:
        """
        Main transformation method that applies appropriate strategies
        
        Args:
            query: User query
            mode: Transformation mode
                - "auto": Automatically choose best strategy
                - "decompose": Force decomposition
                - "multi": Generate multiple variations
                - "rewrite": Rewrite for better retrieval
                - "all": Apply all transformations
                
        Returns:
            Transformed query results
        """
        try:
            logger.info(f"[QUERY_TRANSFORM] Transforming query (mode: {mode})")
            
            result = {
                "original_query": query,
                "mode": mode,
                "transformations": {}
            }
            
            if mode == "auto":
                # Smart decision: complex queries get decomposition, others get rewrite
                if self._is_complex_query(query):
                    result["transformations"]["decomposition"] = self.decompose_query(query)
                    result["recommended_strategy"] = "decomposition"
                else:
                    result["transformations"]["rewrite"] = self.rewrite_query(query)
                    result["recommended_strategy"] = "rewrite"
                    
            elif mode == "decompose":
                result["transformations"]["decomposition"] = self.decompose_query(query)
                result["recommended_strategy"] = "decomposition"
                
            elif mode == "multi":
                result["transformations"]["multi_query"] = self.generate_multi_queries(query)
                result["recommended_strategy"] = "multi_query"
                
            elif mode == "rewrite":
                result["transformations"]["rewrite"] = self.rewrite_query(query)
                result["recommended_strategy"] = "rewrite"
                
            elif mode == "all":
                result["transformations"]["decomposition"] = self.decompose_query(query)
                result["transformations"]["multi_query"] = self.generate_multi_queries(query)
                result["transformations"]["rewrite"] = self.rewrite_query(query)
                result["recommended_strategy"] = "all"
            
            return result
            
        except Exception as e:
            logger.log_error("QueryTransformer.transform_query", e)
            return {
                "original_query": query,
                "mode": mode,
                "error": str(e),
                "transformations": {}
            }
    
    def _is_complex_query(self, query: str) -> bool:
        """Determine if a query is complex and needs decomposition"""
        complexity_indicators = [
            "and", "or", "also", "additionally", "what about",
            "how do", "why", "explain", "tell me about",
            "multiple", "several", "both", "all"
        ]
        
        query_lower = query.lower()
        
        # Check for multiple questions
        if query.count("?") > 1:
            return True
        
        # Check for complexity indicators
        indicator_count = sum(1 for indicator in complexity_indicators if indicator in query_lower)
        
        # Check length
        word_count = len(query.split())
        
        return indicator_count >= 2 or word_count > 15
    
    def _parse_json_response(self, response: str) -> Optional[Dict]:
        """Parse JSON response from LLM, handling markdown code blocks"""
        try:
            # Remove markdown code blocks if present
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()
            
            # Parse JSON
            import json
            return json.loads(response)
        except Exception as e:
            logger.log_error("QueryTransformer._parse_json_response", e)
            return None
    
    def _fallback_decompose(self, query: str) -> Dict:
        """Simple rule-based decomposition fallback"""
        logger.info("[QUERY_TRANSFORM] Using fallback decomposition")
        
        # Split on common conjunctions
        separators = [" and ", " or ", ", and ", ", or ", " also "]
        sub_queries = [query]
        
        for sep in separators:
            if sep in query.lower():
                parts = query.split(sep)
                sub_queries = [p.strip() + "?" if not p.endswith("?") else p.strip() for p in parts]
                break
        
        needs_decomp = len(sub_queries) > 1
        
        return {
            "success": True,
            "needs_decomposition": needs_decomp,
            "sub_queries": sub_queries,
            "reasoning": "Fallback rule-based decomposition",
            "original_query": query
        }
    
    def _fallback_multi_query(self, query: str) -> Dict:
        """Simple query variation fallback"""
        logger.info("[QUERY_TRANSFORM] Using fallback multi-query")
        
        # Generate simple variations
        variations = []
        
        # Add "What" variation
        if not query.lower().startswith("what"):
            variations.append(f"What {query.lower()}")
        
        # Add "How" variation  
        if not query.lower().startswith("how"):
            variations.append(f"How {query.lower()}")
        
        # Add "When" variation
        if not query.lower().startswith("when"):
            variations.append(f"When {query.lower()}")
        
        return {
            "success": True,
            "original_query": query,
            "variations": variations[:3],
            "all_queries": [query] + variations[:3],
            "reasoning": "Fallback rule-based variations"
        }
    
    def _fallback_rewrite(self, query: str) -> Dict:
        """Simple query rewrite fallback"""
        logger.info("[QUERY_TRANSFORM] Using fallback rewrite")
        
        # Simple improvements
        rewritten = query.strip()
        improvements = []
        
        # Add question mark if missing
        if not rewritten.endswith("?"):
            rewritten += "?"
            improvements.append("Added question mark")
        
        # Expand common abbreviations
        abbrev_map = {
            "meds": "medications",
            "doc": "doctor",
            "appt": "appointment",
            "dr": "doctor"
        }
        
        for abbrev, full in abbrev_map.items():
            if abbrev in rewritten.lower():
                rewritten = rewritten.replace(abbrev, full)
                improvements.append(f"Expanded '{abbrev}' to '{full}'")
        
        return {
            "success": True,
            "original_query": query,
            "rewritten_query": rewritten,
            "improvements": improvements,
            "reasoning": "Fallback rule-based rewrite"
        }


def test_query_transformer():
    """Test the query transformer"""
    print("\n" + "="*80)
    print("Testing Query Transformer")
    print("="*80 + "\n")
    
    transformer = QueryTransformer()
    
    # Test 1: Simple query (rewrite)
    print("Test 1: Simple Query")
    query1 = "What meds should I take?"
    result1 = transformer.transform_query(query1, mode="auto")
    print(f"Original: {query1}")
    print(f"Strategy: {result1['recommended_strategy']}")
    if "rewrite" in result1["transformations"]:
        print(f"Rewritten: {result1['transformations']['rewrite']['rewritten_query']}")
    print()
    
    # Test 2: Complex query (decomposition)
    print("Test 2: Complex Query")
    query2 = "What medications should I take, what are the side effects, and when should I follow up?"
    result2 = transformer.transform_query(query2, mode="auto")
    print(f"Original: {query2}")
    print(f"Strategy: {result2['recommended_strategy']}")
    if "decomposition" in result2["transformations"]:
        print("Sub-queries:")
        for i, sq in enumerate(result2['transformations']['decomposition']['sub_queries'], 1):
            print(f"  {i}. {sq}")
    print()
    
    # Test 3: Multi-query generation
    print("Test 3: Multi-Query Generation")
    query3 = "When should I go to the emergency room?"
    result3 = transformer.transform_query(query3, mode="multi")
    print(f"Original: {query3}")
    if "multi_query" in result3["transformations"]:
        print("Variations:")
        for i, var in enumerate(result3['transformations']['multi_query']['variations'], 1):
            print(f"  {i}. {var}")
    print()
    
    print("="*80)
    print("✓ Query Transformer test complete")


if __name__ == "__main__":
    test_query_transformer()

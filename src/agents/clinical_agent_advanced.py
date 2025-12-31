"""
Clinical Agent with Advanced RAG
=================================

Enhanced Clinical Agent using Advanced RAG capabilities:
1. Query transformation for better retrieval
2. Hybrid search (vector + BM25)
3. Cross-encoder reranking
4. Summary index integration
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

from src.mcp.tools import MCPTools
from src.agents.prompts import CLINICAL_AGENT_SYSTEM_PROMPT
from src.config import GOOGLE_API_KEY
from src.utils.logger import get_logger
from src.advanced_rag import AdvancedRAG, create_advanced_rag, BM25Retriever
from src.summary_index import SummaryIndex
from src.pinecone_manager import PineconeManager

logger = get_logger()


class ClinicalAgentAdvanced:
    """Enhanced Clinical Agent with Advanced RAG capabilities"""

    def __init__(self, use_advanced_rag: bool = True):
        """
        Initialize Clinical Agent.

        Args:
            use_advanced_rag: Whether to use Advanced RAG (requires setup)
        """
        self.tools_handler = MCPTools()
        self.use_advanced_rag = use_advanced_rag

        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.3,
        )

        # Initialize Advanced RAG system if enabled
        self.advanced_rag = None
        self.summary_index = None

        if use_advanced_rag:
            try:
                self._initialize_advanced_rag()
                logger.info("✓ Advanced RAG system loaded")
            except Exception as e:
                logger.warning(f"Could not load Advanced RAG system: {e}")
                logger.info("Falling back to basic RAG. Run setup_advanced_rag.py to enable Advanced RAG.")
                self.use_advanced_rag = False

        # Create tools
        self.tools = self._create_tools()

        # Create prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", CLINICAL_AGENT_SYSTEM_PROMPT + self._get_advanced_rag_instructions()),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])

        # Create agent
        agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5,
            return_intermediate_steps=True
        )

        logger.info("✓ Clinical Agent (Advanced) initialized")

    def _initialize_advanced_rag(self):
        """Initialize Advanced RAG components"""
        # Load BM25 index
        bm25_path = Path("data/bm25_index.pkl")
        bm25_retriever = None

        if bm25_path.exists():
            logger.info("Loading BM25 index...")
            with open(bm25_path, 'rb') as f:
                bm25_retriever = pickle.load(f)
            logger.info(f"BM25 index loaded with {len(bm25_retriever.documents)} documents")
        else:
            logger.warning("BM25 index not found. Run setup_advanced_rag.py to create it.")

        # Load summary index
        summary_path = Path("data/summary_index.pkl")
        if summary_path.exists():
            logger.info("Loading summary index...")
            self.summary_index = SummaryIndex(
                llm=self.llm,
                storage_path=str(summary_path)
            )
            logger.info(f"Summary index loaded with {len(self.summary_index)} summaries")
        else:
            logger.warning("Summary index not found. Run setup_advanced_rag.py to create it.")

        # Initialize Pinecone
        pinecone_manager = PineconeManager()

        # Create Advanced RAG system
        self.advanced_rag = create_advanced_rag(
            pinecone_manager=pinecone_manager,
            llm=self.llm,
            documents_for_bm25=bm25_retriever.documents if bm25_retriever else None,
            use_reranking=True
        )

    def _get_advanced_rag_instructions(self) -> str:
        """Get additional instructions for Advanced RAG"""
        if not self.use_advanced_rag:
            return ""

        return """

**ADVANCED RAG CAPABILITIES:**
You have access to an Advanced RAG system with:
1. Query Transformation - Your queries are automatically expanded and refined
2. Hybrid Search - Combines semantic (vector) and keyword (BM25) search
3. Reranking - Results are reranked using cross-encoder for better relevance
4. Summary Index - Quick access to document overviews

When using search_medical_knowledge_advanced, you'll get:
- More relevant results through query transformation
- Better coverage through hybrid search
- Higher quality results through reranking
- Metadata about query transformation and retrieval strategy

Use this tool for all complex medical queries!
"""

    def _create_tools(self) -> List[Tool]:
        """Create agent tools"""
        tools = []

        # Advanced RAG tool (if enabled)
        if self.use_advanced_rag:
            tools.append(Tool(
                name="search_medical_knowledge_advanced",
                func=self._search_knowledge_advanced_wrapper,
                description="""Search comprehensive nephrology textbook using ADVANCED RAG with query transformation, hybrid search, and reranking.

USE THIS TOOL FOR:
- Complex medical questions requiring deep understanding
- Multi-faceted queries that need comprehensive coverage
- Questions about diseases, symptoms, treatments, procedures
- Comparison questions (e.g., "difference between A and B")
- Analytical questions (e.g., "what causes X?")

This provides better results than basic search. Input: query string"""
            ))

            # Summary search tool
            if self.summary_index:
                tools.append(Tool(
                    name="search_summaries",
                    func=self._search_summaries_wrapper,
                    description="""Search high-level document summaries for quick overview.

USE THIS TOOL FOR:
- Getting quick overview of topics
- Understanding document structure
- Finding which sections contain relevant information
- Initial exploration of unfamiliar topics

Input: query string"""
                ))

        # Basic RAG tool (always available as fallback)
        tools.append(Tool(
            name="search_medical_knowledge",
            func=self._search_knowledge_wrapper,
            description="Basic search of nephrology textbook. Use advanced search when available. Input: query string"
        ))

        # Web search tool
        tools.append(Tool(
            name="web_search",
            func=self._web_search_wrapper,
            description="""Search web for CURRENT medical information.

USE THIS TOOL FOR:
- Latest research and studies (2024, 2025)
- Recent clinical guidelines
- New treatments and medications
- Current medical news
- Any query with 'latest', 'recent', 'new'

Input: query string"""
        ))

        return tools

    def _search_knowledge_advanced_wrapper(self, query: str) -> str:
        """Advanced RAG search with query transformation and reranking"""
        try:
            logger.info(f"[CLINICAL-ADVANCED] Advanced RAG Search: {query[:50]}...")

            if not self.advanced_rag:
                logger.warning("[CLINICAL-ADVANCED] Advanced RAG not available, falling back to basic")
                return self._search_knowledge_wrapper(query)

            # Use Advanced RAG system
            documents, transform_result = self.advanced_rag.retrieve(
                query=query,
                top_k=5,
                transform_strategy="auto",
                use_reranking=True
            )

            if not documents:
                logger.warning("[CLINICAL-ADVANCED] No documents retrieved")
                return json.dumps({
                    "success": False,
                    "error": "No relevant documents found",
                    "results": []
                })

            # Format results
            results = []
            for i, doc in enumerate(documents, 1):
                results.append({
                    "rank": i,
                    "content": doc.content,
                    "relevance_score": doc.rerank_score or doc.score,
                    "source": doc.source,
                    "metadata": doc.metadata
                })

            # Build response
            response = {
                "success": True,
                "num_results": len(results),
                "results": results,
                "query_info": {
                    "original_query": transform_result.original_query,
                    "transformed_queries": transform_result.transformed_queries,
                    "query_type": transform_result.query_type.value,
                    "retrieval_strategy": transform_result.suggested_strategy.value,
                    "keywords": transform_result.keywords
                },
                "context": self.advanced_rag.format_context(documents)
            }

            logger.info(f"[CLINICAL-ADVANCED] Retrieved {len(documents)} documents")
            logger.info(f"[CLINICAL-ADVANCED] Query type: {transform_result.query_type.value}")
            logger.info(f"[CLINICAL-ADVANCED] Strategy: {transform_result.suggested_strategy.value}")

            return json.dumps(response, indent=2)

        except Exception as e:
            logger.log_error("ClinicalAgentAdvanced._search_knowledge_advanced_wrapper", e)
            # Fallback to basic search
            return self._search_knowledge_wrapper(query)

    def _search_summaries_wrapper(self, query: str) -> str:
        """Search document summaries"""
        try:
            logger.info(f"[CLINICAL-ADVANCED] Summary Search: {query[:50]}...")

            if not self.summary_index:
                return json.dumps({
                    "success": False,
                    "error": "Summary index not available",
                    "results": []
                })

            # Search summaries
            summaries = self.summary_index.search(query, top_k=5)

            if not summaries:
                return json.dumps({
                    "success": False,
                    "error": "No relevant summaries found",
                    "results": []
                })

            # Format results
            results = []
            for i, summary in enumerate(summaries, 1):
                results.append({
                    "rank": i,
                    "doc_id": summary.doc_id,
                    "summary": summary.summary,
                    "key_concepts": summary.key_concepts,
                    "metadata": summary.metadata
                })

            response = {
                "success": True,
                "num_results": len(results),
                "results": results,
                "message": "Found relevant document summaries. Use search_medical_knowledge_advanced for detailed information."
            }

            logger.info(f"[CLINICAL-ADVANCED] Found {len(summaries)} summaries")

            return json.dumps(response, indent=2)

        except Exception as e:
            logger.log_error("ClinicalAgentAdvanced._search_summaries_wrapper", e)
            return json.dumps({
                "success": False,
                "error": str(e),
                "results": []
            })

    def _search_knowledge_wrapper(self, query: str) -> str:
        """Basic RAG search (fallback)"""
        try:
            logger.info(f"[CLINICAL] Basic RAG Search: {query[:50]}...")
            result = self.tools_handler.search_medical_knowledge(query, top_k=3)

            if result.get("success"):
                logger.info(f"[CLINICAL] RAG found {result.get('num_results', 0)} results")
                return json.dumps(result, indent=2)
            else:
                logger.warning(f"[CLINICAL] RAG search failed: {result.get('error')}")
                return json.dumps({
                    "success": False,
                    "error": "Knowledge base search failed",
                    "results": []
                })
        except Exception as e:
            logger.log_error("ClinicalAgentAdvanced._search_knowledge_wrapper", e)
            return json.dumps({
                "success": False,
                "error": str(e),
                "results": []
            })

    def _web_search_wrapper(self, query: str) -> str:
        """Web search wrapper"""
        try:
            logger.info(f"[CLINICAL] Web Search: {query[:50]}...")

            # Use the simplified MCP web search client
            from src.mcp.web_search_client import search_web

            result = search_web(query, max_results=3)

            if result.get("success"):
                logger.info(f"[CLINICAL] Web search found {result.get('num_results', 0)} results")
                return json.dumps(result, indent=2)
            else:
                logger.warning(f"[CLINICAL] Web search failed: {result.get('error')}")
                # Fallback to regular Tavily search
                result = self.tools_handler.web_search(query, max_results=3)
                return json.dumps(result, indent=2)

        except Exception as e:
            logger.log_error("ClinicalAgentAdvanced._web_search_wrapper", e)
            # Fallback to regular web search
            try:
                result = self.tools_handler.web_search(query, max_results=3)
                return json.dumps(result, indent=2)
            except Exception as fallback_error:
                logger.log_error("ClinicalAgentAdvanced._web_search_wrapper.fallback", fallback_error)
                return json.dumps({
                    "success": False,
                    "error": f"Both MCP and fallback search failed",
                    "results": []
                })

    def _extract_actual_query(self, message: str) -> str:
        """Extract the actual patient query from workflow message"""
        # Remove common routing phrases
        routing_phrases = [
            "**Receptionist Agent:**",
            "This sounds like a medical concern.",
            "Let me connect you with our Clinical Agent",
            "for expert advice.",
            "Receptionist Agent:"
        ]

        cleaned = message
        for phrase in routing_phrases:
            cleaned = cleaned.replace(phrase, "")

        # Clean up whitespace
        cleaned = " ".join(cleaned.split()).strip()

        logger.info(f"[CLINICAL] Extracted query: {cleaned[:100]}...")
        return cleaned

    def process(self, message: str, chat_history: List = None, patient_context: Dict = None) -> Dict:
        """
        Process user message through clinical agent.

        Args:
            message: User medical question
            chat_history: Previous conversation
            patient_context: Patient information for context

        Returns:
            Dict with agent response
        """
        # Extract actual query from routing message
        actual_query = self._extract_actual_query(message)

        logger.info(f"[CLINICAL] Processing: {actual_query[:50]}...")

        # Build chat history for LangChain
        langchain_history = []
        if chat_history:
            for msg in chat_history:
                if msg.get("role") == "user":
                    content = self._extract_actual_query(msg["content"])
                    langchain_history.append(HumanMessage(content=content))
                elif msg.get("role") == "assistant":
                    langchain_history.append(AIMessage(content=msg["content"]))

        # Add patient context if available
        enhanced_query = actual_query
        if patient_context:
            diagnosis = patient_context.get('primary_diagnosis', 'N/A')
            enhanced_query = f"{actual_query}\n\n[Patient Context: {diagnosis}]"

        # Retry logic with better error handling
        max_retries = 2
        last_error = None

        for attempt in range(max_retries):
            try:
                logger.info(f"[CLINICAL] Attempt {attempt + 1}/{max_retries}")

                # Invoke agent
                response = self.agent_executor.invoke({
                    "input": enhanced_query,
                    "chat_history": langchain_history
                })

                # Extract output
                output = self._extract_output(response)

                # Validate response
                if output and len(output.strip()) > 50:
                    # Check if it's just the greeting
                    if "hello ashley" in output.lower() and "please tell me what" in output.lower():
                        logger.warning(f"[CLINICAL] Got generic greeting, retrying...")
                        if attempt < max_retries - 1:
                            import time
                            time.sleep(1)
                            continue

                    logger.info(f"[CLINICAL] Success! Response: {len(output)} chars")

                    return {
                        "message": output,
                        "needs_routing": False,
                        "agent": "clinical_advanced" if self.use_advanced_rag else "clinical"
                    }
                else:
                    logger.warning(f"[CLINICAL] Short/empty response on attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        import time
                        time.sleep(1)
                        continue

            except Exception as e:
                last_error = e
                logger.warning(f"[CLINICAL] Attempt {attempt + 1} failed: {str(e)[:100]}")

                if attempt < max_retries - 1:
                    import time
                    time.sleep(1)
                    continue
                else:
                    logger.log_error("ClinicalAgentAdvanced.process", e)

        # All retries failed - provide fallback
        logger.error("[CLINICAL] All retry attempts exhausted")

        # Import fallback from original agent
        from src.agents.clinical_agent import ClinicalAgent
        fallback_agent = ClinicalAgent()

        return {
            "message": fallback_agent._get_informed_fallback(actual_query, patient_context),
            "needs_routing": False,
            "agent": "clinical_advanced",
            "error": str(last_error) if last_error else "Unknown error"
        }

    def _extract_output(self, response: Dict) -> str:
        """Extract output from agent response"""

        # Try different response formats
        if isinstance(response, dict):
            # Format 1: Direct output key
            if "output" in response:
                return response["output"]

            # Format 2: Messages list
            if "messages" in response:
                messages = response["messages"]
                if messages and len(messages) > 0:
                    last_msg = messages[-1]
                    if hasattr(last_msg, 'content'):
                        return last_msg.content
                    elif isinstance(last_msg, dict) and "content" in last_msg:
                        return last_msg["content"]

            # Format 3: Return value
            if "return_values" in response and "output" in response["return_values"]:
                return response["return_values"]["output"]

        # Fallback: convert to string
        return str(response)


def test_advanced_clinical():
    """Test Advanced Clinical Agent"""
    print("\n" + "="*80)
    print("Testing Advanced Clinical Agent")
    print("="*80 + "\n")

    agent = ClinicalAgentAdvanced(use_advanced_rag=True)

    # Test queries
    test_cases = [
        {
            "query": "What are the differences between acute and chronic kidney disease?",
            "context": {"primary_diagnosis": "CKD Stage 3"}
        },
        {
            "query": "I'm having swelling in my legs. Should I be worried?",
            "context": {"primary_diagnosis": "CKD Stage 3"}
        },
        {
            "query": "What causes kidney failure?",
            "context": None
        }
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}: {test['query']}")
        print('='*80)

        response = agent.process(
            test['query'],
            patient_context=test['context']
        )

        print(f"\nAgent: {response.get('agent')}")
        print(f"Success: {'error' not in response}")
        print(f"Response length: {len(response['message'])} chars")
        print(f"\nResponse preview:")
        print(response['message'][:400] + "..." if len(response['message']) > 400 else response['message'])

    print("\n" + "="*80)
    print("✓ Advanced Clinical Agent test complete")
    print("="*80)


if __name__ == "__main__":
    test_advanced_clinical()

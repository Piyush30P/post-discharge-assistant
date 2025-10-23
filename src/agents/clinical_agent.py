"""
Clinical Agent
Handles complex medical queries using RAG and web search
FIXED: Added retry logic, better error handling, proper response extraction
"""

import json
from typing import Dict, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

from src.mcp.tools import MCPTools
from src.agents.prompts import CLINICAL_AGENT_SYSTEM_PROMPT
from src.config import GOOGLE_API_KEY
from src.utils.logger import get_logger

logger = get_logger()


class ClinicalAgent:
    """Handles complex medical queries with knowledge base access"""
    
    def __init__(self):
        self.tools_handler = MCPTools()
        
        # Initialize LLM with higher quality model
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.3,  # Lower temperature for medical accuracy
        )
        
        # Create tools
        self.tools = [
            Tool(
                name="search_medical_knowledge",
                func=self._search_knowledge_wrapper,
                description="Search comprehensive nephrology textbook for medical information. Use this for medical knowledge, disease information, symptoms, treatments. Input: query string"
            ),
            Tool(
                name="web_search",
                func=self._web_search_wrapper,
                description="Search web for CURRENT medical information. Use this for: latest research, recent studies, new treatments, current guidelines, anything with 'latest', 'recent', 'new', '2024', '2025', 'news'. Input: query string"
            )
        ]
        
        # Create prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", CLINICAL_AGENT_SYSTEM_PROMPT),
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
        
        logger.info("✓ Clinical Agent initialized")
    
    def _search_knowledge_wrapper(self, query: str) -> str:
        """Wrapper for search_medical_knowledge tool"""
        try:
            logger.info(f"[CLINICAL] RAG Search: {query[:50]}...")
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
            logger.log_error("ClinicalAgent._search_knowledge_wrapper", e)
            return json.dumps({
                "success": False,
                "error": str(e),
                "results": []
            })
    
    def _web_search_wrapper(self, query: str) -> str:
        """Wrapper for web_search tool"""
        try:
            logger.info(f"[CLINICAL] Web Search: {query[:50]}...")
            result = self.tools_handler.web_search(query, max_results=3)
            
            if result.get("success"):
                logger.info(f"[CLINICAL] Web search found {result.get('num_results', 0)} results")
                return json.dumps(result, indent=2)
            else:
                logger.warning(f"[CLINICAL] Web search failed: {result.get('error')}")
                return json.dumps({
                    "success": False,
                    "error": "Web search failed",
                    "results": []
                })
        except Exception as e:
            logger.log_error("ClinicalAgent._web_search_wrapper", e)
            return json.dumps({
                "success": False,
                "error": str(e),
                "results": []
            })
    
    def process(self, message: str, chat_history: List = None, patient_context: Dict = None) -> Dict:
        """
        Process user message through clinical agent with retry logic
        
        Args:
            message: User medical question
            chat_history: Previous conversation
            patient_context: Patient information for context
            
        Returns:
            Dict with agent response
        """
        logger.info(f"[CLINICAL] Processing: {message[:50]}...")
        
        # Build chat history for LangChain
        langchain_history = []
        if chat_history:
            for msg in chat_history:
                if msg.get("role") == "user":
                    langchain_history.append(HumanMessage(content=msg["content"]))
                elif msg.get("role") == "assistant":
                    langchain_history.append(AIMessage(content=msg["content"]))
        
        # Add patient context to message if available
        enhanced_message = message
        if patient_context:
            diagnosis = patient_context.get('primary_diagnosis', 'N/A')
            enhanced_message = f"{message}\n\n[Patient Context: {diagnosis}]"
        
        # Retry logic for tool failures
        max_retries = 2
        last_error = None
        
        for attempt in range(max_retries):
            try:
                logger.info(f"[CLINICAL] Attempt {attempt + 1}/{max_retries}")
                
                # Invoke agent
                response = self.agent_executor.invoke({
                    "input": enhanced_message,
                    "chat_history": langchain_history
                })
                
                # Extract output
                output = self._extract_output(response)
                
                if output and len(output.strip()) > 20:  # Valid response
                    logger.info(f"[CLINICAL] Success! Response: {len(output)} chars")
                    
                    return {
                        "message": output,
                        "needs_routing": False,
                        "agent": "clinical"
                    }
                else:
                    logger.warning(f"[CLINICAL] Empty or invalid response on attempt {attempt + 1}")
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
                    logger.log_error("ClinicalAgent.process", e)
        
        # All retries failed
        logger.error("[CLINICAL] All retry attempts exhausted")
        
        return {
            "message": self._get_fallback_response(message, patient_context),
            "needs_routing": False,
            "agent": "clinical",
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
    
    def _get_fallback_response(self, message: str, patient_context: Dict = None) -> str:
        """Generate fallback response when tools fail"""
        
        base_response = """I apologize, but I'm experiencing technical difficulties accessing my medical knowledge base right now. 

However, I want to ensure you get the help you need:

**For Immediate Concerns:**
- If you're experiencing severe symptoms (chest pain, difficulty breathing, severe swelling), please call 911 or visit the nearest emergency room immediately.
- For urgent but non-emergency concerns, contact your healthcare provider's on-call service.

**For General Information:**
- You can contact your nephrologist's office during business hours
- Your discharge paperwork should have emergency contact numbers
- Consider calling your dialysis center if you're on dialysis

**When Our System Is Back:**
Please try asking your question again in a few moments, and I'll do my best to provide evidence-based information.

Is there anything from your discharge instructions I can help clarify while we resolve this technical issue?"""

        return base_response
    
    def _extract_sources(self, response: Dict) -> List[str]:
        """Extract sources from tool calls"""
        sources = []
        intermediate_steps = response.get("intermediate_steps", [])
        
        for step in intermediate_steps:
            if len(step) >= 2:
                tool_name = step[0].tool if hasattr(step[0], 'tool') else "unknown"
                tool_output = str(step[1])
                
                # Extract source info
                if "page" in tool_output.lower() or "source" in tool_output.lower():
                    sources.append(f"{tool_name}: {tool_output[:150]}")
        
        return sources


def test_clinical():
    """Test clinical agent"""
    print("\n" + "="*80)
    print("Testing Clinical Agent with Retry Logic")
    print("="*80 + "\n")
    
    agent = ClinicalAgent()
    
    # Test 1: Medical query (RAG)
    print("Test 1: Medical knowledge query")
    response = agent.process("What are the treatment options for chronic kidney disease stage 3?")
    print(f"Response: {response['message'][:300]}...")
    print(f"Success: {'error' not in response}\n")
    
    # Test 2: Web search query
    print("Test 2: Current information query")
    response = agent.process("What's the latest research on SGLT2 inhibitors for kidney disease?")
    print(f"Response: {response['message'][:300]}...")
    print(f"Success: {'error' not in response}\n")
    
    # Test 3: Symptom query
    print("Test 3: Symptom assessment")
    response = agent.process("I'm having swelling in my legs", 
                            patient_context={"primary_diagnosis": "CKD Stage 3"})
    print(f"Response: {response['message'][:300]}...")
    print(f"Success: {'error' not in response}\n")
    
    print("="*80)
    print("✓ Clinical Agent test complete")


if __name__ == "__main__":
    test_clinical()
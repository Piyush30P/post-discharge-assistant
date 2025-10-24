"""
Clinical Agent - FIXED VERSION
Handles complex medical queries using RAG and web search
Key fixes:
1. Proper message extraction from workflow
2. Enhanced tool invocation
3. Better response handling
4. Improved error recovery
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
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.3,
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
    
    def _extract_actual_query(self, message: str) -> str:
        """
        Extract the actual patient query from workflow message
        Removes receptionist routing text
        """
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
        Process user message through clinical agent
        
        Args:
            message: User medical question (may contain routing text)
            chat_history: Previous conversation
            patient_context: Patient information for context
            
        Returns:
            Dict with agent response
        """
        # CRITICAL FIX: Extract actual query from routing message
        actual_query = self._extract_actual_query(message)
        
        logger.info(f"[CLINICAL] Processing: {actual_query[:50]}...")
        
        # Build chat history for LangChain
        langchain_history = []
        if chat_history:
            for msg in chat_history:
                if msg.get("role") == "user":
                    # Extract actual queries from history too
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
                logger.info(f"[CLINICAL] Invoking agent executor...")
                
                # Invoke agent
                response = self.agent_executor.invoke({
                    "input": enhanced_query,
                    "chat_history": langchain_history
                })
                
                # Extract output
                output = self._extract_output(response)
                
                # Validate response
                if output and len(output.strip()) > 50:  # Reasonable response length
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
                        "agent": "clinical"
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
                    logger.log_error("ClinicalAgent.process", e)
        
        # All retries failed - provide fallback
        logger.error("[CLINICAL] All retry attempts exhausted")
        
        return {
            "message": self._get_informed_fallback(actual_query, patient_context),
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
    
    def _get_informed_fallback(self, query: str, patient_context: Dict = None) -> str:
        """Generate an informed fallback response based on query type"""
        
        query_lower = query.lower()
        
        # Symptom-related queries
        if any(word in query_lower for word in ['swelling', 'pain', 'fever', 'headache', 'nausea', 'dizzy']):
            response = f"""I understand you're experiencing symptoms. Based on your query about {query}, here's what I can tell you:

**When to Seek Immediate Care:**
- Severe or worsening symptoms
- Difficulty breathing or chest pain
- High fever (>101°F / 38.3°C)
- Confusion or altered consciousness
- Severe swelling that's rapidly increasing

**General Guidance for Your Symptoms:**
"""
            if 'swelling' in query_lower:
                response += """
For leg swelling in kidney patients:
- Elevate your legs when resting
- Monitor your fluid intake according to your restrictions
- Check your weight daily
- Contact your nephrologist if swelling worsens
"""
            
            if 'fever' in query_lower or 'headache' in query_lower:
                response += """
For fever or persistent headache:
- Monitor your temperature regularly
- Stay hydrated (within your fluid restrictions)
- Avoid NSAIDs unless approved by your doctor
- Contact your doctor if fever persists beyond 2-3 days
"""
        
        # Research/information queries
        elif any(word in query_lower for word in ['latest', 'research', 'study', 'sglt2', 'treatment', 'new']):
            response = f"""I apologize, but I'm having difficulty accessing current medical literature right now.

Regarding {query}, I recommend:

**For Latest Research:**
- Check with your nephrologist at your next appointment
- Visit reputable sources like:
  - National Kidney Foundation (kidney.org)
  - American Society of Nephrology (asn-online.org)
  - National Institute of Diabetes and Digestive and Kidney Diseases (niddk.nih.gov)

**For Treatment Questions:**
- Discuss any new treatments with your healthcare provider
- Ask about clinical trials in your area if interested
- Ensure any new treatment aligns with your current care plan
"""
        
        # General medical questions
        else:
            response = f"""I apologize, but I'm experiencing technical difficulties accessing my medical knowledge base.

For your question about {query}:

**Recommended Next Steps:**
1. Contact your nephrologist's office during business hours
2. Review your discharge instructions for related guidance
3. If urgent, call your doctor's on-call service
4. For emergencies, call 911 or visit the nearest ER
"""
        
        # Add patient-specific context
        if patient_context and patient_context.get('primary_diagnosis'):
            response += f"\n**Your Diagnosis:** {patient_context['primary_diagnosis']}\n"
        
        response += """
**Important Reminders:**
⚠️ This system is for informational purposes only
⚠️ Always consult your healthcare provider for medical advice
⚠️ Don't stop or change medications without doctor guidance

Would you like me to help you find specific information from your discharge instructions instead?"""
        
        return response


def test_clinical():
    """Test clinical agent with actual queries"""
    print("\n" + "="*80)
    print("Testing FIXED Clinical Agent")
    print("="*80 + "\n")
    
    agent = ClinicalAgent()
    
    # Test queries
    test_cases = [
        {
            "query": "I'm having swelling in my legs. Should I be worried?",
            "context": {"primary_diagnosis": "CKD Stage 3"}
        },
        {
            "query": "What is chronic kidney disease?",
            "context": None
        },
        {
            "query": "What's the latest research on SGLT2 inhibitors?",
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
        
        print(f"\nAgent: clinical")
        print(f"Success: {'error' not in response}")
        print(f"Response length: {len(response['message'])} chars")
        print(f"\nResponse preview:")
        print(response['message'][:300] + "..." if len(response['message']) > 300 else response['message'])
    
    print("\n" + "="*80)
    print("✓ Clinical Agent test complete")
    print("="*80)


if __name__ == "__main__":
    test_clinical()
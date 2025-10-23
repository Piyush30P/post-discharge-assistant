"""
Receptionist Agent
Handles patient greeting, data retrieval, and basic queries
"""

import json
from typing import Dict, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate

from src.mcp.tools import MCPTools
from src.agents.prompts import RECEPTIONIST_SYSTEM_PROMPT
from src.config import GOOGLE_API_KEY
from src.utils.logger import get_logger

logger = get_logger()


class ReceptionistAgent:
    """Handles patient greeting and basic information retrieval"""
    
    def __init__(self):
        self.tools_handler = MCPTools()
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.7,
            
        )
        
        # Create tools
        self.tools = [
            Tool(
                name="get_patient_data",
                func=self.tools_handler.get_patient_data,
                description="Retrieves patient discharge information from database. Input: patient_name (str)"
            )
        ]
        
        # Create prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", RECEPTIONIST_SYSTEM_PROMPT),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        # Create agent
        agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
        self.executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )
        
        logger.info("✓ Receptionist Agent initialized")
    
    def process(self, message: str, chat_history: List = None) -> Dict:
        """
        Process user message through receptionist agent
        
        Args:
            message: User input
            chat_history: Previous conversation history
            
        Returns:
            Agent response and metadata
        """
        try:
            logger.info(f"[RECEPTIONIST] Processing: {message[:50]}...")
            
            # Invoke agent
            response = self.executor.invoke({
                "input": message,
                "chat_history": chat_history or []
            })
            
            # Check if routing needed
            needs_routing = self._check_routing_needed(response["output"])
            
            result = {
                "agent": "receptionist",
                "message": response["output"],
                "needs_routing": needs_routing,
                "route_to": "clinical" if needs_routing else None,
                "intermediate_steps": response.get("intermediate_steps", [])
            }
            
            logger.info(f"[RECEPTIONIST] Response generated (routing: {needs_routing})")
            return result
            
        except Exception as e:
            logger.log_error("ReceptionistAgent.process", e)
            return {
                "agent": "receptionist",
                "message": "I apologize, but I'm having trouble processing your request. Could you please rephrase that?",
                "error": str(e),
                "needs_routing": False
            }
    
    def _check_routing_needed(self, response: str) -> bool:
        """Check if response indicates routing to clinical agent is needed"""
        routing_indicators = [
            "clinical agent",
            "medical specialist",
            "detailed medical",
            "complex question",
            "let me connect you",
            "hand you over"
        ]
        
        response_lower = response.lower()
        return any(indicator in response_lower for indicator in routing_indicators)


def test_receptionist():
    """Test receptionist agent"""
    print("\n" + "="*80)
    print("Testing Receptionist Agent")
    print("="*80 + "\n")
    
    agent = ReceptionistAgent()
    
    # Test 1: Greeting
    print("Test 1: Initial greeting")
    response = agent.process("Hello, I was discharged yesterday")
    print(f"Response: {response['message'][:200]}...\n")
    
    # Test 2: Patient data retrieval
    print("Test 2: Patient data retrieval")
    response = agent.process("My name is John Brown")
    print(f"Response: {response['message'][:200]}...\n")
    
    print("="*80)
    print("✓ Receptionist Agent test complete")


if __name__ == "__main__":
    test_receptionist()
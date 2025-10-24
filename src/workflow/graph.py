"""
LangGraph Workflow - FIXED VERSION
Key fixes:
1. Proper message extraction when routing between agents
2. Only pass actual user queries to clinical agent
3. Better state management
4. Cleaner agent transitions
"""

from typing import Dict, List, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.workflow.state import AgentState, create_initial_state
from src.agents.receptionist_agent import ReceptionistAgent
# Import fixed clinical agent or use the same import
from src.agents.clinical_agent import ClinicalAgent
from src.utils.logger import get_logger

logger = get_logger()


class MultiAgentWorkflow:
    """Manages multi-agent conversation workflow"""
    
    def __init__(self):
        # Initialize agents
        logger.info("Initializing Multi-Agent Workflow...")
        self.receptionist = ReceptionistAgent()
        self.clinical = ClinicalAgent()
        
        # Build graph
        self.graph = self._build_graph()
        self.memory = MemorySaver()
        self.app = self.graph.compile(checkpointer=self.memory)
        
        logger.info("✓ Multi-Agent Workflow ready")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        # Create graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("receptionist", self._receptionist_node)
        workflow.add_node("clinical", self._clinical_node)
        
        # Add edges
        workflow.set_entry_point("receptionist")
        
        # Conditional routing from receptionist
        workflow.add_conditional_edges(
            "receptionist",
            self._route_from_receptionist,
            {
                "clinical": "clinical",
                "end": END
            }
        )
        
        # Clinical goes directly to END
        workflow.add_edge("clinical", END)
        
        return workflow
    
    def _extract_message_content(self, msg) -> str:
        """Helper to extract content from message (dict or object)"""
        if hasattr(msg, 'content'):
            return msg.content
        elif isinstance(msg, dict):
            return msg.get("content", "")
        else:
            return str(msg)
    
    def _convert_messages_to_dict(self, messages: List) -> List[Dict]:
        """Convert messages to dict format for agents"""
        result = []
        for msg in messages:
            if hasattr(msg, 'content'):
                # LangChain Message object
                result.append({
                    "role": "assistant" if msg.type == "ai" else msg.type,
                    "content": msg.content
                })
            elif isinstance(msg, dict):
                result.append(msg)
        return result
    
    def _get_original_user_message(self, messages: List) -> str:
        """
        Get the most recent actual user message
        Skips any routing/agent messages
        """
        for msg in reversed(messages):
            content = self._extract_message_content(msg)
            role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "type", None)
            
            # Look for user messages
            if role == "user":
                return content
        
        # Fallback to last message
        if messages:
            return self._extract_message_content(messages[-1])
        return ""
    
    def _receptionist_node(self, state: AgentState) -> Dict:
        """Process message through receptionist"""
        logger.info("[WORKFLOW] Receptionist node")
        
        # Get last user message
        last_message = self._get_original_user_message(state["messages"])
        
        logger.info(f"[WORKFLOW] Processing user message: {last_message[:50]}...")
        
        # Get chat history (exclude last message)
        chat_history = []
        if len(state["messages"]) > 1:
            chat_history = self._convert_messages_to_dict(state["messages"][:-1])
        
        # Process through receptionist
        response = self.receptionist.process(last_message, chat_history)
        
        # Add agent label to response
        labeled_response = f"**Receptionist Agent:** {response['message']}"
        
        # Store the original user message for clinical agent if routing
        original_user_query = last_message if response.get("needs_routing") else None
        
        # Update state
        return {
            "messages": [{"role": "assistant", "content": labeled_response}],
            "current_agent": "receptionist",
            "needs_routing": response.get("needs_routing", False),
            "route_to": response.get("route_to"),
            "turn_count": state.get("turn_count", 0) + 1,
            # CRITICAL: Store original query for clinical agent
            "last_user_query": original_user_query
        }
    
    def _clinical_node(self, state: AgentState) -> Dict:
        """Process message through clinical agent"""
        logger.info("[WORKFLOW] Clinical node")
        
        # CRITICAL FIX: Get the original user message, not the receptionist's routing message
        user_query = state.get("last_user_query")
        
        if not user_query:
            # Fallback: try to extract from messages
            user_query = self._get_original_user_message(state["messages"])
        
        logger.info(f"[WORKFLOW] Passing to clinical: {user_query[:50]}...")
        
        # Get chat history (excluding routing messages)
        chat_history = []
        for msg in state["messages"]:
            content = self._extract_message_content(msg)
            role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "type", "user")
            
            # Skip receptionist routing messages
            if "let me connect you" in content.lower() or "clinical agent" in content.lower():
                continue
            
            chat_history.append({"role": role, "content": content})
        
        # Process through clinical agent
        response = self.clinical.process(
            user_query,  # Pass the actual user question
            chat_history,
            patient_context=state.get("patient_data")
        )
        
        # Add agent label to response
        labeled_response = f"**Clinical Agent:** {response['message']}"
        
        # Update state
        return {
            "messages": [{"role": "assistant", "content": labeled_response}],
            "current_agent": "clinical",
            "needs_routing": False,
            "turn_count": state.get("turn_count", 0) + 1
        }
    
    def _route_from_receptionist(self, state: AgentState) -> Literal["clinical", "end"]:
        """Determine routing from receptionist"""
        
        if state.get("needs_routing") and state.get("route_to") == "clinical":
            logger.info("[WORKFLOW] Routing to clinical agent")
            return "clinical"
        else:
            logger.info("[WORKFLOW] Ending at receptionist")
            return "end"
    
    def process_message(self, message: str, conversation_id: str = "default") -> str:
        """
        Process a user message through the workflow
        
        Args:
            message: User input
            conversation_id: Unique conversation identifier
            
        Returns:
            Assistant response
        """
        logger.info(f"[WORKFLOW] Processing message: {message[:50]}...")
        
        try:
            # Create config for memory
            config = {"configurable": {"thread_id": conversation_id}}
            
            # Get current state or create new
            try:
                current_state = self.app.get_state(config)
                if current_state and hasattr(current_state, 'values') and current_state.values:
                    state = dict(current_state.values)
                else:
                    state = create_initial_state(conversation_id)
            except Exception as e:
                logger.info(f"[WORKFLOW] Creating new state: {e}")
                state = create_initial_state(conversation_id)
            
            # Ensure state has required keys
            if 'messages' not in state or state['messages'] is None:
                state['messages'] = []
            if 'turn_count' not in state:
                state['turn_count'] = 0
            if 'current_agent' not in state:
                state['current_agent'] = 'receptionist'
            if 'needs_routing' not in state:
                state['needs_routing'] = False
            if 'conversation_id' not in state:
                state['conversation_id'] = conversation_id
            if 'tools_used' not in state:
                state['tools_used'] = []
            if 'patient_name' not in state:
                state['patient_name'] = None
            if 'patient_data' not in state:
                state['patient_data'] = None
            if 'route_to' not in state:
                state['route_to'] = None
            if 'route_reason' not in state:
                state['route_reason'] = None
            if 'last_user_query' not in state:
                state['last_user_query'] = None
            
            # Add user message
            state["messages"].append({"role": "user", "content": message})
            
            # Run workflow
            result = self.app.invoke(state, config)
            
            # Get assistant response
            assistant_messages = []
            for msg in result.get("messages", []):
                if hasattr(msg, 'type') and msg.type in ["assistant", "ai"]:
                    assistant_messages.append({"content": msg.content})
                elif isinstance(msg, dict) and msg.get("role") == "assistant":
                    assistant_messages.append(msg)
            
            if assistant_messages:
                response = assistant_messages[-1]["content"]
                logger.info(f"[WORKFLOW] Response generated ({len(response)} chars)")
                return response
            else:
                return "I apologize, but I couldn't process your message. Please try again."
                
        except Exception as e:
            logger.log_error("MultiAgentWorkflow.process_message", e)
            import traceback
            traceback.print_exc()
            return f"I'm sorry, but I encountered an error. Please try starting a new conversation."
    
    def get_conversation_history(self, conversation_id: str = "default") -> List[Dict]:
        """Get conversation history for a given ID"""
        try:
            config = {"configurable": {"thread_id": conversation_id}}
            state = self.app.get_state(config)
            if state and state.values:
                messages = state.values.get("messages", [])
                return self._convert_messages_to_dict(messages)
            return []
        except:
            return []
    
    def reset_conversation(self, conversation_id: str = "default"):
        """Reset conversation for a given ID"""
        logger.info(f"[WORKFLOW] Resetting conversation: {conversation_id}")


def test_workflow():
    """Test the fixed workflow"""
    print("\n" + "="*80)
    print("Testing FIXED Multi-Agent Workflow")
    print("="*80 + "\n")
    
    workflow = MultiAgentWorkflow()
    
    # Test conversation
    test_messages = [
        "Hello",
        "My name is Ashley King",
        "I'm having swelling in my legs. Should I be worried?",
        "What's the latest research on kidney disease?"
    ]
    
    for i, msg in enumerate(test_messages, 1):
        print(f"\n{'='*80}")
        print(f"Turn {i}")
        print('='*80)
        print(f"User: {msg}")
        
        response = workflow.process_message(msg, "test-fixed")
        
        # Check which agent responded
        if "**Receptionist Agent:**" in response:
            print("✓ Agent: Receptionist")
        elif "**Clinical Agent:**" in response:
            print("✓ Agent: Clinical")
        else:
            print("✗ Agent label missing")
        
        print(f"\nResponse: {response[:200]}...")
    
    print("\n" + "="*80)
    print("✓ Workflow test complete")
    print("="*80)


if __name__ == "__main__":
    test_workflow()
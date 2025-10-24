"""
Workflow State Management - UPDATED
Added last_user_query to track original user messages during routing
"""

from typing import TypedDict, List, Dict, Optional, Annotated
from langgraph.graph import add_messages


class AgentState(TypedDict):
    """State for the multi-agent workflow"""
    
    # Conversation messages
    messages: Annotated[List[Dict], add_messages]
    
    # Patient information
    patient_name: Optional[str]
    patient_data: Optional[Dict]
    
    # Current agent
    current_agent: str  # "receptionist" or "clinical"
    
    # Tool usage tracking
    tools_used: List[str]
    
    # Conversation metadata
    conversation_id: str
    turn_count: int
    
    # Routing decisions
    needs_routing: bool
    route_to: Optional[str]
    route_reason: Optional[str]
    
    # ADDED: Store original user query for proper clinical agent routing
    last_user_query: Optional[str]


def create_initial_state(conversation_id: str) -> AgentState:
    """Create initial state for new conversation"""
    return AgentState(
        messages=[],
        patient_name=None,
        patient_data=None,
        current_agent="receptionist",
        tools_used=[],
        conversation_id=conversation_id,
        turn_count=0,
        needs_routing=False,
        route_to=None,
        route_reason=None,
        last_user_query=None  # ADDED
    )
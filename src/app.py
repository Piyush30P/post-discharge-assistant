"""
Main Application
Streamlit UI for the Post-Discharge Assistant
Enhanced with agent labels, Enter key support, and better UX
FIXED: Better error handling and response display
"""

import streamlit as st

# CRITICAL: set_page_config MUST be first Streamlit command
st.set_page_config(
    page_title="Post-Discharge Medical Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import uuid
from datetime import datetime
from src.workflow.graph import MultiAgentWorkflow
from src.utils.logger import get_logger

logger = get_logger()


# Custom CSS for better UI
def load_custom_css():
    st.markdown("""
    <style>
    /* Main chat container */
    .stChatFloatingInputContainer {
        bottom: 20px;
        background-color: transparent;
    }
    
    /* Chat messages */
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    
    /* User message */
    .stChatMessage[data-testid="user-message"] {
        background-color: #e3f2fd;
    }
    
    /* Assistant message */
    .stChatMessage[data-testid="assistant-message"] {
        background-color: #f5f5f5;
    }
    
    /* Agent labels styling */
    strong {
        color: #1976d2;
        font-size: 0.95rem;
        font-weight: 600;
    }
    
    /* Messages with agent labels */
    p:has(strong:first-child) {
        border-left: 4px solid #4caf50;
        padding-left: 12px;
        margin-left: -12px;
    }
    
    /* Input box styling */
    .stChatInputContainer {
        border-radius: 10px;
        border: 2px solid #1976d2;
    }
    
    /* Better code blocks */
    code {
        background-color: #f5f5f5;
        padding: 2px 6px;
        border-radius: 3px;
        font-size: 0.9em;
    }
    
    /* Citation styling */
    em {
        color: #666;
        font-size: 0.9em;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 1.2rem;
        color: #1976d2;
    }
    
    /* Status box styling */
    .stStatus {
        background-color: #e3f2fd;
    }
    
    /* Buttons */
    .stButton button {
        border-radius: 8px;
        font-weight: 500;
    }
    
    /* Error messages */
    .stAlert {
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)


class PostDischargeAssistant:
    """Main application class"""
    
    def __init__(self):
        # Load custom CSS
        load_custom_css()
        
        # Initialize workflow (only once per session)
        if 'workflow' not in st.session_state:
            with st.spinner("🔄 Initializing AI Assistant..."):
                try:
                    st.session_state.workflow = MultiAgentWorkflow()
                    st.session_state.conversation_id = str(uuid.uuid4())
                    st.session_state.messages = []
                    st.session_state.patient_identified = False
                    st.session_state.retry_count = 0
                    logger.info(f"[APP] New session: {st.session_state.conversation_id}")
                    
                    # Add initial greeting
                    greeting = "**Receptionist Agent:** Hello! I'm your post-discharge care assistant. What's your name?"
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": greeting
                    })
                except Exception as e:
                    st.error(f"Failed to initialize AI Assistant: {str(e)}")
                    logger.log_error("PostDischargeAssistant.__init__", e)
    
    def run(self):
        """Run the Streamlit application"""
        
        # Header with logo and title
        col1, col2 = st.columns([1, 5])
        with col1:
            st.markdown("# 🏥")
        with col2:
            st.title("Post-Discharge Medical Assistant")
            st.markdown("*AI-powered assistant for personalized post-discharge care*")
        
        # Sidebar
        self._render_sidebar()
        
        # Main chat interface
        self._render_chat()
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666; font-size: 0.9rem;'>
            💡 Press <strong>Enter</strong> to send message • Press <strong>Shift+Enter</strong> for new line
        </div>
        """, unsafe_allow_html=True)
    
    def _render_sidebar(self):
        """Render sidebar with controls and info"""
        with st.sidebar:
            # Logo section
            st.markdown("### 🏥 About")
            st.info("""
            This AI assistant helps you understand your discharge instructions 
            and answers medical questions using evidence-based information.
            """)
            
            st.divider()
            
            # Features
            st.markdown("### ✨ Features")
            st.markdown("""
            - 👋 **Patient Greeting** - Personalized care
            - 💊 **Discharge Instructions** - Clear summaries
            - 🔍 **Medical Knowledge** - RAG-powered answers
            - 🌐 **Web Search** - Current information
            - 🤖 **Smart Routing** - Specialized agents
            """)
            
            st.divider()
            
            # Controls
            st.markdown("### 🎛️ Controls")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔄 New Chat", use_container_width=True):
                    st.session_state.messages = []
                    st.session_state.conversation_id = str(uuid.uuid4())
                    st.session_state.patient_identified = False
                    st.session_state.retry_count = 0
                    logger.info(f"[APP] New conversation: {st.session_state.conversation_id}")
                    
                    # Add greeting for new chat
                    greeting = "**Receptionist Agent:** Hello! I'm your post-discharge care assistant. What's your name?"
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": greeting
                    })
                    st.rerun()
            
            with col2:
                if st.button("🗑️ Clear", use_container_width=True):
                    if st.session_state.messages:
                        st.session_state.messages = []
                        st.rerun()
            
            st.divider()
            
            # Session stats
            st.markdown("### 📊 Session Info")
            
            message_count = len([m for m in st.session_state.messages if m.get("role") == "user"])
            st.metric("Messages Sent", message_count)
            st.metric("Session ID", st.session_state.conversation_id[:8] + "...")
            
            if st.session_state.patient_identified:
                st.success("✅ Patient Identified")
            else:
                st.info("👤 Awaiting Patient Name")
            
            st.divider()
            
            # Agents status
            st.markdown("### 🤖 AI Agents")
            st.markdown("""
            **Active Agents:**
            - 🏥 Receptionist Agent
            - 🔬 Clinical AI Agent
            """)
            
            st.divider()
            
            # Medical disclaimer
            st.markdown("### ⚠️ Important")
            st.warning("""
            **Medical Disclaimer**
            
            This is an AI assistant for **educational purposes only**. 
            
            Always consult healthcare professionals for medical advice.
            
            🚨 In case of emergency, call 911 or visit the nearest ER.
            """)
            
            st.divider()
            
            # Help section
            with st.expander("❓ How to Use"):
                st.markdown("""
                1. **Introduce yourself** with your name
                2. **Ask questions** about your discharge
                3. **Get medical info** with citations
                4. **Press Enter** to send messages
                5. **Shift+Enter** for multi-line input
                
                **Example Questions:**
                - "What are my medications?"
                - "What foods should I avoid?"
                - "I'm having swelling in my legs"
                - "What's the latest research on kidney disease?"
                """)
            
            with st.expander("🔧 Troubleshooting"):
                st.markdown("""
                **Slow responses?**
                - First query may take 5-7 seconds
                - Medical queries search knowledge base
                - Web searches take 3-5 seconds
                
                **Not understanding?**
                - Try rephrasing your question
                - Be specific about symptoms
                - Mention your diagnosis if relevant
                
                **Agent not routing?**
                - Make sure to ask medical questions
                - Use phrases like "I'm having..." or "What causes..."
                
                **Technical issues?**
                - Click "🔄 New Chat" to restart
                - Refresh page (F5) if problems persist
                """)
    
    def _render_chat(self):
        """Render chat interface"""
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input - CRITICAL: No 'key' parameter to enable Enter key!
        placeholder_text = "💬 Type your message and press Enter to send..."
        
        if prompt := st.chat_input(placeholder_text):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Check if patient name mentioned
            if not st.session_state.patient_identified:
                if self._has_patient_name(prompt):
                    st.session_state.patient_identified = True
            
            # Get assistant response with status updates
            with st.chat_message("assistant"):
                # Determine query type for better status messages
                query_type = self._classify_query(prompt)
                
                try:
                    if query_type == "medical":
                        status = st.status("🔬 Analyzing medical query...", expanded=True)
                        with status:
                            st.write("🔍 Searching medical knowledge base...")
                            st.write("📚 Reviewing nephrology guidelines...")
                            response = self._get_response(prompt)
                            st.write("✅ Response generated with citations")
                        status.update(label="✅ Analysis Complete", state="complete")
                        
                    elif query_type == "patient_data":
                        status = st.status("📋 Retrieving patient information...", expanded=True)
                        with status:
                            st.write("🔍 Searching database...")
                            st.write("📄 Loading discharge report...")
                            response = self._get_response(prompt)
                            st.write("✅ Patient data retrieved")
                        status.update(label="✅ Data Retrieved", state="complete")
                        
                    elif query_type == "web_search":
                        status = st.status("🌐 Searching for current information...", expanded=True)
                        with status:
                            st.write("🔍 Querying web sources...")
                            st.write("📰 Fetching latest medical literature...")
                            response = self._get_response(prompt)
                            st.write("✅ Current information found")
                        status.update(label="✅ Web Search Complete", state="complete")
                        
                    else:
                        with st.spinner("🤔 Processing your message..."):
                            response = self._get_response(prompt)
                    
                    # Display response
                    st.markdown(response)
                    
                    # Add assistant message to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response
                    })
                    
                    # Reset retry count on success
                    st.session_state.retry_count = 0
                    
                except Exception as e:
                    logger.log_error("StreamlitApp._render_chat", e)
                    error_msg = "**Receptionist Agent:** I apologize, but I encountered an error. Please try rephrasing your question or start a new conversation."
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
    
    def _get_response(self, prompt: str) -> str:
        """Get response from workflow with error handling"""
        try:
            response = st.session_state.workflow.process_message(
                prompt,
                st.session_state.conversation_id
            )
            
            # Validate response
            if not response or len(response.strip()) < 10:
                logger.warning("[APP] Empty or invalid response received")
                return "**Receptionist Agent:** I apologize, but I didn't generate a proper response. Could you please rephrase your question?"
            
            return response
            
        except Exception as e:
            logger.log_error("StreamlitApp._get_response", e)
            st.session_state.retry_count = st.session_state.get('retry_count', 0) + 1
            
            if st.session_state.retry_count >= 3:
                return """**Receptionist Agent:** I'm experiencing persistent technical difficulties. Please:
                
1. Click "🔄 New Chat" to restart the conversation
2. If issues persist, refresh the page (F5)
3. For urgent matters, contact your healthcare provider directly

I apologize for the inconvenience."""
            else:
                return "**Receptionist Agent:** I apologize, but I encountered an error processing your request. Please try again or rephrase your question."
    
    def _has_patient_name(self, message: str) -> bool:
        """Check if message contains patient name"""
        indicators = ['my name is', 'i am', "i'm", 'name is', 'this is', 'call me']
        message_lower = message.lower()
        return any(indicator in message_lower for indicator in indicators)
    
    def _classify_query(self, message: str) -> str:
        """Classify query type for status messages"""
        message_lower = message.lower()
        
        # Web search indicators
        web_keywords = [
            'latest', 'recent', 'new', 'current', '2024', '2025',
            'latest research', 'recent studies', 'newest', 'news'
        ]
        if any(keyword in message_lower for keyword in web_keywords):
            return "web_search"
        
        # Medical query
        medical_keywords = [
            'symptom', 'pain', 'swelling', 'dizzy', 'nausea',
            'disease', 'condition', 'treatment', 'medication',
            'what causes', 'why do', 'how does', 'what is',
            'food', 'diet', 'eat', 'avoid', 'should i',
            'worried', 'concerned', 'problem'
        ]
        if any(keyword in message_lower for keyword in medical_keywords):
            return "medical"
        
        # Patient data
        if self._has_patient_name(message):
            return "patient_data"
        
        # General
        return "general"


def main():
    """Main entry point"""
    try:
        app = PostDischargeAssistant()
        app.run()
    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}")
        st.info("💡 Try refreshing the page (F5) or contact support if the issue persists")
        logger.log_error("Main", e)


if __name__ == "__main__":
    main()
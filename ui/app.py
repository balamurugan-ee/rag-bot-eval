"""
Riverside Hospital Chatbot UI - Production Demo
Phase 1: Client-side conversation history with Streamlit
"""
import streamlit as st
import requests
from typing import Dict
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== CONFIGURATION ====================

class Config:
    """Application configuration"""
    API_BASE_URL = "http://localhost:8000"
    CHAT_ENDPOINT = f"{API_BASE_URL}/chat"
    HEALTH_ENDPOINT = f"{API_BASE_URL}/health"
    API_TIMEOUT = 30
    
    # Department colors for visual distinction
    DEPARTMENT_COLORS = {
        "Cardiology": "#ff6b6b",
        "Pediatrics": "#4ecdc4",
        "Orthopedics": "#45b7d1",
        "Dermatology": "#f9ca24",
        "Neurology": "#a29bfe",
        "Ophthalmology": "#fd79a8",
        "Radiology": "#636e72",
        "General Medicine": "#00b894",
        "Billing": "#fdcb6e",
        "Pharmacy": "#6c5ce7"
    }
    
    SAMPLE_QUERIES = [
        "What are cardiology department hours?",
        "I have chest pain, which department should I visit?",
        "I need to schedule an MRI scan",
        "Where is the pharmacy located?",
        "How do I book an appointment?",
        "What are your emergency contact numbers?"
    ]

# ==================== PAGE CONFIGURATION ====================

st.set_page_config(
    page_title="Riverside Hospital Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================

st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Chat message styling */
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    
    /* Department badge */
    .department-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 0.5rem;
        color: white;
    }
    
    /* Emergency banner */
    .emergency-banner {
        background-color: #fee;
        border-left: 4px solid #c00;
        padding: 1rem;
        border-radius: 0.25rem;
        margin-bottom: 1.5rem;
    }
    
    /* Status indicator */
    .status-dot {
        height: 10px;
        width: 10px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 0.5rem;
    }
    
    .status-online {
        background-color: #00b894;
    }
    
    .status-offline {
        background-color: #d63031;
    }
    
    /* Welcome card */
    .welcome-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        margin-bottom: 2rem;
    }
    
    /* Sample query buttons */
    .stButton button {
        width: 100%;
        text-align: left;
    }
</style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE INITIALIZATION ====================

def init_session_state():
    """Initialize all session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "conversation_started" not in st.session_state:
        st.session_state.conversation_started = False
    
    if "backend_healthy" not in st.session_state:
        st.session_state.backend_healthy = None
    
    if "total_queries" not in st.session_state:
        st.session_state.total_queries = 0
    
    if "department_stats" not in st.session_state:
        st.session_state.department_stats = {}
    
    if "session_start_time" not in st.session_state:
        st.session_state.session_start_time = datetime.now()

# ==================== BACKEND COMMUNICATION ====================

@st.cache_data(ttl=60)
def check_backend_health() -> bool:
    """Check if backend API is accessible (cached for 60s)"""
    try:
        response = requests.get(Config.HEALTH_ENDPOINT, timeout=5)
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Backend health check failed: {e}")
        return False

def send_message_to_backend(message: str) -> Dict:
    """Send message to backend API and get response"""
    try:
        response = requests.post(
            Config.CHAT_ENDPOINT,
            json={"message": message},
            timeout=Config.API_TIMEOUT,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        raise Exception("Request timed out. Please try again.")
    except requests.exceptions.ConnectionError:
        raise Exception("Cannot connect to backend. Please ensure the API server is running.")
    except requests.exceptions.HTTPError as e:
        raise Exception(f"API error: {e.response.status_code}")
    except Exception as e:
        raise Exception(f"Unexpected error: {str(e)}")

# ==================== UI COMPONENTS ====================

def get_department_color(department: str) -> str:
    """Get color for department badge"""
    return Config.DEPARTMENT_COLORS.get(department, "#95a5a6")

def render_department_badge(department: str):
    """Render department badge with color coding"""
    if department:
        color = get_department_color(department)
        st.markdown(
            f'<span class="department-badge" style="background-color: {color};">'
            f'📋 {department}</span>',
            unsafe_allow_html=True
        )

def display_emergency_banner():
    """Display emergency contact banner"""
    st.markdown("""
    <div class="emergency-banner">
        <strong>🚨 Emergency?</strong> Call <strong>+91 44 4100 2299</strong> immediately | 
        Main Reception: <strong>+91 44 4100 2200</strong>
    </div>
    """, unsafe_allow_html=True)

def display_welcome_message():
    """Display welcome message for first-time users"""
    st.markdown("""
    <div class="welcome-card">
        <h2>👋 Welcome to Riverside Multispecialty Hospital!</h2>
        <p>I'm your virtual assistant. I can help you with:</p>
        <ul>
            <li>🏥 Department information and timings</li>
            <li>📅 Appointment scheduling guidance</li>
            <li>💊 Pharmacy and medication queries</li>
            <li>💳 Billing and insurance questions</li>
            <li>🔬 Diagnostic services information</li>
            <li>📞 Contact details for specific departments</li>
        </ul>
        <p><strong>Just type your question below to get started!</strong></p>
    </div>
    """, unsafe_allow_html=True)

def display_conversation_stats():
    """Display conversation statistics in sidebar"""
    st.markdown("### 📊 Session Stats")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Queries", st.session_state.total_queries)
    with col2:
        duration = datetime.now() - st.session_state.session_start_time
        minutes = int(duration.total_seconds() / 60)
        st.metric("Duration", f"{minutes}m")
    
    if st.session_state.department_stats:
        st.markdown("**Departments Visited:**")
        sorted_depts = sorted(
            st.session_state.department_stats.items(),
            key=lambda x: x[1],
            reverse=True
        )
        for dept, count in sorted_depts[:5]:
            color = get_department_color(dept)
            st.markdown(
                f'<span style="color: {color};">●</span> {dept}: {count}',
                unsafe_allow_html=True
            )

def export_conversation():
    """Export conversation history as text file"""
    if not st.session_state.messages:
        st.warning("⚠️ No conversation to export")
        return
    
    export_text = f"# Riverside Hospital Chat Export\n"
    export_text += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    export_text += f"Total Queries: {st.session_state.total_queries}\n"
    export_text += "=" * 60 + "\n\n"
    
    for i, msg in enumerate(st.session_state.messages, 1):
        role = "YOU" if msg["role"] == "user" else "ASSISTANT"
        export_text += f"[{i}] {role}:\n{msg['content']}\n"
        if msg.get("department"):
            export_text += f"   → Department: {msg['department']}\n"
        export_text += "\n" + "-" * 60 + "\n\n"
    
    st.download_button(
        label="💾 Download Conversation",
        data=export_text,
        file_name=f"hospital_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
        use_container_width=True
    )

def display_sidebar():
    """Display sidebar with controls and information"""
    with st.sidebar:
        st.title("🏥 Riverside Hospital")
        st.markdown("**Virtual Assistant Demo**")
        st.markdown("---")
        
        # Backend status
        backend_status = st.session_state.backend_healthy
        if backend_status is None:
            st.session_state.backend_healthy = check_backend_health()
            backend_status = st.session_state.backend_healthy
        
        status_emoji = "🟢" if backend_status else "🔴"
        status_text = "Connected" if backend_status else "Disconnected"
        st.markdown(f"**Status:** {status_emoji} {status_text}")
        
        if not backend_status:
            st.error("⚠️ Backend API is not available")
            if st.button("🔄 Retry Connection", use_container_width=True):
                st.cache_data.clear()
                st.session_state.backend_healthy = check_backend_health()
                st.rerun()
        
        st.markdown("---")
        
        # Conversation stats
        if st.session_state.conversation_started:
            display_conversation_stats()
            st.markdown("---")
        
        # Quick actions
        st.markdown("### ⚡ Quick Actions")
        
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_started = False
            st.session_state.total_queries = 0
            st.session_state.department_stats = {}
            st.session_state.session_start_time = datetime.now()
            st.rerun()
        
        if st.session_state.messages:
            export_conversation()
        
        st.markdown("---")
        
        # Contact info
        st.markdown("### 📞 Contact Information")
        st.markdown("""
        **Main Reception:**  
        +91 44 4100 2200
        
        **Emergency (24/7):**  
        +91 44 4100 2299
        
        **Address:**  
        42 Lake View Road  
        Chennai, Tamil Nadu 600028
        """)
        
        st.markdown("---")
        st.caption("Demo Version | Phase 1: Client-Side History")

# ==================== CONVERSATION LOGIC ====================

def update_stats(department: str):
    """Update conversation statistics"""
    st.session_state.total_queries += 1
    if department:
        current_count = st.session_state.department_stats.get(department, 0)
        st.session_state.department_stats[department] = current_count + 1

def process_user_message(user_input: str):
    """Process user message and get bot response"""
    # Add user message to history
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "timestamp": datetime.now().isoformat()
    })
    
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            try:
                # Call backend API
                response_data = send_message_to_backend(user_input)
                
                # Extract response and department
                assistant_message = response_data.get("response", "")
                department = response_data.get("department", "")
                
                # Display response
                st.markdown(assistant_message)
                render_department_badge(department)
                
                # Add to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": assistant_message,
                    "department": department,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Update stats
                update_stats(department)
                st.session_state.conversation_started = True
                
            except Exception as e:
                error_message = f"❌ **Error:** {str(e)}"
                st.error(error_message)
                
                # Add error to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message,
                    "timestamp": datetime.now().isoformat(),
                    "is_error": True
                })
                
                logger.error(f"Error processing message: {e}")

# ==================== MAIN APPLICATION ====================

def main():
    """Main application entry point"""
    # Initialize session state
    init_session_state()
    
    # Display sidebar
    display_sidebar()
    
    # Main content area
    # Title and Clear Chat button in columns
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title("🏥 Riverside Hospital Virtual Assistant")
    with col2:
        st.write("")  # Spacer for alignment
        if st.button("🗑️ Clear Chat", key="clear_chat_main", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_started = False
            st.session_state.total_queries = 0
            st.session_state.department_stats = {}
            st.session_state.session_start_time = datetime.now()
            st.rerun()
    
    st.markdown("Ask me anything about our hospital services, departments, timings, and more!")
    
    # Emergency banner
    display_emergency_banner()
    
    # Check backend health
    if not st.session_state.backend_healthy:
        st.error("🔌 **Backend API is not available.** Please start the server first:")
        st.code("uvicorn main:app --reload", language="bash")
        st.info("The backend should be running at http://localhost:8000")
        st.stop()
    
    # Display conversation history
    for message in st.session_state.messages:
        is_user = message["role"] == "user"
        is_error = message.get("is_error", False)
        
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if not is_user and not is_error and message.get("department"):
                render_department_badge(message["department"])
    
    # Welcome message for new users
    if not st.session_state.conversation_started and not st.session_state.messages:
        display_welcome_message()
    
    # Handle pending query from sample buttons
    if hasattr(st.session_state, 'pending_query'):
        query = st.session_state.pending_query
        del st.session_state.pending_query
        process_user_message(query)
        st.rerun()
    
    # Quick actions before FAQs - shown only if there's conversation history
    if st.session_state.messages:
        st.markdown("---")
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            if st.button("🗑️ Clear Conversation", key="clear_chat_before_faq", use_container_width=True, type="secondary"):
                st.session_state.messages = []
                st.session_state.conversation_started = False
                st.session_state.total_queries = 0
                st.session_state.department_stats = {}
                st.session_state.session_start_time = datetime.now()
                st.rerun()
    
    # Display FAQ buttons above chat input
    st.markdown("### 💡 Frequently Asked Questions")
    st.markdown("Click any question below to quickly get started:")
    cols = st.columns(2)
    for idx, query in enumerate(Config.SAMPLE_QUERIES):
        with cols[idx % 2]:
            if st.button(f"💬 {query}", key=f"faq_main_{idx}", use_container_width=True):
                st.session_state.pending_query = query
                st.rerun()
    
    st.markdown("---")
    
    # Chat input
    user_input = st.chat_input(
        "Type your question here...",
        key="chat_input"
    )
    
    if user_input:
        process_user_message(user_input)
        st.rerun()

# ==================== ENTRY POINT ====================

if __name__ == "__main__":
    main()

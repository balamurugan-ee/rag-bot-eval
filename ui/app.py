"""
Riverside Hospital Chatbot UI - Production Demo
Phase 1: Client-side conversation history with Streamlit

Two independent backend flows, one screen, one question, two buttons:
  - "Classify" -> POST /classify -> department routing only
  - "Ask"      -> POST /chat     -> RAG-grounded answer only
Classification and RAG generation are deliberately decoupled in the backend,
so the UI lets the user explicitly choose which one to run per query instead
of always calling both together.
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
    CLASSIFY_ENDPOINT = f"{API_BASE_URL}/classify"
    HEALTH_ENDPOINT = f"{API_BASE_URL}/health"
    API_TIMEOUT = 30

    # Professional, muted department palette -- desaturated navy/teal/slate
    # family rather than bright/candy colors, appropriate for clinical software.
    DEPARTMENT_COLORS = {
        "Cardiology": "#1F4E5F",
        "Pediatrics": "#2E7D6B",
        "Orthopedics": "#4A6FA5",
        "Dermatology": "#A67C52",
        "Neurology": "#6B5B95",
        "Ophthalmology": "#5C7A89",
        "Radiology": "#495867",
        "General Medicine": "#3A6351",
        "Billing": "#8B7355",
        "Pharmacy": "#4F5D75",
    }

    # Equal Experts brand tokens, pulled from the live computed styles at
    # equalexperts.com (Lexend typeface, sky-blue accent, navy secondary,
    # near-black header band, flat 0px-radius corners throughout).
    BRAND_BLUE = "#1795D4"    # primary accent / CTA (their H2s, "Search Site" button)
    BRAND_NAVY = "#22567C"    # secondary / links
    BRAND_DARK = "#212526"    # header/nav band
    BRAND_GRAY_BG = "#F4F4F4" # light section background
    BRAND_TEXT = "#545454"    # body copy

    PRIMARY_COLOR = BRAND_BLUE

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

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Lexend:wght@300;400;500;600;700&display=swap');

    /* Scoped to actual text containers only -- NOT a blanket span/div rule.
       Streamlit renders its chat-avatar icons as ligature text ("face",
       "smart_toy") in a Material Symbols icon font; an earlier version of
       this stylesheet forced Lexend onto every <span> with !important and
       broke those icons into literal visible words. Text still inherits
       Lexend from body/.stMarkdown; icon elements are untouched. */
    html, body, .stMarkdown, .stCaption, p, li, label {{
        font-family: 'Lexend', sans-serif;
        color: {Config.BRAND_TEXT};
    }}

    h1, h2, h3, h4 {{
        font-family: 'Lexend', sans-serif;
        font-weight: 500;
    }}

    .stButton button, .stDownloadButton button, .stTextInput input {{
        font-family: 'Lexend', sans-serif;
    }}

    /* Main container */
    .main {{
        padding: 0rem 1rem;
    }}

    [data-testid="stAppViewContainer"] {{
        background-color: #FFFFFF;
    }}

    /* Header band -- mirrors the dark nav/hero band on equalexperts.com */
    .ee-header-band {{
        background-color: {Config.BRAND_DARK};
        padding: 2rem 2.5rem;
        margin: 0 0 1.5rem 0;
    }}
    /* Plain divs, not h1/p -- avoids fighting Streamlit's own (higher-
       specificity, !important) heading color rules, which caused the
       low-contrast gray-on-navy title seen earlier. */
    .ee-header-title {{
        color: #FFFFFF;
        font-family: 'Lexend', sans-serif;
        font-weight: 400;
        font-size: 2rem;
        line-height: 1.3;
    }}
    .ee-header-subtitle {{
        color: #C9CED0;
        font-family: 'Lexend', sans-serif;
        margin-top: 0.4rem;
        font-size: 1rem;
    }}

    /* Chat message styling -- flat, squared, bordered instead of shadowed */
    .stChatMessage {{
        padding: 1rem;
        border-radius: 0px;
        border: 1px solid #E5E5E5;
        margin-bottom: 1rem;
    }}

    /* Department badge -- flat, brand accent, squared corners */
    .department-badge {{
        display: inline-block;
        padding: 0.3rem 0.85rem;
        border-radius: 0px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-top: 0.4rem;
        color: white;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }}

    /* Emergency banner -- keeps semantic red (urgency), flat corners */
    .emergency-banner {{
        background-color: #FDF2F2;
        border-left: 4px solid #A93226;
        padding: 1rem 1.25rem;
        border-radius: 0px;
        margin-bottom: 1.5rem;
        color: #641E16;
    }}

    /* Status indicator */
    .status-dot {{
        height: 10px;
        width: 10px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 0.5rem;
    }}
    .status-online {{ background-color: #27AE60; }}
    .status-offline {{ background-color: #C0392B; }}

    /* Welcome card -- flat, bordered, brand accent */
    .welcome-card {{
        background-color: {Config.BRAND_GRAY_BG};
        border-left: 4px solid {Config.BRAND_BLUE};
        color: {Config.BRAND_TEXT};
        padding: 1.5rem 2rem;
        border-radius: 0px;
        margin-bottom: 2rem;
    }}
    .welcome-card h2 {{
        color: {Config.BRAND_NAVY};
        margin-top: 0;
        font-weight: 500;
    }}

    /* Buttons -- flat, squared, outlined navy by default; the one primary
       CTA on screen ("Ask") gets the solid brand-blue fill, matching the
       single-accent-CTA pattern used on equalexperts.com */
    .stButton button, .stDownloadButton button {{
        border-radius: 0px;
        font-weight: 500;
        border: 1.5px solid {Config.BRAND_NAVY};
        background-color: #FFFFFF;
        color: {Config.BRAND_NAVY};
        padding: 0.6rem 1rem;
        transition: background-color 0.15s ease, color 0.15s ease;
    }}
    .stButton button:hover, .stDownloadButton button:hover {{
        background-color: {Config.BRAND_NAVY};
        color: #FFFFFF;
    }}
    button[kind="primary"] {{
        background-color: {Config.BRAND_BLUE} !important;
        border-color: {Config.BRAND_BLUE} !important;
        color: #FFFFFF !important;
    }}
    button[kind="primary"]:hover {{
        background-color: {Config.BRAND_NAVY} !important;
        border-color: {Config.BRAND_NAVY} !important;
    }}

    /* Sample query (FAQ) buttons -- left-aligned, plain text look */
    .stButton button {{
        width: 100%;
        text-align: left;
    }}

    /* Text input -- flat, squared, brand-blue focus ring */
    .stTextInput input {{
        border-radius: 0px !important;
        border: 1.5px solid #CCCCCC !important;
    }}
    .stTextInput input:focus {{
        border-color: {Config.BRAND_BLUE} !important;
        box-shadow: 0 0 0 1px {Config.BRAND_BLUE} !important;
    }}

    /* Sidebar -- dark band, mirrors EE's nav bar */
    [data-testid="stSidebar"] {{
        background-color: {Config.BRAND_DARK};
    }}
    [data-testid="stSidebar"] * {{
        color: #FFFFFF !important;
    }}
    [data-testid="stSidebar"] hr {{
        border-color: rgba(255, 255, 255, 0.15);
    }}
    [data-testid="stSidebar"] .stButton button {{
        background-color: transparent;
        border: 1.5px solid #FFFFFF;
        color: #FFFFFF;
    }}
    [data-testid="stSidebar"] .stButton button:hover {{
        background-color: {Config.BRAND_BLUE};
        border-color: {Config.BRAND_BLUE};
        color: #FFFFFF;
    }}
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

    if "input_key_counter" not in st.session_state:
        st.session_state.input_key_counter = 0

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

def get_department(message: str) -> str:
    """
    Flow 1: classification only. Calls /classify, which is a separate concern
    from /chat's RAG answer -- the backend does not return department from /chat.
    """
    try:
        response = requests.post(
            Config.CLASSIFY_ENDPOINT,
            json={"message": message},
            timeout=Config.API_TIMEOUT,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json().get("department", "")
    except Exception as e:
        logger.error(f"Classification request failed: {e}")
        return ""

def send_message_to_backend(message: str) -> Dict:
    """
    Flow 2: RAG answer only. Calls /chat, which returns {"response": ...}
    with no department field.
    """
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
    return Config.DEPARTMENT_COLORS.get(department, "#7F8C8D")

def render_department_badge(department: str):
    """Render department badge with color coding"""
    if department:
        color = get_department_color(department)
        st.markdown(
            f'<span class="department-badge" style="background-color: {color};">'
            f'{department}</span>',
            unsafe_allow_html=True
        )

def display_emergency_banner():
    """Display emergency contact banner"""
    st.markdown("""
    <div class="emergency-banner">
        <strong>Emergency?</strong> Call <strong>+91 44 4100 2299</strong> immediately &middot;
        Main Reception: <strong>+91 44 4100 2200</strong>
    </div>
    """, unsafe_allow_html=True)

def display_welcome_message():
    """Display welcome message for first-time users"""
    st.markdown("""
    <div class="welcome-card">
        <h2>Welcome to Riverside Multispecialty Hospital</h2>
        <p>I'm your virtual assistant. I can help you with:</p>
        <ul>
            <li>Department information and timings</li>
            <li>Appointment scheduling guidance</li>
            <li>Pharmacy and medication queries</li>
            <li>Billing and insurance questions</li>
            <li>Diagnostic services information</li>
            <li>Contact details for specific departments</li>
        </ul>
        <p><strong>Type your question below, then choose <em>Classify</em> to find the right
        department, or <em>Ask</em> for a detailed answer.</strong></p>
    </div>
    """, unsafe_allow_html=True)

def display_conversation_stats():
    """Display conversation statistics in sidebar"""
    st.markdown("### Session Stats")

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
        st.warning("No conversation to export")
        return

    export_text = f"# Riverside Hospital Chat Export\n"
    export_text += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    export_text += f"Total Queries: {st.session_state.total_queries}\n"
    export_text += "=" * 60 + "\n\n"

    for i, msg in enumerate(st.session_state.messages, 1):
        role = "YOU" if msg["role"] == "user" else "ASSISTANT"
        action_label = {"classify": "Classify", "ask": "Ask"}.get(msg.get("action"), "")
        export_text += f"[{i}] {role} ({action_label}):\n{msg.get('content') or ''}\n"
        if msg.get("department"):
            export_text += f"   -> Department: {msg['department']}\n"
        export_text += "\n" + "-" * 60 + "\n\n"

    st.download_button(
        label="Download Conversation",
        data=export_text,
        file_name=f"hospital_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
        use_container_width=True
    )

def clear_conversation():
    """Reset all conversation-related session state"""
    st.session_state.messages = []
    st.session_state.conversation_started = False
    st.session_state.total_queries = 0
    st.session_state.department_stats = {}
    st.session_state.session_start_time = datetime.now()

def display_sidebar():
    """Display sidebar with controls and information"""
    with st.sidebar:
        st.title("Riverside Hospital")
        st.markdown("**Virtual Assistant Demo**")
        st.markdown("---")

        # Backend status
        backend_status = st.session_state.backend_healthy
        if backend_status is None:
            st.session_state.backend_healthy = check_backend_health()
            backend_status = st.session_state.backend_healthy

        status_class = "status-online" if backend_status else "status-offline"
        status_text = "Connected" if backend_status else "Disconnected"
        st.markdown(
            f'<span class="status-dot {status_class}"></span>**Status:** {status_text}',
            unsafe_allow_html=True
        )

        if not backend_status:
            st.error("Backend API is not available")
            if st.button("Retry Connection", use_container_width=True):
                st.cache_data.clear()
                st.session_state.backend_healthy = check_backend_health()
                st.rerun()

        st.markdown("---")

        # Conversation stats
        if st.session_state.conversation_started:
            display_conversation_stats()
            st.markdown("---")

        # Quick actions
        st.markdown("### Quick Actions")

        if st.button("Clear Chat", use_container_width=True):
            clear_conversation()
            st.rerun()

        if st.session_state.messages:
            export_conversation()

        st.markdown("---")

        # Contact info
        st.markdown("### Contact Information")
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

def update_stats(department: str = None):
    """Update conversation statistics"""
    st.session_state.total_queries += 1
    if department:
        current_count = st.session_state.department_stats.get(department, 0)
        st.session_state.department_stats[department] = current_count + 1

def process_classify_action(query: str):
    """
    User pressed 'Classify': run /classify only, store the result. No inline
    rendering here -- the caller reruns immediately after, and rendering
    happens uniformly (for both the latest turn and history) in main().
    """
    st.session_state.messages.append({
        "role": "user",
        "content": query,
        "action": "classify",
        "timestamp": datetime.now().isoformat()
    })

    with st.spinner("Classifying..."):
        try:
            department = get_department(query)
            st.session_state.messages.append({
                "role": "assistant",
                "content": None,
                "action": "classify",
                "department": department,
                "timestamp": datetime.now().isoformat()
            })
            update_stats(department)
            st.session_state.conversation_started = True

        except Exception as e:
            error_message = f"**Error:** {str(e)}"
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_message,
                "action": "classify",
                "timestamp": datetime.now().isoformat(),
                "is_error": True
            })
            logger.error(f"Error processing classify action: {e}")

def process_ask_action(query: str):
    """User pressed 'Ask': run /chat only, store the result (see docstring above)."""
    st.session_state.messages.append({
        "role": "user",
        "content": query,
        "action": "ask",
        "timestamp": datetime.now().isoformat()
    })

    with st.spinner("Thinking..."):
        try:
            response_data = send_message_to_backend(query)
            assistant_message = response_data.get("response", "")
            st.session_state.messages.append({
                "role": "assistant",
                "content": assistant_message,
                "action": "ask",
                "timestamp": datetime.now().isoformat()
            })
            update_stats()
            st.session_state.conversation_started = True

        except Exception as e:
            error_message = f"**Error:** {str(e)}"
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_message,
                "action": "ask",
                "timestamp": datetime.now().isoformat(),
                "is_error": True
            })
            logger.error(f"Error processing ask action: {e}")

def render_message_pair(user_msg: Dict, assistant_msg: Dict):
    """Render one user turn + its assistant result as a pair of chat bubbles."""
    with st.chat_message("user"):
        action_tag = {"classify": "🏷️ Classify", "ask": "💬 Ask"}.get(user_msg.get("action"), "")
        if action_tag:
            st.caption(action_tag)
        st.markdown(user_msg["content"])

    with st.chat_message("assistant"):
        if assistant_msg.get("is_error"):
            st.markdown(assistant_msg["content"])
        elif assistant_msg.get("action") == "classify":
            st.markdown("**Department:**")
            render_department_badge(assistant_msg.get("department"))
        else:
            st.markdown(assistant_msg["content"])

# ==================== MAIN APPLICATION ====================

def main():
    """Main application entry point"""
    # Initialize session state
    init_session_state()

    # Display sidebar
    display_sidebar()

    # Main content area -- dark header band, mirrors equalexperts.com's nav/hero
    st.markdown("""
    <div class="ee-header-band">
        <div class="ee-header-title">Riverside Hospital Virtual Assistant</div>
        <div class="ee-header-subtitle">Ask me anything about our hospital services, departments, timings, and more.</div>
    </div>
    """, unsafe_allow_html=True)

    # Emergency banner
    display_emergency_banner()

    # Check backend health
    if not st.session_state.backend_healthy:
        st.error("**Backend API is not available.** Please start the server first:")
        st.code("uvicorn main:app --reload", language="bash")
        st.info("The backend should be running at http://localhost:8000")
        st.stop()

    # Single question input, paired with two explicit action buttons. Kept
    # at the top of the page (not below a growing history) so it's always
    # in the same place and the result of the latest action always renders
    # directly below it -- no scrolling to find either.
    #
    # The widget key increments after every action so the box starts empty
    # again on the next run (Streamlit disallows mutating a widget's bound
    # session_state key after it has already been instantiated this run,
    # so a fresh key is the simplest safe way to reset it).
    input_key = f"query_input_{st.session_state.input_key_counter}"
    query = st.text_input(
        "Your question",
        key=input_key,
        placeholder="e.g. What are cardiology department hours?",
        label_visibility="collapsed"
    )

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        classify_clicked = st.button("Classify", use_container_width=True)
    with btn_col2:
        ask_clicked = st.button("Ask", type="primary", use_container_width=True)

    if classify_clicked and query.strip():
        process_classify_action(query.strip())
        st.session_state.input_key_counter += 1
        st.rerun()
    elif ask_clicked and query.strip():
        process_ask_action(query.strip())
        st.session_state.input_key_counter += 1
        st.rerun()

    st.markdown("---")

    # Latest result -- rendered right here, directly below the input, every
    # time. Full history moves into a collapsed expander further down so it
    # never pushes the input (or the latest answer) out of view.
    messages = st.session_state.messages
    if messages:
        st.markdown("#### Latest")
        render_message_pair(messages[-2], messages[-1])

        earlier = messages[:-2]
        if earlier:
            with st.expander(f"Earlier in this session ({len(earlier) // 2} more)"):
                for i in range(0, len(earlier) - 1, 2):
                    render_message_pair(earlier[i], earlier[i + 1])
    else:
        display_welcome_message()

    # FAQ buttons -- tucked into an expander so they don't compete with the
    # input for attention; clicking one only fills the box, the user still
    # chooses which action (Classify / Ask) to run on it.
    with st.expander("Sample questions"):
        cols = st.columns(2)
        for idx, sample_query in enumerate(Config.SAMPLE_QUERIES):
            with cols[idx % 2]:
                if st.button(sample_query, key=f"faq_main_{idx}", use_container_width=True):
                    st.session_state[f"query_input_{st.session_state.input_key_counter}"] = sample_query
                    st.rerun()

# ==================== ENTRY POINT ====================

if __name__ == "__main__":
    main()

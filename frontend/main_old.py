"""
MedSync AI - Unified Multimodal Surgical Workflow
Advanced Streamlit Dashboard with Medical Image Analysis, Pre-Op Briefing, and Agentic RAG
"""

import streamlit as st
import requests
import json
import threading
import time
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from concurrent.futures import ThreadPoolExecutor
import io
import base64

# Medical imaging imports
from PIL import Image

# Import multimodal processor module
try:
    from multimodal_processor import (
        MedicalFileProcessor, AnatomicalAnalyzer, PreOpBriefGenerator,
        initialize_multimodal_session_state, get_file_hash
    )
    MULTIMODAL_AVAILABLE = True
except ImportError:
    MULTIMODAL_AVAILABLE = False
    st.warning("Multimodal processing module not fully available. Install required packages.")

# Configure page layout
st.set_page_config(
    page_title="MedSync AI - Multimodal Surgical Intelligence",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CUSTOM CSS FOR DARK MODE THEME
# ============================================================================

CUSTOM_CSS = """
<style>
    :root {
        --primary-color: #00D9FF;
        --secondary-color: #FF006E;
        --success-color: #06FFA5;
        --warning-color: #FFB400;
        --danger-color: #FF006E;
        --dark-bg: #0A0E27;
        --darker-bg: #05070F;
        --surface: #1A1F3A;
        --surface-light: #252D47;
        --text-primary: #E0E7FF;
        --text-secondary: #A0AEC0;
        --border-color: #2D3B5C;
    }

    body {
        background-color: var(--dark-bg);
        color: var(--text-primary);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    .main {
        background-color: var(--dark-bg);
    }

    .sidebar .sidebar-content {
        background-color: var(--darker-bg);
    }

    .metric-card {
        background: linear-gradient(135deg, var(--surface) 0%, var(--surface-light) 100%);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0, 217, 255, 0.1);
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        border-color: var(--primary-color);
        box-shadow: 0 4px 25px rgba(0, 217, 255, 0.2);
        transform: translateY(-2px);
    }

    .status-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .status-ok {
        background-color: rgba(6, 255, 165, 0.15);
        color: var(--success-color);
        border: 1px solid var(--success-color);
    }

    .status-warning {
        background-color: rgba(255, 180, 0, 0.15);
        color: var(--warning-color);
        border: 1px solid var(--warning-color);
    }

    .status-danger {
        background-color: rgba(255, 0, 110, 0.15);
        color: var(--danger-color);
        border: 1px solid var(--danger-color);
    }

    .section-header {
        color: var(--primary-color);
        border-bottom: 2px solid var(--primary-color);
        padding-bottom: 10px;
        margin: 20px 0 15px 0;
        font-size: 1.3em;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color) 0%, #0099FF 100%);
        color: var(--darker-bg);
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .stButton > button:hover {
        box-shadow: 0 0 20px rgba(0, 217, 255, 0.4);
        transform: translateY(-2px);
    }

    .preop-brief {
        background: linear-gradient(135deg, rgba(0, 217, 255, 0.1) 0%, rgba(255, 0, 110, 0.1) 100%);
        border-left: 4px solid var(--primary-color);
        border-radius: 8px;
        padding: 20px;
        margin: 15px 0;
        font-size: 0.95em;
        line-height: 1.6;
    }

    .alert-container {
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid;
    }

    .alert-high {
        background-color: rgba(255, 0, 110, 0.1);
        border-left-color: var(--danger-color);
        color: var(--danger-color);
    }

    .alert-medium {
        background-color: rgba(255, 180, 0, 0.1);
        border-left-color: var(--warning-color);
        color: var(--warning-color);
    }

    .alert-low {
        background-color: rgba(6, 255, 165, 0.1);
        border-left-color: var(--success-color);
        color: var(--success-color);
    }

    .chat-container {
        height: 400px;
        overflow-y: auto;
        padding: 15px;
        background-color: var(--surface);
        border-radius: 12px;
        border: 1px solid var(--border-color);
        margin-bottom: 15px;
    }

    .chat-message {
        margin: 10px 0;
        padding: 12px 15px;
        border-radius: 8px;
        max-width: 85%;
    }

    .chat-user {
        margin-left: auto;
        background: linear-gradient(135deg, var(--primary-color) 0%, #0099FF 100%);
        color: var(--darker-bg);
        text-align: right;
        font-weight: 500;
    }

    .chat-assistant {
        background-color: var(--surface-light);
        border-left: 3px solid var(--primary-color);
        color: var(--text-primary);
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
SYSTEM_PING_INTERVAL = 5  # seconds
MAX_WORKERS = 3

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

# Initialize baseline session state
if "patient_data" not in st.session_state:
    st.session_state.patient_data = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "roadmap_data" not in st.session_state:
    st.session_state.roadmap_data = None
if "current_step" not in st.session_state:
    st.session_state.current_step = 0
if "medical_analysis" not in st.session_state:
    st.session_state.medical_analysis = None
if "system_status" not in st.session_state:
    st.session_state.system_status = None

# Multimodal session state
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = {}
if "file_analyses" not in st.session_state:
    st.session_state.file_analyses = {}
if "anatomical_findings" not in st.session_state:
    st.session_state.anatomical_findings = None
if "preop_brief" not in st.session_state:
    st.session_state.preop_brief = None
if "agentic_rag_history" not in st.session_state:
    st.session_state.agentic_rag_history = []

# ============================================================================
# SYSTEM STATUS MONITORING (Real-time Pinging)
# ============================================================================

class SystemMonitor:
    """Monitor backend system status with real-time pinging"""
    
    def __init__(self, api_url: str, ping_interval: int = 5):
        self.api_url = api_url
        self.ping_interval = ping_interval
        self.last_status = None
        self.last_check = None
        self.monitor_thread = None
        self.running = False
    
    def check_health(self) -> Dict:
        """Check backend health"""
        try:
            response = requests.get(
                f"{self.api_url}/health",
                timeout=3
            )
            if response.status_code == 200:
                return response.json()
        except:
            pass
        
        return {
            "status": "offline",
            "llm_available": False,
            "vector_db_collections": {"clinical": 0, "surgical": 0}
        }
    
    def start_monitoring(self):
        """Start background monitoring"""
        if not self.running:
            self.running = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.running:
            self.last_status = self.check_health()
            self.last_check = datetime.now()
            time.sleep(self.ping_interval)
    
    def get_status(self) -> Dict:
        """Get current status"""
        if self.last_status is None:
            self.last_status = self.check_health()
            self.last_check = datetime.now()
        return self.last_status

# Initialize system monitor
system_monitor = SystemMonitor(API_BASE_URL, SYSTEM_PING_INTERVAL)
system_monitor.start_monitoring()

# ============================================================================
# API COMMUNICATION FUNCTIONS
# ============================================================================

def register_patient(patient_data: Dict) -> Dict:
    """Register a patient with the backend"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/patient/register",
            json=patient_data,
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Patient registration error: {e}")
        st.error(f"Failed to register patient: {str(e)}")
        return None

def query_surgeon_assistant(query: str, patient_id: str, context: Optional[str] = None) -> Dict:
    """Send a query to the surgeon assistant with agentic RAG"""
    try:
        payload = {
            "query": query,
            "patient_id": patient_id
        }
        if context:
            payload["context"] = context
        
        response = requests.post(
            f"{API_BASE_URL}/api/surgeon/query",
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Surgeon query error: {e}")
        st.error(f"Failed to process query: {str(e)}")
        return None

def analyze_medical_file(file_data: bytes, file_type: str) -> Dict:
    """Send medical file to backend for analysis"""
    try:
        files = {
            'file': ("medical_file", io.BytesIO(file_data)),
            'file_type': (None, file_type)
        }
        response = requests.post(
            f"{API_BASE_URL}/api/analyze-medical-file",
            files=files,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Medical file analysis error: {e}")
        st.error(f"Failed to analyze file: {str(e)}")
        return None

# ============================================================================
# UI COMPONENTS - MULTIMODAL INTAKE
# ============================================================================

def render_multimodal_sidebar():
    """Render multimodal file intake in sidebar"""
    with st.sidebar:
        st.markdown(
            "<h3 style='color: #00D9FF; margin-top: 20px;'>📁 Medical Files</h3>",
            unsafe_allow_html=True
        )
        
        uploaded_file = st.file_uploader(
            "Upload medical file (PDF, DICOM, PNG, JPG)",
            type=['pdf', 'dcm', 'png', 'jpg', 'jpeg'],
            key="medical_file_uploader",
            help="Upload clinical reports (PDF), surgical plans, or medical imaging"
        )
        
        if uploaded_file is not None:
            if MULTIMODAL_AVAILABLE:
                is_valid, error_msg = MedicalFileProcessor.validate_file(uploaded_file)
                
                if not is_valid:
                    st.error(f"File validation failed: {error_msg}")
                else:
                    file_type = MedicalFileProcessor.get_file_type(uploaded_file.name)
                    
                    # Display file info
                    st.write(f"📄 **File**: {uploaded_file.name}")
                    st.write(f"📊 **Type**: {file_type.upper()}")
                    st.write(f"💾 **Size**: {uploaded_file.size / 1024:.1f} KB")
                    
                    # Generate file hash for caching
                    file_hash = get_file_hash(uploaded_file)
                    
                    # Store in session state
                    st.session_state.uploaded_files[file_hash] = {
                        'name': uploaded_file.name,
                        'type': file_type,
                        'timestamp': datetime.now().isoformat(),
                        'data': uploaded_file.getvalue()
                    }
                    
                    # Process based on file type
                    if file_type == 'pdf':
                        st.info("📄 PDF clinical report uploaded")
                        # Extract and analyze text
                        pdf_text = MedicalFileProcessor.extract_pdf_text(uploaded_file)
                        st.session_state.file_analyses[file_hash] = {
                            'type': 'pdf',
                            'content': pdf_text[:500],
                            'full_text': pdf_text
                        }
                    
                    elif file_type == 'dicom':
                        st.success("🖼️ DICOM image loaded")
                        st.info("Ready for anatomical analysis")
                    
                    elif file_type == 'image':
                        st.success("🖼️ Medical image uploaded")
                        # Display thumbnail
                        image = MedicalFileProcessor.load_image(uploaded_file)
                        if image:
                            st.image(image, use_column_width=True, caption="Uploaded Medical Image")
            else:
                st.error("Multimodal processing module not available. Install required dependencies.")

def render_system_status_monitor():
    """Render real-time system status monitor"""
    with st.sidebar:
        st.markdown("---")
        st.markdown(
            "<h3 style='color: #00D9FF;'>🔔 System Status</h3>",
            unsafe_allow_html=True
        )
        
        # Get current status
        status = system_monitor.get_status()
        
        # LLM Status
        llm_status = "✅ Online" if status.get("llm_available") else "⚠️ Offline"
        llm_color = "#06FFA5" if status.get("llm_available") else "#FFB400"
        
        st.markdown(
            f"""
            <div style='padding: 10px; background: rgba(0,217,255,0.1); border-radius: 8px; border-left: 3px solid {llm_color};'>
                <p style='margin: 0; font-size: 0.85em; color: #A0AEC0;'>LLM Engine</p>
                <p style='margin: 5px 0; color: {llm_color}; font-weight: bold;'>{llm_status}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Vector DB Status
        collections = status.get("vector_db_collections", {})
        clinical_count = collections.get("clinical", 0)
        
        st.markdown(
            f"""
            <div style='padding: 10px; background: rgba(0,217,255,0.1); border-radius: 8px; border-left: 3px solid #00D9FF; margin-top: 10px;'>
                <p style='margin: 0; font-size: 0.85em; color: #A0AEC0;'>Vector Database</p>
                <p style='margin: 5px 0; color: #00D9FF; font-weight: bold;'>{clinical_count} documents</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Last check
        if system_monitor.last_check:
            check_time = system_monitor.last_check.strftime("%H:%M:%S")
            st.caption(f"Last checked: {check_time}")

# ============================================================================
# UI COMPONENTS - PRE-OP BRIEFING
# ============================================================================

def render_preop_brief(patient_data: Dict, uploaded_file_data: Optional[Dict] = None):
    """Render Pre-Op Brief section"""
    st.markdown(
        "<h3 class='section-header'>🏥 Pre-Operative Brief</h3>",
        unsafe_allow_html=True
    )
    
    if not MULTIMODAL_AVAILABLE:
        st.error("Multimodal processor required. Install dependencies first.")
        return
    
    # Prepare anatomical findings
    anatomical_findings = {
        'liver_segments': [],
        'vascular_involvement': [],
        'surgical_complexity': 'Standard'
    }
    
    # Extract from file analysis if available
    if uploaded_file_data:
        for file_hash, analysis in st.session_state.file_analyses.items():
            if analysis['type'] == 'pdf':
                anatomical_findings = AnatomicalAnalyzer.identify_anatomical_findings(
                    analysis.get('full_text', '')
                )
    
    # Generate Pre-Op Brief
    clinical_text = ""
    if st.session_state.file_analyses:
        first_key = list(st.session_state.file_analyses.keys())[0]
        clinical_text = st.session_state.file_analyses[first_key].get('full_text', '')
    
    preop_brief = PreOpBriefGenerator.generate_brief(
        patient_data,
        clinical_text,
        anatomical_findings
    )
    
    # Store in session
    st.session_state.preop_brief = preop_brief
    st.session_state.anatomical_findings = anatomical_findings
    
    # Display Brief
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(
            f"""
            <div class='preop-brief'>
            <h4 style='margin-top: 0;'>PATIENT SUMMARY</h4>
            <b>ID:</b> {preop_brief['patient_summary']['id']}<br>
            <b>Age:</b> {preop_brief['patient_summary']['age']} years<br>
            <b>Risk Factors:</b> {len(preop_brief['patient_summary']['risk_factors'])} identified
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        complexity_color = "#FF006E" if anatomical_findings['surgical_complexity'] == 'Complex' else "#06FFA5"
        st.markdown(
            f"""
            <div style='background: rgba(255,0,110,0.1); border-radius: 8px; padding: 15px; border-left: 4px solid {complexity_color};'>
                <p style='margin: 0; font-size: 0.9em; color: #A0AEC0;'>Complexity</p>
                <p style='margin: 5px 0; color: {complexity_color}; font-weight: bold; font-size: 1.2em;'>
                    {anatomical_findings['surgical_complexity'].upper()}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Detailed briefing
    with st.expander("📋 Detailed Clinical Summary"):
        st.markdown(preop_brief['clinical_summary'])
    
    with st.expander("🔍 Anatomical Findings"):
        st.markdown(preop_brief['anatomical_summary'])
    
    with st.expander("🛠️ Procedure Plan"):
        st.markdown(preop_brief['procedure_plan'])
    
    # Critical alerts
    if preop_brief['critical_alerts']:
        st.markdown("<h4 style='color: #FF006E;'>⚠️ CRITICAL ALERTS</h4>", unsafe_allow_html=True)
        for alert in preop_brief['critical_alerts']:
            st.markdown(
                f"""
                <div class='alert-container alert-high'>
                    <strong>🔴 {alert['alert']}</strong><br>
                    <em>Action:</em> {alert['action']}
                </div>
                """,
                unsafe_allow_html=True
            )
    
    # Equipment prep
    st.markdown("<h4>✓ Equipment Preparation</h4>", unsafe_allow_html=True)
    for item in preop_brief['equipment_prep']:
        st.markdown(f"<p style='margin: 5px 0;'>{item}</p>", unsafe_allow_html=True)

# ============================================================================
# UI COMPONENTS - SEQUENTIAL SURGICAL ROADMAP
# ============================================================================

def render_sequential_surgical_roadmap(patient_data: Dict, anatomical_findings: Dict):
    """Render dynamic surgical roadmap based on anatomy"""
    st.markdown(
        "<h3 class='section-header'>🗺️ Sequential Surgical Roadmap</h3>",
        unsafe_allow_html=True
    )
    
    if not MULTIMODAL_AVAILABLE:
        st.error("Multimodal processor required. Install dependencies first.")
        return
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("⬅️ Previous", use_container_width=True):
            if st.session_state.current_step > 0:
                st.session_state.current_step -= 1
                st.rerun()
    
    with col2:
        st.markdown(
            f"<div style='text-align: center; padding: 15px;'>"
            f"<span style='font-size: 1.3em; color: #00D9FF;'>Step {st.session_state.current_step + 1} / "
            f"{len(AnatomicalAnalyzer.SURGICAL_STEPS)}</span></div>",
            unsafe_allow_html=True
        )
    
    with col3:
        if st.button("Next ➡️", use_container_width=True):
            if st.session_state.current_step < len(AnatomicalAnalyzer.SURGICAL_STEPS) - 1:
                st.session_state.current_step += 1
                st.rerun()
    
    # Get current step with anatomy-based risks
    current_step = AnatomicalAnalyzer.get_next_surgical_step(
        st.session_state.current_step,
        anatomical_findings
    )
    
    # Display current step details
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(
            f"""
            <div class='metric-card'>
                <h3 style='color: #00D9FF; margin-top: 0;'>Step {current_step['step']}: {current_step['name']}</h3>
                <p><strong>Description:</strong> {current_step['description']}</p>
                <p><strong>Instruments:</strong> {', '.join(current_step.get('instruments', ['N/A']))}</p>
                <p><strong>Anatomical Structures:</strong> {', '.join(current_step.get('anatomy', ['N/A']))}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        # Risk assessment for current step
        if current_step.get('risks'):
            st.markdown("<h4 style='color: #FFB400;'>⚠️ Step Risks</h4>", unsafe_allow_html=True)
            for risk in current_step['risks']:
                severity_class = f"alert-{risk['severity'].lower()}"
                st.markdown(
                    f"""
                    <div class='alert-container {severity_class}'>
                        <strong>{risk['severity']}: {risk['risk']}</strong><br>
                        <em>→ {risk['mitigation']}</em>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.markdown(
                "<div style='padding: 15px; background: rgba(6,255,165,0.1); border-radius: 8px; color: #06FFA5;'>"
                "<strong>✓ Low Risk Step</strong></div>",
                unsafe_allow_html=True
            )

# ============================================================================
# UI COMPONENTS - AGENTIC RAG DOUBT CLEARING
# ============================================================================

def render_agentic_rag_assistant(patient_data: Dict):
    """Render Agentic RAG window for surgeon doubt clearing"""
    st.markdown(
        "<h3 class='section-header'>🧠 Agentic Q&A (Decompose-Retrieve-Synthesize)</h3>",
        unsafe_allow_html=True
    )
    
    st.info(
        "💡 **Agentic RAG Logic**: Questions are decomposed into sub-queries, "
        "medical literature is retrieved, and responses are synthesized with "
        "scan context integration."
    )
    
    # Chat display
    chat_container = st.container()
    
    with chat_container:
        st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
        
        if st.session_state.chat_history:
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.markdown(
                        f"""
                        <div class='chat-message chat-user'>
                            {message['content']}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div class='chat-message chat-assistant'>
                            {message['content']}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        else:
            st.markdown(
                """
                <div style='text-align: center; padding: 40px; color: #A0AEC0;'>
                    <p>No questions yet. Ask about surgical technique, anatomy, or risks.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Input area
    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_query = st.text_input(
            "Ask a surgical question...",
            placeholder="E.g., How to handle anomalous cystic artery in this patient?",
            key="agentic_rag_input"
        )
    
    with col2:
        send_button = st.button("Send", key="btn_send_rag", use_container_width=True)
    
    # Process query with agentic RAG
    if send_button and user_query:
        # Add user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_query
        })
        
        # Create context from uploaded files
        context = ""
        if st.session_state.file_analyses:
            for file_hash, analysis in st.session_state.file_analyses.items():
                if analysis['type'] == 'pdf':
                    context = analysis.get('full_text', '')[:1000]
                    break
        
        # Query backend with agentic RAG
        with st.spinner("🤔 Decomposing query, retrieving documents, synthesizing response..."):
            response = query_surgeon_assistant(
                user_query,
                patient_data.get("patient_id", "p001"),
                context=context
            )
        
        if response:
            # Add assistant response
            if "response" in response:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response["response"]
                })
            
            # Display reasoning path
            if "reasoning_path" in response:
                with st.expander("🔗 Decomposition & Retrieval Chain"):
                    reasoning = response["reasoning_path"]
                    if isinstance(reasoning, list):
                        for i, step in enumerate(reasoning, 1):
                            st.write(f"**Step {i}**: {step}")
                    else:
                        st.write(reasoning)
            
            st.rerun()
        else:
            st.error("Could not process query. Ensure backend is running.")

# ============================================================================
# HEADER
# ============================================================================

def render_header():
    """Render main header"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown(
            """
            <h1 style='text-align: center; color: #00D9FF; text-transform: uppercase; 
            letter-spacing: 2px; margin-bottom: 10px;'>
            🏥 MedSync AI
            </h1>
            <p style='text-align: center; color: #A0AEC0; font-size: 0.9em;'>
            Unified Multimodal Surgical Intelligence Platform
            </p>
            """,
            unsafe_allow_html=True
        )

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point"""
    
    # Render header
    render_header()
    
    # Render sidebar with multimodal intake and system status
    render_multimodal_sidebar()
    render_system_status_monitor()
    
    st.markdown("---")
    
    # Main navigation
    page = st.radio(
        "Select View",
        [
            "📋 Patient Registration",
            "🏥 Pre-Op Briefing",
            "🗺️ Surgical Roadmap",
            "🧠 Surgeon Q&A"
        ],
        label_visibility="collapsed",
        horizontal=True
    )
    
    st.markdown("---")
    
    # Patient registration
    if page == "📋 Patient Registration":
        st.markdown(
            "<h2 style='color: #00D9FF;'>Patient Registration & Intake</h2>",
            unsafe_allow_html=True
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            patient_id = st.text_input("Patient ID", placeholder="p001")
            age = st.number_input("Age", min_value=0, value=58)
            on_anticoagulation = st.checkbox("On Anticoagulation")
            platelet_count = st.number_input("Platelet Count", min_value=0, value=95000)
        
        with col2:
            on_aspirin = st.checkbox("On Aspirin")
            has_coagulopathy = st.checkbox("Coagulopathy")
            mh_family_history = st.checkbox("Family History")
            has_renal_impairment = st.checkbox("Renal Impairment")
        
        has_hepatic_impairment = st.checkbox("Hepatic Impairment")
        
        if st.button("🔐 Register Patient", use_container_width=True):
            if not patient_id:
                st.error("Patient ID required")
            else:
                patient_data = {
                    "patient_id": patient_id,
                    "age": age,
                    "on_anticoagulation": on_anticoagulation,
                    "on_aspirin": on_aspirin,
                    "platelet_count": platelet_count,
                    "has_coagulopathy": has_coagulopathy,
                    "mh_family_history": mh_family_history,
                    "has_renal_impairment": has_renal_impairment,
                    "has_hepatic_impairment": has_hepatic_impairment
                }
                
                result = register_patient(patient_data)
                if result:
                    st.session_state.patient_data = patient_data
                    st.success(f"✓ Patient {patient_id} registered")
    
    # Pre-Op Briefing
    elif page == "🏥 Pre-Op Briefing":
        if not st.session_state.patient_data:
            st.info("👤 Please register a patient first")
        else:
            render_preop_brief(
                st.session_state.patient_data,
                st.session_state.uploaded_files if st.session_state.uploaded_files else None
            )
    
    # Surgical Roadmap
    elif page == "🗺️ Surgical Roadmap":
        if not st.session_state.patient_data:
            st.info("👤 Please register a patient first")
        else:
            anatomical_findings = st.session_state.anatomical_findings or {
                'liver_segments': [],
                'vascular_involvement': [],
                'surgical_complexity': 'Standard'
            }
            render_sequential_surgical_roadmap(st.session_state.patient_data, anatomical_findings)
    
    # Agentic RAG
    elif page == "🧠 Surgeon Q&A":
        if not st.session_state.patient_data:
            st.info("👤 Please register a patient first")
        else:
            render_agentic_rag_assistant(st.session_state.patient_data)

if __name__ == "__main__":
    main()

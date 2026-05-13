"""
MedSync AI - Enhanced Professional Doc-to-Doc Surgical Consultant
A comprehensive multimodal surgical intelligence system with automatic analysis initiation
"""

import streamlit as st
import requests
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

# Configure page
st.set_page_config(
    page_title="MedSync AI - Surgical Consultant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# API Configuration
API_BASE_URL = "http://localhost:8000"
HEALTH_CHECK_INTERVAL = 10

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CUSTOM STYLING
# ============================================================================

CUSTOM_CSS = """
<style>
    :root {
        --primary: #00D9FF;
        --secondary: #FF006E;
        --success: #06FFA5;
        --warning: #FFB400;
        --danger: #FF006E;
        --dark: #0A0E27;
        --surface: #1A1F3A;
        --text-primary: #E0E7FF;
        --text-secondary: #A0AEC0;
    }

    body { background-color: var(--dark); color: var(--text-primary); }
    
    .doctor-message {
        background: rgba(0, 217, 255, 0.08);
        border-left: 4px solid var(--primary);
        padding: 15px;
        border-radius: 8px;
        margin: 12px 0;
        font-size: 0.95em;
        line-height: 1.6;
    }

    .medical-section {
        background: rgba(6, 255, 165, 0.05);
        border: 1px solid rgba(6, 255, 165, 0.2);
        padding: 12px;
        border-radius: 6px;
        margin: 8px 0;
    }

    .risk-high {
        background: rgba(255, 0, 110, 0.1);
        border-left: 3px solid #FF006E;
        color: #FF6B9D;
    }

    .risk-medium {
        background: rgba(255, 180, 0, 0.1);
        border-left: 3px solid #FFB400;
        color: #FFD700;
    }

    .risk-low {
        background: rgba(6, 255, 165, 0.1);
        border-left: 3px solid #06FFA5;
        color: #06FFA5;
    }

    .status-indicator {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.85em;
        font-weight: bold;
        margin: 5px 0;
    }

    .status-ready {
        background: rgba(6, 255, 165, 0.2);
        color: #06FFA5;
    }

    .status-offline {
        background: rgba(255, 0, 110, 0.2);
        color: #FF006E;
    }

    .surgical-step {
        background: var(--surface);
        border-left: 3px solid var(--warning);
        padding: 12px;
        margin: 10px 0;
        border-radius: 4px;
    }

    .analysis-container {
        background: rgba(0, 217, 255, 0.03);
        border: 1px solid rgba(0, 217, 255, 0.2);
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def initialize_session_state():
    """Initialize session state"""
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        
        # Patient context
        st.session_state.patient = {
            "age": 0,
            "gender": "Not specified",
            "bp_systolic": 0,
            "bp_diastolic": 0,
            "blood_sugar": 0,
            "height": 0,
            "weight": 0,
            "bmi": 0,
            "diagnosis": "",
            "medications": [],
            "allergies": "",
            "comorbidities": [],
            "recent_labs": "",
            "scan_findings": ""
        }
        
        # Conversation
        st.session_state.conversation = []  # Full conversation history
        st.session_state.initial_analysis = None  # First bot analysis
        st.session_state.analysis_shown = False
        
        # System
        st.session_state.gemini_available = False
        st.session_state.last_health_check = 0
        st.session_state.active_session = False


initialize_session_state()


# ============================================================================
# API FUNCTIONS
# ============================================================================

def check_api_health() -> bool:
    """Check if backend API is available"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            st.session_state.gemini_available = data.get("gemini_api_available", False)
            return True
    except:
        pass
    return False


def extract_report_findings(uploaded_image) -> str:
    """
    Extract text and findings from uploaded medical report images using Gemini vision
    """
    try:
        import base64
        image_bytes = uploaded_image.read()
        base64_image = base64.standard_b64encode(image_bytes).decode('utf-8')
        
        # Determine image type
        image_type = "image/jpeg"
        if uploaded_image.name.lower().endswith('.png'):
            image_type = "image/png"
        elif uploaded_image.name.lower().endswith('.gif'):
            image_type = "image/gif"
        elif uploaded_image.name.lower().endswith('.webp'):
            image_type = "image/webp"
        
        # Call backend to analyze image with Gemini
        response = requests.post(
            f"{API_BASE_URL}/api/analyze-report",
            json={
                "image_base64": base64_image,
                "image_type": image_type,
                "filename": uploaded_image.name
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            return data.get("extracted_findings", "Could not extract findings from image")
        else:
            return f"Error analyzing image: {response.status_code}"
    
    except Exception as e:
        return f"Error processing image: {str(e)}"


def generate_medical_analysis(patient_data: Dict, analysis_type: str = "comprehensive") -> str:
    """
    Generate comprehensive medical analysis from Gemini
    
    analysis_type: "comprehensive" (initial), "roadmap" (surgery plan), "qa" (questions)
    """
    
    # Build patient context
    context = format_patient_context(patient_data)
    
    if analysis_type == "comprehensive":
        prompt = f"""{context}

Provide ONLY a comprehensive and COMPLETE pre-operative clinical analysis. 

Do NOT include any greeting, courtesy phrases, or opening remarks. Go directly to the analysis.

Structure your response with these EXACT sections and provide COMPLETE content for each:

**1. CLINICAL IMPRESSION**
Complete clinical assessment of the patient's current condition. Provide thorough evaluation of:
- Diagnosis and pathology
- Current clinical status and disease severity
- Organ system assessment (cardiovascular, respiratory, renal, hepatic, etc.)
- Overall fitness for surgery
- Risk factors specific to this patient
Provide a FULL, COMPLETE impression, not abbreviated.

**2. PRE-OPERATIVE RISK STRATIFICATION**
Detailed analysis of:
- Hemorrhage risk and potential sources
- Anesthetic risks based on comorbidities
- Organ-specific risks (cardiac, pulmonary, renal, hepatic)
- Thromboembolic risk
- Infection risk
- Quantify risks where possible with percentages or severity scores

**3. SURGICAL APPROACH & PROCEDURAL STEPS**
Complete step-by-step surgical plan:
- Positioning and exposure
- Initial exploration and assessment
- Dissection planes and critical structures
- Main procedure steps in sequence
- Closure technique
- Expected duration
- Key anatomical landmarks and critical view of safety

**4. INTRA-OPERATIVE RISK MITIGATION**
Specific strategies for THIS patient:
- Hemorrhage prevention and management
- Anesthetic considerations and alternatives
- Monitoring parameters
- Critical checkpoints during surgery
- Emergency protocols if complications arise

**5. SUCCESS RATE & OUTCOME PROBABILITY**
Evidence-based assessment:
- Estimated surgical success rate (%) for this specific patient
- Reasoning based on patient factors, pathology, and surgeon experience
- Realistic complications timeline
- Expected recovery trajectory

**6. POST-OPERATIVE MANAGEMENT**
Detailed recovery plan:
- ICU vs ward admission criteria
- Monitoring parameters and frequency
- Pain management strategy
- Drain management (if applicable)
- Nutrition and mobilization timeline
- Discharge criteria
- Outpatient follow-up schedule (timing and specialists)

**7. TEAM COORDINATION & CONTINGENCIES**
Operational details:
- Required team composition (surgeons, anesthesia, nursing, ICU)
- Communication checkpoints
- Backup surgical plans if primary approach fails
- Equipment and resource requirements
- Blood product availability

**8. EVIDENCE-BASED GUIDELINES & RECOMMENDATIONS**
Professional standards:
- Relevant surgical society guidelines
- Evidence-based best practices for this case
- Preventive measures for known complications
- Quality metrics for surgical success

Provide FULL, COMPREHENSIVE, PROFESSIONAL analysis without any opening pleasantries. MUST be complete and thorough."""
    
    elif analysis_type == "roadmap":
        prompt = f"""{context}

{st.session_state.initial_analysis if st.session_state.initial_analysis else ""}

Create a detailed surgical roadmap using the CholecT45 framework (for cholecystectomy):

**DETAILED PROCEDURAL ROADMAP**
- Step 1-2: Positioning & access
- Step 3-4: Initial exploration & dissection
- Step 5-6: Critical view of safety achievement
- Step 7-8: Arterial and duct division
- Step 9: Specimen extraction and closure

For each step: describe technique, anatomical pearls, specific risks for THIS patient, troubleshooting strategies.

Be surgical and specific."""
    
    else:  # qa
        prompt = f"""{context}

{st.session_state.initial_analysis if st.session_state.initial_analysis else ""}

Provide expert surgical consultation addressing the surgeon's questions with precision and practicality."""
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/surgeon/query",
            json={"query": prompt, "patient_id": "medsync_patient"},
            timeout=180
        )
        
        if response.status_code == 200:
            data = response.json()
            result = data.get("response", "Error generating response")
            return result
        else:
            error_msg = f"API Error: {response.status_code}"
            return error_msg
    
    except requests.exceptions.Timeout as e:
        error_msg = f"⏱️ Request timeout (180s). AI is working on comprehensive analysis. Please wait or try again in a moment."
        return error_msg
    except requests.exceptions.ConnectionError as e:
        error_msg = f"❌ Cannot connect to backend at {API_BASE_URL}. Is it running? Error: {str(e)}"
        return error_msg
    except Exception as e:
        error_msg = f"Error: {type(e).__name__}: {str(e)}"
        return error_msg


def format_patient_context(patient_data: Dict) -> str:
    """Format patient data for AI analysis"""
    
    context = f"""PATIENT CLINICAL PROFILE:
Age: {patient_data.get('age', 'Not specified')} years
Gender: {patient_data.get('gender', 'Not specified')}
BMI: {patient_data.get('bmi', 'Not calculated')}

VITALS:
BP: {patient_data.get('bp_systolic', 0)}/{patient_data.get('bp_diastolic', 0)} mmHg
Blood Sugar: {patient_data.get('blood_sugar', 0)} mg/dL
Height: {patient_data.get('height', 0)}cm | Weight: {patient_data.get('weight', 0)}kg

CLINICAL DATA:
Diagnosis: {patient_data.get('diagnosis', 'Not specified')}
Medications: {', '.join(patient_data.get('medications', [])) or 'None'}
Allergies: {patient_data.get('allergies', 'None')}
Comorbidities: {', '.join(patient_data.get('comorbidities', [])) or 'None'}

IMAGING & LABS:
Scan Findings: {patient_data.get('scan_findings', 'Not provided')}
Recent Labs: {patient_data.get('recent_labs', 'Not provided')}"""
    
    return context


# ============================================================================
# MAIN INTERFACE
# ============================================================================

st.title("🏥 MedSync AI - Professional Surgical Consultant")
st.subheader("Doc-to-Doc Evidence-Based Surgical Planning")

# Health check
current_time = time.time()
if current_time - st.session_state.last_health_check > HEALTH_CHECK_INTERVAL:
    check_api_health()
    st.session_state.last_health_check = current_time

# Status
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.markdown("**System Status**")
with col2:
    if st.session_state.gemini_available:
        st.markdown('<span class="status-indicator status-ready">🟢 Online</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-indicator status-offline">🔴 Offline</span>', unsafe_allow_html=True)
with col3:
    if st.button("🔄 Refresh", key="refresh_btn"):
        check_api_health()
        st.rerun()

st.markdown("---")

# ============================================================================
# SIDEBAR: Patient Data Collection
# ============================================================================

with st.sidebar:
    st.markdown("## 📋 Patient Information")
    st.markdown("Fill in patient details to initiate analysis")
    st.markdown("---")
    
    # Demographics
    st.markdown("### Demographics")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.patient['age'] = st.number_input(
            "Age *", min_value=0, max_value=120, 
            value=st.session_state.patient['age'], key="age"
        )
    with col2:
        st.session_state.patient['gender'] = st.selectbox(
            "Gender", ["Male", "Female", "Other"],
            key="gender"
        )
    
    # Vitals
    st.markdown("### Vital Signs")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.patient['bp_systolic'] = st.number_input(
            "BP Sys *", min_value=50, max_value=250, value=120, key="bp_sys"
        )
    with col2:
        st.session_state.patient['bp_diastolic'] = st.number_input(
            "BP Dia *", min_value=30, max_value=150, value=80, key="bp_dia"
        )
    
    st.session_state.patient['blood_sugar'] = st.number_input(
        "Blood Sugar (mg/dL) *", min_value=0, max_value=500,
        value=st.session_state.patient['blood_sugar'], key="sugar"
    )
    
    # Anthropometry
    st.markdown("### Anthropometry")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.patient['height'] = st.number_input(
            "Height (cm)", min_value=100, max_value=220,
            value=st.session_state.patient['height'] if st.session_state.patient['height'] > 100 else 170,
            key="height"
        )
    with col2:
        st.session_state.patient['weight'] = st.number_input(
            "Weight (kg)", min_value=30, max_value=200,
            value=st.session_state.patient['weight'] if st.session_state.patient['weight'] > 30 else 70,
            key="weight"
        )
    
    # Clinical Info
    st.markdown("### Clinical Information")
    st.session_state.patient['diagnosis'] = st.text_input(
        "Diagnosis", placeholder="e.g., Cholelithiasis with acute cholecystitis",
        key="diagnosis"
    )
    
    st.session_state.patient['medications'] = st.multiselect(
        "Current Medications",
        ["Aspirin", "Anticoagulants", "Beta Blockers", "ACE Inhibitors", 
         "Statins", "Diabetic Meds", "Other"],
        key="meds"
    )
    
    st.session_state.patient['allergies'] = st.text_input(
        "Drug Allergies", placeholder="e.g., Penicillin, NSAIDs",
        key="allergies"
    )
    
    st.session_state.patient['comorbidities'] = st.multiselect(
        "Comorbidities",
        ["Hypertension", "Diabetes", "CAD", "COPD", "Renal Disease", 
         "Hepatic Disease", "Bleeding Disorder"],
        key="comorbidities"
    )
    
    # Imaging
    st.markdown("### Imaging & Labs")
    
    # Image extraction for scan findings
    st.markdown("#### 📸 Upload Imaging Report")
    scan_image = st.file_uploader(
        "Upload scan report image (CT, USG, MRI, X-ray, etc.)",
        type=["jpg", "jpeg", "png", "gif", "webp"],
        key="scan_image"
    )
    
    if scan_image:
        st.info("🔄 Analyzing imaging report...")
        extracted_findings = extract_report_findings(scan_image)
        st.session_state.patient['scan_findings'] = extracted_findings
        st.success("✓ Imaging findings extracted")
    
    st.session_state.patient['scan_findings'] = st.text_area(
        "Scan Findings",
        placeholder="Describe imaging findings from CT, USG, MRI, etc. (Or upload report image above)",
        height=100, 
        value=st.session_state.patient['scan_findings'],
        key="scans"
    )
    
    # Image extraction for lab findings
    st.markdown("#### 🧪 Upload Lab Report")
    lab_image = st.file_uploader(
        "Upload lab report image",
        type=["jpg", "jpeg", "png", "gif", "webp"],
        key="lab_image"
    )
    
    if lab_image:
        st.info("🔄 Analyzing lab report...")
        extracted_labs = extract_report_findings(lab_image)
        st.session_state.patient['recent_labs'] = extracted_labs
        st.success("✓ Lab values extracted")
    
    st.session_state.patient['recent_labs'] = st.text_area(
        "Recent Lab Values",
        placeholder="e.g., Hemoglobin 12.5, Platelets 250k, INR 1.0 (Or upload report image above)",
        height=100,
        value=st.session_state.patient['recent_labs'],
        key="labs"
    )
    
    st.markdown("---")
    
    # Initiate Analysis
    # Essential fields only: Age, BP, and Blood Sugar
    patient_ready = (
        st.session_state.patient['age'] > 0 and
        st.session_state.patient['bp_systolic'] > 0 and
        st.session_state.patient['bp_diastolic'] > 0 and
        st.session_state.patient['blood_sugar'] > 0
    )
    
    if st.button("🚀 INITIATE ANALYSIS", use_container_width=True,
                 disabled=not (patient_ready and st.session_state.gemini_available)):
        st.session_state.active_session = True
        st.rerun()


# ============================================================================
# MAIN CONTENT: Conversation Interface
# ============================================================================

if not st.session_state.active_session:
    st.info("""
    ### Welcome to MedSync AI
    
    To begin a professional surgical consultation:
    
    1. **Fill patient information** in the left sidebar
    2. Complete: Age, Vitals, Diagnosis, Imaging findings
    3. Click **"INITIATE ANALYSIS"** to start
    
    The AI will provide:
    - Comprehensive clinical assessment
    - Current condition & risk factors
    - Surgical approach & detailed steps
    - Intra-operative risks & success probability
    - Post-operative care plan
    
    All in a professional doc-to-doc format.
    """)


else:
    # Generate initial analysis if not done
    if not st.session_state.analysis_shown:
        st.info("🔄 Analyzing patient profile and generating comprehensive assessment...")
        
        with st.spinner("🧠 AI is reviewing patient data and preparing response..."):
            analysis = generate_medical_analysis(st.session_state.patient, "comprehensive")
        
        if not analysis or "Error" in analysis or "Cannot connect" in analysis:
            st.error(f"Failed to generate analysis: {analysis}")
        else:
            st.session_state.initial_analysis = analysis
            st.session_state.analysis_shown = True
            st.session_state.conversation.append({
                "role": "assistant",
                "content": analysis,
                "timestamp": datetime.now().isoformat()
            })
            st.rerun()
    
    # Display conversation
    st.markdown("## 🩺 Clinical Consultation")
    st.markdown("---")
    
    # Display the initial analysis
    if st.session_state.conversation:
        for msg in st.session_state.conversation:
            if msg["role"] == "assistant":
                st.markdown("<div class='doctor-message'>", unsafe_allow_html=True)
                st.markdown(msg["content"])
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"**👨‍⚕️ Surgeon:** {msg['content']}")
    
    st.markdown("---")
    
    # Follow-up questions
    st.markdown("## Follow-Up Questions")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📋 Generate Surgical Roadmap", use_container_width=True, key="roadmap_btn"):
            with st.spinner("🗺️ Creating detailed surgical roadmap..."):
                roadmap = generate_medical_analysis(st.session_state.patient, "roadmap")
            
            st.session_state.conversation.append({
                "role": "assistant",
                "content": f"**SURGICAL ROADMAP & CholecT45 FRAMEWORK**\n\n{roadmap}",
                "timestamp": datetime.now().isoformat()
            })
            st.rerun()
    
    with col2:
        if st.button("❓ Ask Specific Question", use_container_width=True, key="ask_btn"):
            st.session_state.show_qa = True
    
    # Q&A Interface
    if hasattr(st.session_state, 'show_qa') and st.session_state.show_qa:
        st.markdown("### Technical Question")
        user_q = st.text_area("Ask a specific surgical question:", 
                             placeholder="e.g., How would you handle variant anatomy in this case?",
                             key="qa_input")
        
        if st.button("Submit Question", key="submit_q"):
            if user_q:
                st.session_state.conversation.append({
                    "role": "human",
                    "content": user_q,
                    "timestamp": datetime.now().isoformat()
                })
                
                with st.spinner("💭 Consulting AI expert..."):
                    response = generate_medical_analysis(st.session_state.patient, "qa")
                
                st.session_state.conversation.append({
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.now().isoformat()
                })
                
                st.session_state.show_qa = False
                st.rerun()
    
    # End Session
    if st.button("🔄 New Patient", use_container_width=True):
        st.session_state.active_session = False
        st.session_state.analysis_shown = False
        st.session_state.conversation = []
        st.session_state.initial_analysis = None
        st.rerun()

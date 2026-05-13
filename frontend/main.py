"""
MedSync AI - Conversational Multimodal Surgical Consultant
A Doc-to-Doc interaction paradigm for surgical decision support
Phases: Assessment → Roadmap → Doubt Clearing
"""

import streamlit as st
import requests
import json
import base64
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

# Medical imaging
from PIL import Image
import io

# Import multimodal processor
from multimodal_processor import (
    MedicalFileProcessor, 
    ScanFindingsExtractor,
    AnatomicalAnalyzer,
    PreOpBriefGenerator
)

# Configure page layout
st.set_page_config(
    page_title="MedSync AI - Surgical Consultant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Configuration
API_BASE_URL = "http://localhost:8000"
HEALTH_CHECK_INTERVAL = 5

# ============================================================================
# CUSTOM CSS FOR DARK MODE THEME
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
        --darker: #05070F;
        --surface: #1A1F3A;
        --surface-light: #252D47;
        --text-primary: #E0E7FF;
        --text-secondary: #A0AEC0;
        --border: #2D3B5C;
    }

    body, .main {
        background-color: var(--dark);
        color: var(--text-primary);
    }

    .sidebar .sidebar-content {
        background-color: var(--darker);
    }

    /* Chat message styling */
    .chat-message {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid;
        word-wrap: break-word;
        line-height: 1.6;
    }

    .chat-message.human {
        background: rgba(0, 217, 255, 0.1);
        border-left-color: var(--primary);
        margin-left: 30px;
        text-align: left;
    }

    .chat-message.assistant {
        background: rgba(0, 217, 255, 0.05);
        border-left-color: var(--success);
        margin-right: 30px;
    }

    .patient-context-card {
        background: linear-gradient(135deg, var(--surface) 0%, var(--surface-light) 100%);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0, 217, 255, 0.1);
    }

    .phase-indicator {
        font-weight: bold;
        padding: 8px 12px;
        border-radius: 6px;
        margin: 5px 0;
        font-size: 12px;
        text-align: center;
    }

    .phase-0 { background: rgba(160, 174, 192, 0.2); color: #A0AEC0; }
    .phase-1 { background: rgba(0, 217, 255, 0.2); color: var(--primary); }
    .phase-2 { background: rgba(255, 180, 0, 0.2); color: var(--warning); }
    .phase-3 { background: rgba(6, 255, 165, 0.2); color: var(--success); }

    .status-online {
        color: var(--success);
        font-weight: bold;
        display: inline-block;
    }

    .status-offline {
        color: var(--danger);
        font-weight: bold;
        display: inline-block;
    }

    .input-group {
        background: var(--surface);
        padding: 12px;
        border-radius: 8px;
        border: 1px solid var(--border);
        margin: 8px 0;
    }

    .scan-uploader {
        background: var(--surface);
        padding: 15px;
        border-radius: 8px;
        border: 2px dashed var(--border);
        text-align: center;
        margin: 10px 0;
    }

    .vital-input {
        background: var(--surface);
        padding: 10px;
        border-radius: 6px;
        margin: 5px 0;
    }

    .context-summary {
        background: rgba(0, 217, 255, 0.08);
        border-left: 4px solid var(--primary);
        padding: 12px;
        border-radius: 6px;
        font-size: 0.9em;
        margin: 10px 0;
    }

    .phase-transition {
        background: linear-gradient(135deg, rgba(255, 180, 0, 0.1) 0%, rgba(0, 217, 255, 0.1) 100%);
        border: 1px solid var(--warning);
        border-radius: 8px;
        padding: 15px;
        margin: 15px 0;
        text-align: center;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def initialize_session_state():
    """Initialize session state for conversational workflow"""
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        
        # Patient context (Side panel inputs)
        st.session_state.patient_context = {
            "age": 50,
            "bp_systolic": 120,
            "bp_diastolic": 80,
            "sugar": 100,
            "height": 170.0,
            "weight": 70.0,
            "previous_operations": "",
            "chronic_conditions": "",
            "medications": {
                "aspirin": False,
                "anticoagulants": False,
                "coagulopathy": False
            },
            "scans": [],
            "scan_findings": []
        }
        
        # Conversation state
        st.session_state.messages = []
        st.session_state.current_phase = 0  # 0: Idle, 1: Assessment, 2: Roadmap, 3: Doubt Clearing
        st.session_state.phase_complete = {1: False, 2: False, 3: False}
        st.session_state.roadmap_steps = []
        st.session_state.scan_findings = ""
        st.session_state.assessment_summary = ""
        
        # System state
        st.session_state.gemini_api_available = False
        st.session_state.last_health_check = 0
        st.session_state.health_check_interval = HEALTH_CHECK_INTERVAL

initialize_session_state()

# ============================================================================
# API HELPER FUNCTIONS
# ============================================================================

def check_system_health() -> bool:
    """Check backend health and Gemini API availability"""
    try:
        # Check backend connection
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            st.session_state.gemini_api_available = data.get("gemini_api_available", False)
            if st.session_state.gemini_api_available:
                logger.info("✓ Gemini API is connected")
            else:
                logger.warning("⚠️ Gemini API not connected - check GEMINI_API_KEY")
            return True
        else:
            logger.warning(f"Health check returned status {response.status_code}")
    except requests.exceptions.Timeout:
        logger.warning(f"Health check timeout")
    except requests.exceptions.ConnectionError:
        logger.warning(f"Cannot connect to backend at {API_BASE_URL}")
    except Exception as e:
        logger.info(f"Health check failed: {e}")
    return False

def query_llm(prompt: str, patient_data: Dict = None, images: List = None) -> str:
    """Query the LLM via backend API with optional images"""
    try:
        payload = {
            "query": prompt,
            "patient_id": "console_patient"
        }
        
        if patient_data:
            payload.update(patient_data)
        
        # Gemini API is fast - 10-30 second timeout
        response = requests.post(
            f"{API_BASE_URL}/api/surgeon/query",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            return data.get("response", "Error: Empty response")
        else:
            return f"Backend error: {response.status_code}"
    
    except requests.exceptions.Timeout:
        return "⏱️ The API is processing your query. Please wait..."
    except requests.exceptions.ConnectionError:
        return "❌ Cannot connect to backend. Ensure backend is running on localhost:8000"
    except Exception as e:
        return f"Error: {str(e)}"

def process_medical_scan(uploaded_file) -> Dict:
    """
    Process uploaded medical scan and extract key findings using multimodal processor
    """
    try:
        # Validate file
        is_valid, message = MedicalFileProcessor.validate_file(uploaded_file)
        if not is_valid:
            return {"error": message}
        
        # Extract DICOM metadata if applicable
        metadata = None
        if MedicalFileProcessor.get_file_type(uploaded_file.name) == 'dicom':
            metadata = MedicalFileProcessor.extract_dicom_metadata(uploaded_file)
        
        # Analyze scan findings
        findings = ScanFindingsExtractor.analyze_scan(uploaded_file, metadata)
        
        return findings
    except Exception as e:
        logger.error(f"Error processing scan: {e}")
        return {"error": str(e)}

def calculate_hemorrhage_risk(patient_ctx: Dict) -> float:
    """
    Calculate hemorrhage risk score based on patient factors and scan findings
    Range: 0.0 - 1.0
    """
    risk = 0.0
    
    # Medication factors
    if patient_ctx.get("medications", {}).get("anticoagulants"):
        risk += 0.25
    if patient_ctx.get("medications", {}).get("aspirin"):
        risk += 0.10
    if patient_ctx.get("medications", {}).get("coagulopathy"):
        risk += 0.30
    
    # Age factor
    age = patient_ctx.get("age", 50)
    if age > 70:
        risk += 0.15
    
    # Scan risk flags
    scan_findings = patient_ctx.get("scan_findings", [])
    for scan in scan_findings:
        if "vascular" in str(scan.get("risk_flags", "")).lower():
            risk += 0.20
        if "abnormal" in str(scan.get("findings_summary", "")).lower():
            risk += 0.10
    
    return min(1.0, risk)

def calculate_anesthetic_risk(patient_ctx: Dict) -> float:
    """
    Calculate anesthetic reaction risk score
    Range: 0.0 - 1.0
    """
    risk = 0.0
    
    # Age factor
    age = patient_ctx.get("age", 50)
    if age > 75:
        risk += 0.20
    elif age > 65:
        risk += 0.10
    
    # Vital signs stability
    bp_sys = patient_ctx.get("bp_systolic", 120)
    bp_dia = patient_ctx.get("bp_diastolic", 80)
    
    if bp_sys > 160 or bp_sys < 90 or bp_dia > 100 or bp_dia < 50:
        risk += 0.15
    
    # Metabolic factors
    sugar = patient_ctx.get("sugar", 100)
    if sugar > 200 or sugar < 70:
        risk += 0.10
    
    bmi = calculate_bmi(patient_ctx)
    if bmi > 35 or bmi < 18.5:
        risk += 0.10
    
    return min(1.0, risk)

def calculate_bmi(patient_ctx: Dict) -> float:
    """Calculate BMI from height and weight"""
    height_m = patient_ctx.get("height", 170) / 100
    weight = patient_ctx.get("weight", 70)
    if height_m > 0:
        return weight / (height_m ** 2)
    return 22.0

def format_patient_context() -> str:
    """Format patient context for system prompt"""
    ctx = st.session_state.patient_context
    med_list = []
    
    if ctx["medications"]["aspirin"]:
        med_list.append("Aspirin")
    if ctx["medications"]["anticoagulants"]:
        med_list.append("Anticoagulants")
    if ctx["medications"]["coagulopathy"]:
        med_list.append("Coagulopathy")
    
    medications_str = ", ".join(med_list) if med_list else "None"
    
    bmi = calculate_bmi(ctx)
    hemorrhage_risk = calculate_hemorrhage_risk(ctx)
    anesthetic_risk = calculate_anesthetic_risk(ctx)
    
    # Format hemorrhage and anesthetic risk as percentages
    hemorrhage_pct = int(hemorrhage_risk * 100)
    anesthetic_pct = int(anesthetic_risk * 100)
    
    context = f"""PATIENT PROFILE:
- Age: {ctx['age']} years
- BP: {ctx['bp_systolic']}/{ctx['bp_diastolic']} mmHg
- Blood Sugar: {ctx['sugar']} mg/dL
- Height: {ctx['height']}cm | Weight: {ctx['weight']}kg | BMI: {bmi:.1f}
- Previous Operations: {ctx['previous_operations'] or 'None'}
- Chronic Conditions: {ctx['chronic_conditions'] or 'None'}
- Medications: {medications_str}
- Scans Uploaded: {len(ctx['scans'])} files

RISK ASSESSMENT:
- Hemorrhage Risk: {hemorrhage_pct}%
- Anesthetic Reaction Risk: {anesthetic_pct}%

SCAN FINDINGS:
"""
    
    # Add scan findings if available
    if ctx['scans']:
        scan_context = ScanFindingsExtractor.synthesize_clinical_context(ctx['scans'])
        context += scan_context
    else:
        context += "No scans uploaded yet."
    
    return context

def is_patient_context_complete() -> bool:
    """Check if patient context is sufficiently populated"""
    ctx = st.session_state.patient_context
    return (
        ctx['age'] > 0 and 
        ctx['bp_systolic'] > 0 and 
        len(ctx['scans']) > 0
    )

# ============================================================================
# PHASE 1: ASSESSMENT
# ============================================================================

def phase_1_assessment():
    """Automated assessment phase initiated when patient context is populated"""
    
    if st.session_state.phase_complete[1]:
        return
    
    if not is_patient_context_complete():
        st.warning("⚠️ Please fill in Age, Vitals, and upload at least one scan to proceed with Assessment.")
        return
    
    # Calculate risk factors
    ctx = st.session_state.patient_context
    hemorrhage_risk = calculate_hemorrhage_risk(ctx)
    anesthetic_risk = calculate_anesthetic_risk(ctx)
    
    # Estimate success rate (inverse of risks, with some baseline)
    success_rate = max(55, 95 - (int(hemorrhage_risk * 30) + int(anesthetic_risk * 20)))
    
    # Generate assessment prompt with detailed clinical context
    assessment_prompt = f"""You are an expert surgical consultant reviewing a patient before surgery. Use a conversational, collegial tone (Doc-to-Doc).

{format_patient_context()}

Based on this patient's comprehensive profile with multimodal imaging:

1. **Clinical Impression**: Review the patient demographics, vitals, medications, and scan findings. What is your initial clinical assessment?

2. **Risk Analysis**: 
   - Hemorrhage Risk: {int(hemorrhage_risk*100)}%
   - Anesthetic Reaction Risk: {int(anesthetic_risk*100)}%
   - What are the primary risk factors THIS patient faces?

3. **Success Estimate**: Based on the risk profile and imaging findings, what is your estimated procedural success rate?

4. **Key Considerations**: What should we prioritize in the surgical plan given this patient's condition?

5. **Pre-operative Optimization**: Any urgent steps before surgery?

Format as a collegial surgical discussion. Be precise, acknowledge the scan findings, and be actionable. Keep under 350 words."""
    
    # Query LLM
    with st.spinner("🔄 Analyzing patient context and scan findings..."):
        response = query_llm(assessment_prompt)
    
    # Store assessment summary
    st.session_state.assessment_summary = response
    
    # Add to conversation
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "phase": 1,
        "timestamp": datetime.now().isoformat()
    })
    
    st.session_state.phase_complete[1] = True

# ============================================================================
# PHASE 2: ROADMAP
# ============================================================================

def phase_2_roadmap():
    """Surgical roadmap phase - CholecT45 workflow"""
    
    if st.session_state.phase_complete[2]:
        return
    
    if not st.session_state.phase_complete[1]:
        st.warning("⏳ Complete Assessment Phase first.")
        return
    
    # Get anatomical findings from scans
    ctx = st.session_state.patient_context
    anatomical_findings = AnatomicalAnalyzer.identify_anatomical_findings(
        format_patient_context()
    )
    
    roadmap_prompt = f"""{format_patient_context()}

ASSESSMENT SUMMARY:
{st.session_state.assessment_summary}

Using the CholecT45 surgical framework (9-step hepatobiliary workflow), provide a detailed surgical roadmap:

1. **Procedure Overview**: Given this patient's risk profile and imaging, outline the 9-step surgical approach
2. **CholecT45 Steps**: 
   - Steps 1-3: Preparation, positioning, trocar placement, insufflation
   - Steps 4-7: Critical view of safety, artery division, triangle clearance, duct division
   - Steps 8-9: Gallbladder dissection, closure
3. **Anatomical Landmarks**: Key structures to identify, especially given {'the vascular complexity' if anatomical_findings.get('vascular_involvement') else 'standard anatomy'}
4. **Critical Checkpoints**: Where to pause and reassess
5. **Alternative Procedures**: Fallback plans if complications arise
6. **Timeline**: Estimated duration per phase

Be surgical and specific. Address the patient's risk factors. Keep under 450 words but be comprehensive."""
    
    with st.spinner("🗺️ Generating surgical roadmap with CholecT45 framework..."):
        response = query_llm(roadmap_prompt)
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "phase": 2,
        "timestamp": datetime.now().isoformat()
    })
    
    st.session_state.roadmap_steps = response.split('\n')
    st.session_state.phase_complete[2] = True

# ============================================================================
# SIDEBAR: PATIENT CONTEXT MANAGER
# ============================================================================

with st.sidebar:
    st.markdown("## 🏥 Patient Context Manager")
    st.markdown("---")
    
    # Phase indicator
    phase_names = {0: "Idle", 1: "Assessment", 2: "Roadmap", 3: "Doubt Clearing"}
    phase_num = st.session_state.current_phase
    st.markdown(
        f"<div class='phase-indicator phase-{phase_num}'>📍 Phase: {phase_names.get(phase_num)}</div>",
        unsafe_allow_html=True
    )
    
    # System Status
    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.gemini_api_available:
            st.markdown("<span class='status-online'>✓ Gemini Connected</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span class='status-offline'>⚠️ Gemini Offline</span>", unsafe_allow_html=True)
    
    with col2:
        if st.button("🔄", use_container_width=True, key="refresh_health"):
            check_system_health()
            st.rerun()
    
    st.markdown("---")
    
    # === PATIENT DEMOGRAPHICS ===
    st.markdown("### 👤 Demographics")
    st.session_state.patient_context['age'] = st.number_input(
        "Age (years)", min_value=0, max_value=120,
        value=st.session_state.patient_context['age'],
        key="age_input"
    )
    
    # === VITAL SIGNS ===
    st.markdown("### 💓 Vital Signs")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.patient_context['bp_systolic'] = st.number_input(
            "BP Sys (mmHg)", min_value=60, max_value=200,
            value=st.session_state.patient_context['bp_systolic'],
            key="bp_sys", help="Systolic Blood Pressure"
        )
    with col2:
        st.session_state.patient_context['bp_diastolic'] = st.number_input(
            "BP Dia (mmHg)", min_value=40, max_value=130,
            value=st.session_state.patient_context['bp_diastolic'],
            key="bp_dias", help="Diastolic Blood Pressure"
        )
    
    st.session_state.patient_context['sugar'] = st.number_input(
        "Blood Sugar (mg/dL)", min_value=0, max_value=500,
        value=st.session_state.patient_context['sugar'],
        key="sugar_input"
    )
    
    # === ANTHROPOMETRICS ===
    st.markdown("### 📏 Anthropometrics")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.patient_context['height'] = st.number_input(
            "Height (cm)", min_value=50.0, max_value=250.0,
            value=st.session_state.patient_context['height'],
            key="height_input"
        )
    with col2:
        st.session_state.patient_context['weight'] = st.number_input(
            "Weight (kg)", min_value=0.0, max_value=250.0,
            value=st.session_state.patient_context['weight'],
            key="weight_input"
        )
    
    # === MEDICAL HISTORY ===
    st.markdown("### 📋 Medical History")
    st.session_state.patient_context['previous_operations'] = st.text_area(
        "Previous Ops",
        value=st.session_state.patient_context['previous_operations'],
        height=50,
        placeholder="e.g., Appendectomy 2015",
        key="prev_ops"
    )
    
    st.session_state.patient_context['chronic_conditions'] = st.text_area(
        "Chronic Conditions",
        value=st.session_state.patient_context['chronic_conditions'],
        height=50,
        placeholder="e.g., Diabetes, HTN",
        key="chronic_cond"
    )
    
    # === MEDICATIONS ===
    st.markdown("### 💊 Medications")
    st.session_state.patient_context['medications']['aspirin'] = st.checkbox(
        "Aspirin",
        value=st.session_state.patient_context['medications']['aspirin'],
        key="med_aspirin"
    )
    st.session_state.patient_context['medications']['anticoagulants'] = st.checkbox(
        "Anticoagulants",
        value=st.session_state.patient_context['medications']['anticoagulants'],
        key="med_anticoag"
    )
    st.session_state.patient_context['medications']['coagulopathy'] = st.checkbox(
        "Coagulopathy",
        value=st.session_state.patient_context['medications']['coagulopathy'],
        key="med_coag"
    )
    
    st.markdown("---")
    
    # === MULTIMODAL SCAN UPLOADER ===
    st.markdown("### 📁 Medical Scans")
    st.markdown("*Upload X-ray, CT, MRI (DICOM/PDF) or Lab Results*")
    
    uploaded_files = st.file_uploader(
        "Upload scans (X-ray, CT, MRI, PDF)",
        type=["dcm", "pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="scan_uploader"
    )
    
    if uploaded_files:
        for file in uploaded_files:
            file_name = file.name
            # Check if already processed (by filename)
            if file_name not in [f.get('filename', '') for f in st.session_state.patient_context['scans']]:
                # Process scan with multimodal processor
                findings = process_medical_scan(file)
                
                if "error" not in findings:
                    st.session_state.patient_context['scans'].append(findings)
                    st.session_state.patient_context['scan_findings'].append(findings)
                    
                    # Show success with findings summary
                    with st.expander(f"✓ {file_name}", expanded=False):
                        if findings.get('findings_summary'):
                            st.write(f"**Findings**: {findings.get('findings_summary')}")
                        if findings.get('modality'):
                            st.write(f"**Modality**: {findings.get('modality')}")
                        if findings.get('anatomical_concerns'):
                            st.write("**Anatomical Concerns**:")
                            for concern in findings.get('anatomical_concerns', []):
                                st.write(f"  - {concern}")
                        if findings.get('risk_flags'):
                            st.write("**Risk Flags**:")
                            for flag in findings.get('risk_flags', []):
                                st.write(f"  🚩 {flag}")
                else:
                    st.error(f"Error processing {file_name}: {findings.get('error')}")
        
        st.success(f"✓ {len(st.session_state.patient_context['scans'])} scan(s) processed")
    
    st.markdown("---")
    
    # === CONTEXTUAL SUMMARY ===
    if is_patient_context_complete():
        st.markdown(
            f"""<div class='context-summary'>
            ✓ <b>Context Ready</b><br>
            Age: {st.session_state.patient_context['age']}<br>
            BP: {st.session_state.patient_context['bp_systolic']}/{st.session_state.patient_context['bp_diastolic']}<br>
            Scans: {len(st.session_state.patient_context['scans'])}
            </div>""",
            unsafe_allow_html=True
        )

# ============================================================================
# MAIN CONTENT: CONVERSATION INTERFACE
# ============================================================================

st.markdown("""
<h1 style='text-align: center; color: #00D9FF; text-transform: uppercase; letter-spacing: 2px;'>
🏥 MedSync AI
</h1>
<p style='text-align: center; color: #A0AEC0; font-size: 0.95em;'>
Conversational Multimodal Surgical Consultant
</p>
""", unsafe_allow_html=True)

st.markdown("---")

# Display conversation history
for message in st.session_state.messages:
    role = message.get("role", "assistant")
    content = message.get("content", "")
    phase = message.get("phase", 0)
    
    if role == "human":
        st.markdown(f"""
        <div class='chat-message human'>
            <strong>You:</strong><br>{content}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class='chat-message assistant'>
            <strong>🏥 Consultant (Phase {phase}):</strong><br>{content}
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# === PHASE WORKFLOW ===

if st.session_state.current_phase == 0:
    st.info("👋 Welcome to MedSync AI Surgical Consultant.")
    st.markdown("""
    **How it works:**
    1. **Fill patient context** in the sidebar (demographics, vitals, scans)
    2. **Assessment Phase**: I'll analyze the patient and provide clinical insights
    3. **Roadmap Phase**: Detailed surgical workflow and CholecT45 steps
    4. **Doubt Clearing**: Real-time Q&A during surgical planning
    
    Let's start! Click "Begin Assessment" when ready.
    """)
    
    if st.button("📊 Begin Assessment", key="start_assess", use_container_width=True):
        st.session_state.current_phase = 1
        phase_1_assessment()
        st.rerun()

elif st.session_state.current_phase == 1:
    if not st.session_state.phase_complete[1]:
        st.info("💡 Initiating Assessment Phase...")
        if st.button("Analyze Patient", key="run_assess", use_container_width=True):
            phase_1_assessment()
            st.rerun()
    else:
        st.markdown("""
        <div class='phase-transition'>
        <strong>✓ Assessment Complete</strong><br>
        Shall we proceed to the surgical roadmap?
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🗺️ Generate Roadmap", key="to_roadmap", use_container_width=True):
            st.session_state.current_phase = 2
            phase_2_roadmap()
            st.rerun()

elif st.session_state.current_phase == 2:
    if not st.session_state.phase_complete[2]:
        st.info("🗺️ Generating Surgical Roadmap...")
        if st.button("Create Roadmap", key="run_roadmap", use_container_width=True):
            phase_2_roadmap()
            st.rerun()
    else:
        st.markdown("""
        <div class='phase-transition'>
        <strong>✓ Roadmap Generated</strong><br>
        Ready for technical Q&A and doubt clearing
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("❓ Ask Questions", key="to_qa", use_container_width=True):
            st.session_state.current_phase = 3
            st.rerun()

elif st.session_state.current_phase == 3:
    st.markdown("### ❓ Surgeon Doubt Clearing")
    st.markdown("""
    **I am here to clarify any technical doubts or provide alternative procedural insights.**
    
    Some topics we can discuss:
    - Instrument selection and technique
    - Anatomical variations and how to manage them
    - Complication prevention and management
    - Alternative approaches if standard procedure becomes difficult
    - Post-operative management considerations
    """)

# === CHAT INPUT ===

st.markdown("---")

# Status indicator
if st.session_state.gemini_api_available:
    st.markdown('<span class="status-online">✓ Ready</span>', unsafe_allow_html=True)
else:
    st.markdown('<span class="status-offline">⚠️ Offline</span>', unsafe_allow_html=True)

# Chat input must be at the top level, not inside columns
user_input = st.chat_input(
    "Ask a surgical question or request clarification...",
    key="chat_input"
)

if user_input and st.session_state.gemini_api_available:
    # Add user message
    st.session_state.messages.append({
        "role": "human",
        "content": user_input,
        "phase": st.session_state.current_phase,
        "timestamp": datetime.now().isoformat()
    })
    
    # Generate context-aware prompt
    system_context = format_patient_context()
    phase_guidance = {
        1: """You are a senior surgical consultant providing INITIAL CLINICAL ASSESSMENT. 
Focus on: patient risk profile, key findings from imaging, success rate estimation, and pre-operative optimization.
Be thorough but concise. Acknowledge what you've learned from the scans.""",
        2: """You are providing a DETAILED SURGICAL ROADMAP using the CholecT45 framework (9-step hepatobiliary workflow).
Focus on: step-by-step procedure, anatomical landmarks, critical checkpoints, and alternative approaches.
Address the specific risks this patient faces. Be procedurally specific.""",
        3: """You are a senior surgical expert in DOUBT CLEARING and technical consultation.
Focus on: answering specific technical questions, providing alternative approaches, explaining anatomy,
managing complications, and instrument selection. Be expert-level, precise, and collegial."""
    }
    
    current_guidance = phase_guidance.get(st.session_state.current_phase, 
        "Provide expert surgical consultation based on the patient context.")
    
    full_prompt = f"""{system_context}

PREVIOUS ASSESSMENT:
{st.session_state.assessment_summary if st.session_state.assessment_summary else "Awaiting assessment phase completion."}

PHASE GUIDANCE:
{current_guidance}

SURGEON'S QUESTION: {user_input}

Respond as a senior surgeon to a colleague. Be precise, collegial, actionable, and practical. Under 250 words."""
    
    # Query LLM
    with st.spinner("💭 Thinking..."):
        response = query_llm(full_prompt)
    
    # Add assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "phase": st.session_state.current_phase,
        "timestamp": datetime.now().isoformat()
    })
    
    st.rerun()

elif user_input and not st.session_state.gemini_api_available:
    st.error("⚠️ LLM is currently offline. Refresh or check backend.")

# === BACKGROUND HEALTH CHECK ===

current_time = time.time()
if current_time - st.session_state.last_health_check > st.session_state.health_check_interval:
    check_system_health()
    st.session_state.last_health_check = current_time

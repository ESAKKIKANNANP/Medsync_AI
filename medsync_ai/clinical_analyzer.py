"""
Clinical Analysis Module
Processes medical scans, KAB reports, and generates professional clinical analysis
"""
import json
import base64
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class KABReport:
    """Knowledge and Brief Report"""
    patient_id: str
    age: int
    gender: str
    chief_complaint: str
    presenting_symptoms: List[str]
    medical_history: List[str]
    medications: List[str]
    allergies: List[str]
    vital_signs: Dict[str, str]
    lab_results: Dict[str, str]
    imaging_findings: List[str]
    diagnoses: List[str]
    risk_factors: List[str]

@dataclass
class MedicalScan:
    """Medical imaging data"""
    scan_id: str
    scan_type: str  # CT, MRI, X-ray, Ultrasound, etc.
    file_path: Optional[str] = None
    base64_data: Optional[str] = None
    description: str = ""

class ClinicalAnalyzer:
    """Professional clinical analysis engine using Gemini API"""
    
    def __init__(self, api_key: str):
        """Initialize with Gemini API key"""
        if not GENAI_AVAILABLE:
            raise ImportError("google-generativeai required")
        
        self.api_key = api_key
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        logger.info("Clinical Analyzer initialized with Gemini 2.0 Flash")
    
    def load_scan(self, file_path: str) -> Optional[str]:
        """Load medical scan and convert to base64"""
        try:
            with open(file_path, 'rb') as f:
                scan_data = f.read()
                return base64.b64encode(scan_data).decode()
        except Exception as e:
            logger.error(f"Failed to load scan: {e}")
            return None
    
    def analyze_patient_clinical_condition(self, 
                                          kab_report: KABReport,
                                          scans: List[MedicalScan]) -> Dict[str, Any]:
        """
        Comprehensive clinical analysis using Gemini
        
        Args:
            kab_report: Patient's Knowledge and Brief Report
            scans: List of medical scans
            
        Returns:
            Professional clinical analysis
        """
        
        # Build comprehensive patient context
        patient_context = self._build_patient_context(kab_report)
        
        # Prepare scan analysis prompt
        scan_analysis_prompt = self._prepare_scan_prompts(scans)
        
        # Main clinical analysis prompt
        analysis_prompt = f"""
You are a highly experienced physician providing professional clinical analysis.

PATIENT INFORMATION:
{patient_context}

MEDICAL IMAGING:
{scan_analysis_prompt}

Please provide a comprehensive professional clinical analysis including:

1. CLINICAL SUMMARY
   - Synthesis of presenting symptoms and findings
   - Relationship between clinical presentation and imaging

2. DIFFERENTIAL DIAGNOSES
   - Most likely diagnoses with supporting evidence
   - Alternative diagnoses to consider
   - Rationale for each diagnosis

3. CLINICAL RISK ASSESSMENT
   - Immediate risks and complications
   - Severity assessment
   - Urgency of intervention

4. RECOMMENDED INVESTIGATIONS
   - Additional imaging or labs needed
   - Priority and rationale for each

5. CLINICAL RECOMMENDATIONS
   - Immediate management steps
   - Treatment options with evidence
   - Follow-up plan
   - Patient counseling points

6. CLINICAL NOTES
   - Key clinical pearls
   - Important considerations
   - Contraindications to consider

Format your response professionally for medical records.
"""
        
        try:
            # Generate analysis with multimodal content
            content_parts = [analysis_prompt]
            
            # Add actual scan images if available
            for scan in scans:
                if scan.base64_data:
                    content_parts.append({
                        "mime_type": self._get_mime_type(scan.scan_type),
                        "data": scan.base64_data
                    })
            
            response = self.model.generate_content(content_parts)
            
            return {
                "status": "success",
                "patient_id": kab_report.patient_id,
                "analysis": response.text,
                "input_tokens_used": self._estimate_tokens(patient_context + analysis_prompt),
                "model": "gemini-2.0-flash",
                "timestamp": self._get_timestamp()
            }
        
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {
                "status": "error",
                "patient_id": kab_report.patient_id,
                "error": str(e)
            }
    
    def diagnosis_specific_analysis(self,
                                   kab_report: KABReport,
                                   suspected_diagnosis: str,
                                   scans: List[MedicalScan]) -> Dict[str, Any]:
        """
        Deep-dive analysis for specific diagnosis
        """
        
        patient_context = self._build_patient_context(kab_report)
        scan_analysis = self._prepare_scan_prompts(scans)
        
        prompt = f"""
PROFESSIONAL DIAGNOSTIC ANALYSIS

Patient: {kab_report.patient_id} | Age: {kab_report.age} | Gender: {kab_report.gender}

SUSPECTED DIAGNOSIS: {suspected_diagnosis}

PATIENT CLINICAL CONTEXT:
{patient_context}

IMAGING FINDINGS:
{scan_analysis}

As a specialist, provide a detailed analysis of whether the clinical presentation and imaging 
support the suspected diagnosis of {suspected_diagnosis}:

1. DIAGNOSTIC CRITERIA MET
   - Which diagnostic criteria are satisfied?
   - Which are lacking?

2. IMAGING CORRELATION
   - Which imaging findings support this diagnosis?
   - Are there atypical features?

3. DIFFERENTIAL CONSIDERATIONS
   - What diagnoses should be ruled out?
   - Key distinguishing features

4. DIAGNOSTIC CERTAINTY
   - Confidence level (High/Moderate/Low)
   - What additional information would increase certainty?

5. MANAGEMENT IMPLICATIONS
   - How does this diagnosis change management?
   - Prognosis considerations
   - Treatment response expectations

Provide professional medical analysis with evidence-based reasoning.
"""
        
        try:
            response = self.model.generate_content(prompt)
            
            return {
                "status": "success",
                "patient_id": kab_report.patient_id,
                "diagnosis": suspected_diagnosis,
                "analysis": response.text,
                "tokens_used": self._estimate_tokens(prompt),
                "model": "gemini-2.0-flash"
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def generate_clinical_advice(self,
                                kab_report: KABReport,
                                clinical_question: str,
                                scans: List[MedicalScan]) -> Dict[str, Any]:
        """
        Answer specific clinical questions about the patient
        """
        
        patient_context = self._build_patient_context(kab_report)
        
        prompt = f"""
CLINICAL CONSULTATION

Patient Profile:
{patient_context}

Clinical Question: {clinical_question}

Drawing from your medical expertise, analyze this patient and answer the clinical question comprehensively.

Provide:
1. Direct answer to the question
2. Clinical reasoning
3. Supporting evidence from patient data
4. Risk considerations
5. Alternative perspectives to consider
6. Recommendations

Ensure your response is professional and suitable for medical documentation.
"""
        
        try:
            response = self.model.generate_content(prompt)
            
            return {
                "status": "success",
                "patient_id": kab_report.patient_id,
                "question": clinical_question,
                "answer": response.text,
                "tokens_used": self._estimate_tokens(prompt)
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def _build_patient_context(self, kab_report: KABReport) -> str:
        """Build comprehensive patient context string"""
        
        context = f"""
DEMOGRAPHICS:
- Patient ID: {kab_report.patient_id}
- Age: {kab_report.age} years
- Gender: {kab_report.gender}

CHIEF COMPLAINT:
{kab_report.chief_complaint}

PRESENTING SYMPTOMS:
{', '.join(kab_report.presenting_symptoms)}

MEDICAL HISTORY:
{', '.join(kab_report.medical_history) if kab_report.medical_history else 'None reported'}

CURRENT MEDICATIONS:
{', '.join(kab_report.medications) if kab_report.medications else 'None'}

ALLERGIES:
{', '.join(kab_report.allergies) if kab_report.allergies else 'NKDA'}

VITAL SIGNS:
{json.dumps(kab_report.vital_signs, indent=2)}

LABORATORY RESULTS:
{json.dumps(kab_report.lab_results, indent=2)}

IMAGING FINDINGS (Summary):
{', '.join(kab_report.imaging_findings)}

CLINICAL DIAGNOSES:
{', '.join(kab_report.diagnoses)}

RISK FACTORS:
{', '.join(kab_report.risk_factors)}
"""
        return context
    
    def _prepare_scan_prompts(self, scans: List[MedicalScan]) -> str:
        """Prepare scan analysis sections"""
        
        scan_text = ""
        for scan in scans:
            scan_text += f"""
Scan: {scan.scan_type} (ID: {scan.scan_id})
Description: {scan.description}
---"""
        
        return scan_text if scan_text else "No imaging descriptions provided"
    
    def _get_mime_type(self, scan_type: str) -> str:
        """Get MIME type for scan"""
        mime_types = {
            'CT': 'image/jpeg',
            'MRI': 'image/jpeg',
            'X-ray': 'image/jpeg',
            'Ultrasound': 'image/jpeg',
            'PET': 'image/jpeg',
            'Pathology': 'image/jpeg'
        }
        return mime_types.get(scan_type, 'image/jpeg')
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation"""
        return len(text.split()) // 4  # Approximate
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()

"""
Medical File Processor and Anatomical Analyzer
Handles DICOM, PDF, and image file processing with anatomical analysis
"""

import hashlib
import json
import re
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging
import io

# Medical imaging imports
try:
    import pydicom
    DICOM_AVAILABLE = True
except ImportError:
    DICOM_AVAILABLE = False

try:
    from PyPDF2 import PdfReader
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from PIL import Image
    import numpy as np
    IMAGE_AVAILABLE = True
except ImportError:
    IMAGE_AVAILABLE = False

import streamlit as st

logger = logging.getLogger(__name__)

# ============================================================================
# MEDICAL FILE PROCESSOR
# ============================================================================

class MedicalFileProcessor:
    """Process medical files: DICOM, PDF, PNG, JPG"""
    
    SUPPORTED_FORMATS = {'.pdf', '.dcm', '.png', '.jpg', '.jpeg'}
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
    
    @staticmethod
    def validate_file(uploaded_file) -> Tuple[bool, str]:
        """Validate file type and size"""
        # Check size
        if uploaded_file.size > MedicalFileProcessor.MAX_FILE_SIZE:
            return False, f"File exceeds {MedicalFileProcessor.MAX_FILE_SIZE / 1024 / 1024:.0f}MB limit"
        
        # Check extension
        name_lower = uploaded_file.name.lower()
        if not any(name_lower.endswith(fmt) for fmt in MedicalFileProcessor.SUPPORTED_FORMATS):
            return False, f"Unsupported format. Supported: {', '.join(MedicalFileProcessor.SUPPORTED_FORMATS)}"
        
        return True, "Valid"
    
    @staticmethod
    def get_file_type(filename: str) -> str:
        """Determine file type from extension"""
        filename_lower = filename.lower()
        if filename_lower.endswith('.pdf'):
            return 'pdf'
        elif filename_lower.endswith('.dcm'):
            return 'dicom'
        elif filename_lower.endswith(('.png', '.jpg', '.jpeg')):
            return 'image'
        return 'unknown'
    
    @staticmethod
    def extract_pdf_text(uploaded_file) -> str:
        """Extract text from PDF using PyPDF2"""
        if not PDF_AVAILABLE:
            return "PDF processing not available. Install PyPDF2."
        
        try:
            pdf_reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            logger.error(f"PDF extraction error: {e}")
            return f"Error extracting PDF: {str(e)}"
    
    @staticmethod
    def load_dicom_image(uploaded_file) -> Optional[np.ndarray]:
        """Load DICOM image with pydicom"""
        if not DICOM_AVAILABLE:
            return None
        
        try:
            dicom_data = pydicom.dcmread(io.BytesIO(uploaded_file.getvalue()))
            
            # Get pixel array
            if hasattr(dicom_data, 'pixel_array'):
                pixel_array = dicom_data.pixel_array
                
                # Normalize to 0-255 range for display
                if pixel_array.max() > 255:
                    pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min()) * 255
                
                return pixel_array.astype(np.uint8)
        except Exception as e:
            logger.error(f"DICOM loading error: {e}")
        
        return None
    
    @staticmethod
    def extract_dicom_metadata(uploaded_file) -> Dict[str, Any]:
        """Extract comprehensive DICOM metadata for clinical analysis"""
        if not DICOM_AVAILABLE:
            return {"error": "pydicom not available"}
        
        try:
            dicom_data = pydicom.dcmread(io.BytesIO(uploaded_file.getvalue()))
            
            metadata = {
                "filename": uploaded_file.name,
                "file_size_kb": uploaded_file.size / 1024,
                "upload_time": datetime.now().isoformat()
            }
            
            # Patient demographics
            if hasattr(dicom_data, 'PatientName'):
                metadata["patient_name"] = str(dicom_data.PatientName)
            if hasattr(dicom_data, 'PatientID'):
                metadata["patient_id"] = str(dicom_data.PatientID)
            if hasattr(dicom_data, 'PatientAge'):
                metadata["patient_age"] = str(dicom_data.PatientAge)
            if hasattr(dicom_data, 'PatientSex'):
                metadata["patient_sex"] = str(dicom_data.PatientSex)
            
            # Study/Series information
            if hasattr(dicom_data, 'StudyDate'):
                metadata["study_date"] = str(dicom_data.StudyDate)
            if hasattr(dicom_data, 'SeriesDate'):
                metadata["series_date"] = str(dicom_data.SeriesDate)
            if hasattr(dicom_data, 'StudyDescription'):
                metadata["study_description"] = str(dicom_data.StudyDescription)
            if hasattr(dicom_data, 'SeriesDescription'):
                metadata["series_description"] = str(dicom_data.SeriesDescription)
            if hasattr(dicom_data, 'SeriesNumber'):
                metadata["series_number"] = int(dicom_data.SeriesNumber)
            
            # Modality
            if hasattr(dicom_data, 'Modality'):
                metadata["modality"] = str(dicom_data.Modality)
            
            # Image properties
            if hasattr(dicom_data, 'Rows'):
                metadata["image_rows"] = int(dicom_data.Rows)
            if hasattr(dicom_data, 'Columns'):
                metadata["image_columns"] = int(dicom_data.Columns)
            if hasattr(dicom_data, 'BitsAllocated'):
                metadata["bits_allocated"] = int(dicom_data.BitsAllocated)
            if hasattr(dicom_data, 'BitsStored'):
                metadata["bits_stored"] = int(dicom_data.BitsStored)
            
            # Protocol/body part
            if hasattr(dicom_data, 'ProtocolName'):
                metadata["protocol"] = str(dicom_data.ProtocolName)
            if hasattr(dicom_data, 'BodyPartExamined'):
                metadata["body_part"] = str(dicom_data.BodyPartExamined)
            
            # Manufacturer and device info
            if hasattr(dicom_data, 'Manufacturer'):
                metadata["manufacturer"] = str(dicom_data.Manufacturer)
            if hasattr(dicom_data, 'ManufacturersModelName'):
                metadata["model"] = str(dicom_data.ManufacturersModelName)
            
            metadata["extraction_status"] = "success"
            return metadata
            
        except Exception as e:
            logger.error(f"DICOM metadata extraction error: {e}")
            return {
                "filename": uploaded_file.name,
                "error": str(e),
                "extraction_status": "failed"
            }
    
    @staticmethod
    def load_image(uploaded_file) -> Optional[Image.Image]:
        """Load standard image (PNG, JPG)"""
        if not IMAGE_AVAILABLE:
            return None
        
        try:
            image = Image.open(io.BytesIO(uploaded_file.getvalue()))
            return image
        except Exception as e:
            logger.error(f"Image loading error: {e}")
        
        return None
    
    @staticmethod
    def convert_to_base64(file_data: bytes) -> str:
        """Convert file to base64 for API transmission"""
        import base64
        return base64.b64encode(file_data).decode('utf-8')

# ============================================================================
# SCAN FINDINGS EXTRACTOR
# ============================================================================

class ScanFindingsExtractor:
    """Extract and synthesize clinical findings from medical scans"""
    
    # Common pathological findings by modality
    SCAN_FINDINGS_TEMPLATES = {
        'CT': {
            'normal': 'No acute findings. Normal liver attenuation and morphology.',
            'disease_keywords': ['cirrhosis', 'lesion', 'infiltration', 'inflammation', 'edema', 'ascites', 'stones'],
            'vessel_keywords': ['portal hypertension', 'thrombosis', 'stenosis', 'collaterals']
        },
        'MRI': {
            'normal': 'No acute signal abnormalities. Normal hepatic echotexture.',
            'disease_keywords': ['hyperintense', 'hypointense', 'signal', 'enhancement', 'restriction', 'fibrosis'],
            'vessel_keywords': ['flow void', 'patency', 'thrombosis']
        },
        'XR': {
            'normal': 'Standard anatomy, no acute findings.',
            'disease_keywords': ['opacification', 'lucency', 'obstruction', 'perforation', 'calcification'],
            'vessel_keywords': ['aortic', 'vascular', 'stent']
        },
        'US': {
            'normal': 'Normal echotexture without focal lesion.',
            'disease_keywords': ['hyperechoic', 'hypoechoic', 'anechoic', 'heterogeneous', 'shadowing'],
            'vessel_keywords': ['flow', 'patent', 'stenosis', 'resistive index']
        }
    }
    
    @staticmethod
    def analyze_scan(uploaded_file, metadata: Dict = None) -> Dict[str, Any]:
        """
        Analyze uploaded scan and extract key findings
        
        Args:
            uploaded_file: Streamlit UploadedFile
            metadata: DICOM metadata dict (optional)
        
        Returns:
            Dict with findings, risk factors, and clinical summary
        """
        
        file_type = MedicalFileProcessor.get_file_type(uploaded_file.name)
        findings = {
            'filename': uploaded_file.name,
            'file_type': file_type,
            'timestamp': datetime.now().isoformat(),
            'modality': 'Unknown',
            'findings_summary': '',
            'anatomical_concerns': [],
            'risk_flags': [],
            'clinical_recommendations': []
        }
        
        try:
            if file_type == 'dicom':
                if metadata is None:
                    metadata = MedicalFileProcessor.extract_dicom_metadata(uploaded_file)
                
                findings['modality'] = metadata.get('modality', 'CT')
                findings['metadata'] = metadata
                
                # Extract clinical findings from metadata descriptions
                series_desc = metadata.get('series_description', '').lower()
                study_desc = metadata.get('study_description', '').lower()
                combined_desc = f"{series_desc} {study_desc}"
                
                findings = ScanFindingsExtractor._analyze_description_text(
                    combined_desc, findings, findings['modality']
                )
                
            elif file_type == 'pdf':
                text = MedicalFileProcessor.extract_pdf_text(uploaded_file)
                findings['extracted_text'] = text[:500]
                findings = ScanFindingsExtractor._analyze_description_text(
                    text.lower(), findings, 'PDF Report'
                )
                
            elif file_type == 'image':
                findings['modality'] = 'Image'
                findings['image_loaded'] = True
                findings['findings_summary'] = f"Image file '{uploaded_file.name}' loaded for radiologist review."
                findings['clinical_recommendations'].append("Radiologist review required for detailed interpretation")
            
            return findings
            
        except Exception as e:
            logger.error(f"Scan analysis error: {e}")
            findings['error'] = str(e)
            findings['findings_summary'] = f"Error analyzing scan: {str(e)}"
            return findings
    
    @staticmethod
    def _analyze_description_text(text: str, findings: Dict, modality: str) -> Dict:
        """
        Analyze textual description for clinical findings
        
        Args:
            text: Scan description or extracted text
            findings: Current findings dict
            modality: Imaging modality
        
        Returns:
            Updated findings dict
        """
        
        template = ScanFindingsExtractor.SCAN_FINDINGS_TEMPLATES.get(modality, {})
        
        # Check for disease keywords
        disease_found = False
        disease_mentions = []
        
        for keyword in template.get('disease_keywords', []):
            if keyword in text:
                disease_found = True
                disease_mentions.append(keyword)
        
        # Check for vascular keywords
        vascular_concerns = []
        for keyword in template.get('vessel_keywords', []):
            if keyword in text:
                vascular_concerns.append(keyword)
        
        # Generate findings summary
        if disease_found:
            findings['findings_summary'] = f"Abnormal findings identified: {', '.join(set(disease_mentions))}. Detailed interpretation required."
            findings['risk_flags'].append("Abnormal findings - High risk case")
        else:
            findings['findings_summary'] = f"Scan reviewed. {template.get('normal', 'No acute findings noted.')} Detailed radiologist report recommended."
        
        if vascular_concerns:
            findings['anatomical_concerns'].extend([f"Vascular concern: {v}" for v in vascular_concerns])
            findings['risk_flags'].append("Vascular complexity - Surgical approach adjustment needed")
            findings['clinical_recommendations'].append("Consider vascular surgery consultation")
        
        # Add general recommendations
        if disease_found or vascular_concerns:
            findings['clinical_recommendations'].append("Pre-operative imaging conference recommended")
            findings['clinical_recommendations'].append("Consider advanced hemostasis planning")
        
        return findings
    
    @staticmethod
    def synthesize_clinical_context(scans_data: List[Dict]) -> str:
        """
        Synthesize clinical context from all uploaded scans
        
        Args:
            scans_data: List of scan findings dicts
        
        Returns:
            Synthesized clinical context string
        """
        
        context = "## MULTIMODAL SCAN SYNTHESIS\n\n"
        
        if not scans_data:
            return context + "No scans uploaded yet.\n"
        
        # Compile findings
        context += f"**Scans Reviewed**: {len(scans_data)} imaging studies\n\n"
        
        all_findings = []
        all_concerns = []
        all_flags = []
        
        for scan in scans_data:
            if scan.get('findings_summary'):
                all_findings.append(scan['findings_summary'])
            all_concerns.extend(scan.get('anatomical_concerns', []))
            all_flags.extend(scan.get('risk_flags', []))
        
        if all_findings:
            context += "**Imaging Findings**:\n"
            for finding in all_findings:
                context += f"- {finding}\n"
            context += "\n"
        
        if all_concerns:
            context += "**Anatomical Considerations**:\n"
            for concern in set(all_concerns):
                context += f"⚠️ {concern}\n"
            context += "\n"
        
        if all_flags:
            context += "**Risk Flags**:\n"
            for flag in set(all_flags):
                context += f"🚩 {flag}\n"
        
        return context


# ============================================================================
# ANATOMICAL ANALYZER WITH CHOLECT45 WORKFLOW
# ============================================================================

class AnatomicalAnalyzer:
    """Analyze anatomical structures and surgical workflow"""
    
    # Hepatobiliary surgery anatomical landmarks
    ANATOMICAL_LANDMARKS = {
        'liver_segments': {
            'Segment I': 'Caudate lobe',
            'Segment II': 'Left lateral superior',
            'Segment III': 'Left lateral inferior',
            'Segment IV': 'Left medial',
            'Segment V': 'Right anterior inferior',
            'Segment VI': 'Right posterior inferior',
            'Segment VII': 'Right posterior superior',
            'Segment VIII': 'Right anterior superior'
        },
        'vascular_structures': {
            'Hepatic artery proper': 'Supplies oxygenated blood to liver',
            'Cystic artery': 'Supplies gallbladder',
            'Hepatic vein': 'Drains liver',
            'Portal vein': 'Brings nutrient-rich blood from GI tract'
        },
        'biliary_structures': {
            'Common hepatic duct': 'Formed by union of right and left hepatic ducts',
            'Cystic duct': 'Drains gallbladder',
            'Common bile duct': 'Carries bile to small intestine',
            'Ampulla of Vater': 'Opening into duodenum'
        }
    }
    
    # CholecT45: 9-step hepatobiliary surgical workflow
    SURGICAL_STEPS = [
        {
            'step': 1,
            'name': 'Preparation & Patient Positioning',
            'description': 'Patient supine, prepare surgical field, establish IV access for anesthesia',
            'instruments': ['Electrosurgical unit', 'Suction', 'Light source'],
            'anatomy': ['Skin', 'Subcutaneous tissue', 'Fascia'],
            'risks': [
                {'severity': 'low', 'risk': 'Skin irritation', 'mitigation': 'Use protective drapes'}
            ]
        },
        {
            'step': 2,
            'name': 'Trocar Placement',
            'description': 'Create 4 laparoscopic ports (Epigastric, Right midclavicular, Right anterior axillary, subxiphoid)',
            'instruments': ['Trocars (5mm, 10-12mm)', 'Camera', 'Light source'],
            'anatomy': ['Abdominal wall', 'Peritoneum'],
            'risks': [
                {'severity': 'medium', 'risk': 'Vascular or bowel injury', 'mitigation': 'Use open Hasson technique for first port'},
                {'severity': 'medium', 'risk': 'Gas embolism', 'mitigation': 'Correct trocar angle and depth'}
            ]
        },
        {
            'step': 3,
            'name': 'Insufflation & Initial Exploration',
            'description': 'Establish pneumoperitoneum (12-15 mmHg CO2), perform systematic abdominal exploration',
            'instruments': ['CO2 insufflator', 'Camera', 'Laparoscope'],
            'anatomy': ['Peritoneal cavity', 'Liver', 'Gallbladder', 'Hepatic vasculature'],
            'risks': []
        },
        {
            'step': 4,
            'name': 'Position Gallbladder & Acquire Critical View of Safety',
            'description': 'Grasp fundus, retract cephalad to achieve critical view of safety (CVS) - 2 ducts and 2 artery',
            'instruments': ['Graspers', 'Retractors', 'Monopolar cautery'],
            'anatomy': ['Cystic artery', 'Cystic vein', 'Common bile duct', 'Right hepatic artery'],
            'risks': [
                {'severity': 'high', 'risk': 'Misidentification of anatomy', 'mitigation': 'Follow CVS protocol - clear triangle, 2 structures only'},
                {'severity': 'high', 'risk': 'Anomalous cystic artery', 'mitigation': 'Be aware of variations - may arise from RHA or anterior'}
            ]
        },
        {
            'step': 5,
            'name': 'Cystic Artery Division',
            'description': 'Divide cystic artery using cautery hook, clip, or vessel sealer after confirming only 2 structures in triangle',
            'instruments': ['Monopolar/bipolar cautery', 'Clips', 'Harmonic scalpel', 'Vessel sealer'],
            'anatomy': ['Cystic artery', 'Cystic vein', 'Common bile duct'],
            'risks': [
                {'severity': 'high', 'risk': 'Right hepatic artery damage', 'mitigation': 'Confirm only cystic artery present'},
                {'severity': 'medium', 'risk': 'Bleeding from artery', 'mitigation': 'Have clips ready'}
            ]
        },
        {
            'step': 6,
            'name': 'Clear Triangle of Calot',
            'description': 'Complete dissection of triangle of Calot to ensure clear view before duct division',
            'instruments': ['Electrosurgery hook', 'Graspers', 'Scissors'],
            'anatomy': ['Triangle of Calot', 'Liver bed', 'Peritoneum', 'Right hepatic artery'],
            'risks': [
                {'severity': 'medium', 'risk': 'Accessory ducts', 'mitigation': 'Carefully inspect for small ducts'},
                {'severity': 'low', 'risk': 'Thermal injury to duct', 'mitigation': 'Use careful electrosurgery technique'}
            ]
        },
        {
            'step': 7,
            'name': 'Cystic Duct Division',
            'description': 'Clip or divide cystic duct after CVS confirmation and careful dissection',
            'instruments': ['Clips', 'Stapler', 'Electrocautery', 'Vessel sealer'],
            'anatomy': ['Cystic duct', 'Common bile duct', 'Liver'],
            'risks': [
                {'severity': 'high', 'risk': 'CBD injury', 'mitigation': 'Ensure 2 clear ducts visible; CBD enters separately'},
                {'severity': 'medium', 'risk': 'Bile leak', 'mitigation': 'Ensure secure clip/staple placement'}
            ]
        },
        {
            'step': 8,
            'name': 'Gallbladder Dissection & Removal',
            'description': 'Dissect gallbladder off liver bed using electrosurgery, place in specimen bag, remove via epigastric port',
            'instruments': ['Electrosurgery hook/spatula', 'Specimen bag', 'Graspers'],
            'anatomy': ['Liver bed', 'Peritoneum', 'Cystic artery stump'],
            'risks': [
                {'severity': 'medium', 'risk': 'Gallbladder perforation/bile spill', 'mitigation': 'Use specimen bag if perforation occurs'},
                {'severity': 'low', 'risk': 'Liver capsule injury', 'mitigation': 'Gentle dissection technique'}
            ]
        },
        {
            'step': 9,
            'name': 'Closure & Hemostasis',
            'description': 'Irrigate liver bed, ensure hemostasis, remove trocars under direct visualization, close fasciae (>10mm) and skin',
            'instruments': ['Irrigator', 'Suction', 'Electrocautery', 'Clips', 'Sutures'],
            'anatomy': ['Liver bed', 'Trocar sites', 'Fascia', 'Skin'],
            'risks': [
                {'severity': 'medium', 'risk': 'Bile leak from small ducts', 'mitigation': 'Irrigate and ensure hemostasis'},
                {'severity': 'low', 'risk': 'Port site hernia', 'mitigation': 'Close fasciae on 10mm+ ports with 0-Vicryl'}
            ]
        }
    ]
    
    @staticmethod
    def identify_anatomical_findings(clinical_text: str) -> Dict:
        """Identify anatomical findings from clinical text"""
        findings = {
            'liver_segments': [],
            'vascular_involvement': [],
            'surgical_complexity': 'Standard'
        }
        
        text_lower = clinical_text.lower()
        
        # Check for liver segments mentioned
        for segment in AnatomicalAnalyzer.ANATOMICAL_LANDMARKS['liver_segments'].keys():
            if segment.lower() in text_lower:
                findings['liver_segments'].append(segment)
        
        # Check for vascular findings
        vascular_keywords = {
            'portal vein thrombosis': 'thrombosis|clot',
            'hepatic artery stenosis': 'stenosis',
            'anomalous vessels': 'variant|anomalous|anomaly'
        }
        
        for condition, keywords in vascular_keywords.items():
            if re.search(keywords, text_lower):
                findings['vascular_involvement'].append(condition)
        
        # Assess complexity
        complex_indicators = [
            'cirrhosis', 'fibrosis', 'previous surgery', 'inflammation',
            'variant', 'anomaly', 'adhesions', 'difficult'
        ]
        
        if any(indicator in text_lower for indicator in complex_indicators):
            findings['surgical_complexity'] = 'Complex'
        
        return findings
    
    @staticmethod
    def get_next_surgical_step(current_step_idx: int, patient_anatomy: Dict) -> Dict:
        """Get current surgical step with anatomy-based risk assessment"""
        if current_step_idx >= len(AnatomicalAnalyzer.SURGICAL_STEPS):
            return AnatomicalAnalyzer.SURGICAL_STEPS[-1]
        
        step = AnatomicalAnalyzer.SURGICAL_STEPS[current_step_idx].copy()
        
        # Add anatomy-specific risks for steps 4-7 (critical view steps)
        if current_step_idx in [3, 4, 5, 6]:  # Steps 4-7 (0-indexed)
            if patient_anatomy.get('vascular_involvement'):
                for involvement in patient_anatomy['vascular_involvement']:
                    step['risks'].append({
                        'severity': 'high',
                        'risk': f'Patient has {involvement}',
                        'mitigation': 'Proceed with extreme caution. Consider opening if unclear.'
                    })
        
        return step

# ============================================================================
# PRE-OP BRIEF GENERATOR
# ============================================================================

class PreOpBriefGenerator:
    """Generate comprehensive Pre-Operative Briefings"""
    
    @staticmethod
    def generate_brief(
        patient_data: Dict,
        clinical_text: str = "",
        anatomical_findings: Dict = None
    ) -> Dict:
        """Generate comprehensive pre-op brief"""
        
        if anatomical_findings is None:
            anatomical_findings = {
                'liver_segments': [],
                'vascular_involvement': [],
                'surgical_complexity': 'Standard'
            }
        
        # Risk factor assessment
        risk_factors = []
        risk_score = 0
        
        if patient_data.get('on_anticoagulation'):
            risk_factors.append("On anticoagulation - bleeding risk")
            risk_score += 20
        if patient_data.get('on_aspirin'):
            risk_factors.append("On aspirin - platelet dysfunction")
            risk_score += 10
        if patient_data.get('has_coagulopathy'):
            risk_factors.append("Known coagulopathy - bleeding disorder")
            risk_score += 25
        if patient_data.get('platelet_count', 150000) < 100000:
            risk_factors.append(f"Low platelets ({patient_data.get('platelet_count', 0)}/µL) - bleeding risk")
            risk_score += 15
        if patient_data.get('has_renal_impairment'):
            risk_factors.append("Renal impairment - medication metabolism affected")
            risk_score += 10
        if patient_data.get('has_hepatic_impairment'):
            risk_factors.append("Hepatic impairment - complex case")
            risk_score += 20
        if patient_data.get('mh_family_history'):
            risk_factors.append("Positive surgical family history")
            risk_score += 5
        
        # Age-related risk
        age = patient_data.get('age', 0)
        if age > 70:
            risk_factors.append(f"Advanced age ({age}y) - increased operative risk")
            risk_score += 10
        
        # Surgical complexity contribution
        if anatomical_findings.get('surgical_complexity') == 'Complex':
            risk_score += 15
        
        risk_score = min(95, risk_score)  # Cap at 95%
        
        # Build brief sections
        brief = {
            'patient_summary': {
                'id': patient_data.get('patient_id', 'Unknown'),
                'age': age,
                'risk_factors': risk_factors,
                'risk_score': risk_score
            },
            'clinical_summary': PreOpBriefGenerator._generate_clinical_summary(clinical_text),
            'anatomical_summary': PreOpBriefGenerator._generate_anatomical_summary(anatomical_findings),
            'procedure_plan': PreOpBriefGenerator._generate_procedure_plan(anatomical_findings),
            'critical_alerts': PreOpBriefGenerator._generate_alerts(patient_data, risk_factors),
            'equipment_prep': PreOpBriefGenerator._generate_equipment_list(anatomical_findings)
        }
        
        return brief
    
    @staticmethod
    def _generate_clinical_summary(text: str) -> str:
        """Generate clinical summary from text"""
        if not text:
            return "No clinical text provided. Standard preoperative evaluation recommended."
        
        summary = text[:500]
        if len(text) > 500:
            summary += "...\n\n[See full clinical report for complete details]"
        
        return summary
    
    @staticmethod
    def _generate_anatomical_summary(findings: Dict) -> str:
        """Generate anatomical summary"""
        summary = "## Anatomical Assessment\n\n"
        
        if findings.get('liver_segments'):
            summary += f"**Liver Involvement**: {', '.join(findings['liver_segments'])}\n\n"
        
        if findings.get('vascular_involvement'):
            summary += f"⚠️ **Vascular Complexity**: {', '.join(findings['vascular_involvement'])}\n\n"
        
        summary += f"**Surgical Complexity**: {findings.get('surgical_complexity', 'Standard')}\n"
        
        return summary
    
    @staticmethod
    def _generate_procedure_plan(findings: Dict) -> str:
        """Generate surgical procedure plan"""
        complexity = findings.get('surgical_complexity', 'Standard')
        
        if complexity == 'Complex':
            plan = """## Recommended Approach

1. **Pre-Incision Briefing**: Full team review of anatomy and risks
2. **Careful Dissection**: Slow progression with frequent anatomical confirmation
3. **Critical View of Safety (CVS)**: Mandatory before duct division
4. **Conversion Threshold**: Low threshold for conversion to open approach
5. **Intra-operative Imaging**: Consider cholangiography if anatomy unclear

**Key Consideration**: This patient's anatomy or condition warrants heightened vigilance."""
        else:
            plan = """## Standard Laparoscopic Cholecystectomy Plan

1. **Patient positioning**: Supine, left side elevated (reverse Trendelenburg prepared)
2. **Trocar placement**: Standard 4-port approach
3. **Critical View of Safety (CVS)**: Confirm before any duct/artery division
4. **Standard dissection**: Cystic artery first, then duct
5. **Hemostasis and closure**: Standard port closure

**Expected Duration**: 45-90 minutes
**Expected Blood Loss**: Minimal"""
        
        return plan
    
    @staticmethod
    def _generate_alerts(patient_data: Dict, risk_factors: List) -> List[Dict]:
        """Generate critical alerts"""
        alerts = []
        
        if patient_data.get('has_coagulopathy') or patient_data.get('on_anticoagulation'):
            alerts.append({
                'alert': 'Bleeding Risk - Coagulation Disorder Identified',
                'action': 'Have FFP, cryoprecipitate, and transfusion protocol ready. Consider cell salvage.'
            })
        
        if patient_data.get('platelet_count', 150000) < 50000:
            alerts.append({
                'alert': 'Severe Thrombocytopenia - Transfusion Available',
                'action': 'Type and cross. Have platelets standing by.'
            })
        
        if patient_data.get('has_hepatic_impairment'):
            alerts.append({
                'alert': 'Hepatic Impairment - Modified Approach',
                'action': 'Careful dissection to avoid liver injury. Monitor lactate/INR.'
            })
        
        if patient_data.get('age', 0) > 80:
            alerts.append({
                'alert': 'Advanced Age - Increased Perioperative Risk',
                'action': 'ICU monitoring post-op. Reduced operative time is goal.'
            })
        
        return alerts
    
    @staticmethod
    def _generate_equipment_list(findings: Dict) -> List[str]:
        """Generate equipment preparation checklist"""
        equipment = [
            "✓ Laparoscopic tower (working)",
            "✓ 4 trocars (5mm, 10/12mm available)",
            "✓ High-definition camera system",
            "✓ Instruments: Graspers, dissectors, clips, cautery hook",
            "✓ Light source with backup",
            "✓ Suction/irrigation system",
            "✓ CO2 insufflator with flow meter"
        ]
        
        if findings.get('surgical_complexity') == 'Complex':
            equipment.extend([
                "⚠ Open surgical instruments (conversion kit ready)",
                "⚠ Cholangiography tube and imaging capability",
                "⚠ Vascular clamps available"
            ])
        
        equipment.extend([
            "✓ Specimen bag",
            "✓ Closure materials: Fascial sutures (0-Vicryl), skin closure"
        ])
        
        return equipment

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def initialize_multimodal_session_state():
    """Initialize multimodal session state in Streamlit"""
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
# UTILITY FUNCTIONS
# ============================================================================

def get_file_hash(uploaded_file) -> str:
    """Generate hash of file for caching"""
    file_data = uploaded_file.getvalue()
    return hashlib.md5(file_data).hexdigest()[:8]

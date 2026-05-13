"""
FastAPI REST API for MedSync AI
Provides endpoints for surgeon queries, roadmap management, and real-time monitoring
"""
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import json
import base64
from medsync_ai.utils.logger import get_logger
from medsync_ai.config.settings import API_HOST, API_PORT, GEMINI_API_KEY
from medsync_ai.rag_engine import SurgeonDoubtRAGEngine, LLMClient
from medsync_ai.surgical_roadmap import SurgicalRoadmapGenerator
from medsync_ai.surgical_vision import VQLAIntegration
from medsync_ai.vector_db import VectorDatabase
from medsync_ai.clinical_analyzer import ClinicalAnalyzer, KABReport, MedicalScan

logger = get_logger(__name__)

# Request/Response Models
class SurgeonQuery(BaseModel):
    """Surgeon query request"""
    query: str
    patient_id: Optional[str] = None
    surgical_scene: Optional[str] = None
    current_frame: Optional[str] = None

class SurgeonQueryResponse(BaseModel):
    """Response to surgeon query"""
    query: str
    response: str
    hemorrhage_risk_score: str
    anesthetic_risk_score: str
    hemorrhage_alert: bool
    anesthetic_alert: bool
    retrieved_docs: int

class RoadmapRequest(BaseModel):
    """Request to initialize surgical roadmap"""
    procedure_type: str
    patient_id: Optional[str] = None

class RoadmapUpdate(BaseModel):
    """Update current roadmap step"""
    action: str  # "next_step", "add_alert"
    alert_data: Optional[Dict] = None

class PatientData(BaseModel):
    """Patient clinical data"""
    patient_id: str
    age: int
    on_anticoagulation: bool = False
    on_aspirin: bool = False
    platelet_count: Optional[int] = None
    has_coagulopathy: bool = False
    mh_family_history: bool = False
    has_renal_impairment: bool = False
    has_hepatic_impairment: bool = False
    hemorrhage_risk: float = 0.0
    anesthetic_reaction_risk: float = 0.0

class KABReportRequest(BaseModel):
    """Knowledge and Brief Report Request"""
    patient_id: str
    age: int
    gender: str
    chief_complaint: str
    presenting_symptoms: List[str]
    medical_history: List[str] = []
    medications: List[str] = []
    allergies: List[str] = []
    vital_signs: Dict[str, str]
    lab_results: Dict[str, str] = {}
    imaging_findings: List[str] = []
    diagnoses: List[str] = []
    risk_factors: List[str] = []

class MedicalScanInfo(BaseModel):
    """Medical scan information"""
    scan_id: str
    scan_type: str  # CT, MRI, X-ray, Ultrasound, PET, Pathology
    description: str = ""
    base64_data: Optional[str] = None  # Base64 encoded image

class ClinicalAnalysisRequest(BaseModel):
    """Request for clinical analysis"""
    kab_report: KABReportRequest
    medical_scans: List[MedicalScanInfo] = []

class DiagnosisAnalysisRequest(BaseModel):
    """Request for diagnosis-specific analysis"""
    kab_report: KABReportRequest
    suspected_diagnosis: str
    medical_scans: List[MedicalScanInfo] = []

class ClinicalQuestionRequest(BaseModel):
    """Request to answer clinical questions"""
    kab_report: KABReportRequest
    clinical_question: str
    medical_scans: List[MedicalScanInfo] = []

class MedSyncAPI:
    """MedSync AI API Server"""
    
    def __init__(self):
        """Initialize API"""
        self.app = FastAPI(
            title="MedSync AI - Multi-Modal Surgical Intelligence",
            description="RAG-powered surgical decision support system",
            version="0.1.0"
        )
        
        # Initialize components
        self.vector_db = VectorDatabase()
        self.clinical_analyzer = ClinicalAnalyzer(api_key=GEMINI_API_KEY)
        self.llm_client = LLMClient()
        self.rag_engine = SurgeonDoubtRAGEngine(self.vector_db, self.llm_client)
        self.vqla = VQLAIntegration()
        
        # Active sessions
        self.patient_sessions: Dict[str, Dict] = {}
        self.roadmaps: Dict[str, SurgicalRoadmapGenerator] = {}
        
        self._setup_routes()
        logger.info("MedSync API initialized")
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/health")
        def health_check():
            """Health check endpoint"""
            return {
                "status": "ok",
                "gemini_api_available": self.llm_client.is_available(),
                "vector_db_collections": self.vector_db.get_collection_stats()
            }
        
        @self.app.post("/api/patient/register")
        def register_patient(patient_data: PatientData):
            """Register patient for surgical session"""
            self.patient_sessions[patient_data.patient_id] = {
                "data": patient_data.dict(),
                "created_at": str(__import__("datetime").datetime.now())
            }
            logger.info(f"Patient registered: {patient_data.patient_id}")
            return {"status": "ok", "patient_id": patient_data.patient_id}
        
        @self.app.post("/api/surgeon/query")
        async def surgeon_query(request: SurgeonQuery):
            """
            Process surgeon query with RAG engine
            """
            try:
                patient_data = None
                if request.patient_id and request.patient_id in self.patient_sessions:
                    patient_data = self.patient_sessions[request.patient_id]["data"]
                
                result = self.rag_engine.answer_surgeon_query(
                    query=request.query,
                    patient_data=patient_data,
                    surgical_scene=request.surgical_scene
                )
                
                return SurgeonQueryResponse(
                    query=result["query"],
                    response=result["response"],
                    hemorrhage_risk_score=result["hemorrhage_risk_score"],
                    anesthetic_risk_score=result["anesthetic_risk_score"],
                    hemorrhage_alert=result["hemorrhage_alert"],
                    anesthetic_alert=result["anesthetic_alert"],
                    retrieved_docs=result["retrieved_clinical_docs"]
                )
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/analyze-report")
        async def analyze_report(request: dict):
            """
            Extract findings from medical report images using Gemini vision
            """
            try:
                import base64
                from io import BytesIO
                from PIL import Image
                
                # Decode base64 image
                image_data = base64.b64decode(request.get("image_base64", ""))
                image = Image.open(BytesIO(image_data))
                
                # Generate prompt for report analysis
                prompt = f"""You are a medical document analysis expert. 
Analyze this medical report image and extract key clinical findings, values, measurements, and diagnoses.

Provide a concise, structured summary of:
1. **Report Type**: What type of report is this? (e.g., Lab report, CT scan, X-ray, MRI, USG, etc.)
2. **Key Findings**: Main clinical findings or abnormalities
3. **Measurements/Values**: Any specific measurements, lab values, or numeric data
4. **Clinical Impression**: Overall impression or diagnosis if stated
5. **Recommendations**: Any recommendations mentioned

Format as clear bullet points for easy integration into surgical planning."""
                
                # Use Gemini to analyze the image
                response_text = self.llm_client.generate(
                    prompt=prompt,
                    images=[image]
                )
                
                logger.info(f"Report analyzed: {request.get('filename', 'unknown')}")
                
                return {
                    "status": "ok",
                    "extracted_findings": response_text
                }
            
            except Exception as e:
                logger.error(f"Error analyzing report: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/roadmap/initialize")
        def initialize_roadmap(request: RoadmapRequest):
            """Initialize surgical roadmap for procedure"""
            try:
                roadmap = SurgicalRoadmapGenerator(request.procedure_type)
                roadmap_id = request.patient_id or f"roadmap_{len(self.roadmaps)}"
                self.roadmaps[roadmap_id] = roadmap
                
                current_step = roadmap.get_current_step()
                return {
                    "roadmap_id": roadmap_id,
                    "procedure": request.procedure_type,
                    "total_steps": len(roadmap.steps),
                    "current_step": {
                        "number": 1,
                        "phase": current_step.phase.value,
                        "description": current_step.description
                    }
                }
            except Exception as e:
                logger.error(f"Error initializing roadmap: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/roadmap/{roadmap_id}")
        def get_roadmap(roadmap_id: str):
            """Get current roadmap status"""
            if roadmap_id not in self.roadmaps:
                raise HTTPException(status_code=404, detail="Roadmap not found")
            
            roadmap = self.roadmaps[roadmap_id]
            return roadmap.generate_roadmap_report()
        
        @self.app.post("/api/roadmap/{roadmap_id}/update")
        def update_roadmap(roadmap_id: str, update: RoadmapUpdate):
            """Update roadmap state"""
            if roadmap_id not in self.roadmaps:
                raise HTTPException(status_code=404, detail="Roadmap not found")
            
            roadmap = self.roadmaps[roadmap_id]
            
            try:
                if update.action == "next_step":
                    roadmap.advance_to_next_step()
                
                elif update.action == "add_alert":
                    if update.alert_data:
                        roadmap.add_complication_alert(
                            severity=update.alert_data.get("severity", "medium"),
                            risk_type=update.alert_data.get("risk_type", "unknown"),
                            description=update.alert_data.get("description", ""),
                            risk_score=update.alert_data.get("risk_score", 0.5),
                            mitigation_steps=update.alert_data.get("mitigation_steps", [])
                        )
                
                return roadmap.generate_roadmap_report()
            except Exception as e:
                logger.error(f"Error updating roadmap: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/surgical-vision/{frame_id}")
        def analyze_frame(frame_id: str):
            """Analyze surgical frame for instruments and structures"""
            try:
                analysis = self.vqla.analyze_frame(frame_id)
                return analysis
            except Exception as e:
                logger.error(f"Error analyzing frame: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/vectordb/stats")
        def get_vectordb_stats():
            """Get vector database statistics"""
            return self.vector_db.get_collection_stats()
        
        # =====================================================================
        # CLINICAL ANALYSIS ENDPOINTS - Professional Clinical AI
        # =====================================================================
        
        @self.app.post("/api/clinical/analyze")
        def analyze_patient_condition(request: ClinicalAnalysisRequest):
            """
            Comprehensive clinical analysis
            Analyzes patient condition based on KAB report and medical scans
            Uses Gemini API for professional-grade clinical reasoning
            """
            try:
                # Build KAB report object
                kab = KABReport(
                    patient_id=request.kab_report.patient_id,
                    age=request.kab_report.age,
                    gender=request.kab_report.gender,
                    chief_complaint=request.kab_report.chief_complaint,
                    presenting_symptoms=request.kab_report.presenting_symptoms,
                    medical_history=request.kab_report.medical_history,
                    medications=request.kab_report.medications,
                    allergies=request.kab_report.allergies,
                    vital_signs=request.kab_report.vital_signs,
                    lab_results=request.kab_report.lab_results,
                    imaging_findings=request.kab_report.imaging_findings,
                    diagnoses=request.kab_report.diagnoses,
                    risk_factors=request.kab_report.risk_factors
                )
                
                # Build medical scans
                scans = [
                    MedicalScan(
                        scan_id=scan.scan_id,
                        scan_type=scan.scan_type,
                        base64_data=scan.base64_data,
                        description=scan.description
                    )
                    for scan in request.medical_scans
                ]
                
                # Perform comprehensive analysis
                analysis = self.clinical_analyzer.analyze_patient_clinical_condition(kab, scans)
                
                logger.info(f"Clinical analysis completed for patient {request.kab_report.patient_id}")
                return analysis
            
            except Exception as e:
                logger.error(f"Clinical analysis error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/clinical/diagnosis-analysis")
        def diagnosis_specific_analysis(request: DiagnosisAnalysisRequest):
            """
            Deep-dive analysis for suspected diagnosis
            Evaluates how well imaging and clinical findings support the diagnosis
            """
            try:
                kab = KABReport(
                    patient_id=request.kab_report.patient_id,
                    age=request.kab_report.age,
                    gender=request.kab_report.gender,
                    chief_complaint=request.kab_report.chief_complaint,
                    presenting_symptoms=request.kab_report.presenting_symptoms,
                    medical_history=request.kab_report.medical_history,
                    medications=request.kab_report.medications,
                    allergies=request.kab_report.allergies,
                    vital_signs=request.kab_report.vital_signs,
                    lab_results=request.kab_report.lab_results,
                    imaging_findings=request.kab_report.imaging_findings,
                    diagnoses=request.kab_report.diagnoses,
                    risk_factors=request.kab_report.risk_factors
                )
                
                scans = [
                    MedicalScan(
                        scan_id=scan.scan_id,
                        scan_type=scan.scan_type,
                        base64_data=scan.base64_data,
                        description=scan.description
                    )
                    for scan in request.medical_scans
                ]
                
                analysis = self.clinical_analyzer.diagnosis_specific_analysis(
                    kab, request.suspected_diagnosis, scans
                )
                
                logger.info(f"Diagnosis analysis completed for {request.suspected_diagnosis}")
                return analysis
            
            except Exception as e:
                logger.error(f"Diagnosis analysis error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/clinical/question")
        def answer_clinical_question(request: ClinicalQuestionRequest):
            """
            Answer specific clinical questions about the patient
            Based on KAB report and imaging, provide professional medical advice
            """
            try:
                kab = KABReport(
                    patient_id=request.kab_report.patient_id,
                    age=request.kab_report.age,
                    gender=request.kab_report.gender,
                    chief_complaint=request.kab_report.chief_complaint,
                    presenting_symptoms=request.kab_report.presenting_symptoms,
                    medical_history=request.kab_report.medical_history,
                    medications=request.kab_report.medications,
                    allergies=request.kab_report.allergies,
                    vital_signs=request.kab_report.vital_signs,
                    lab_results=request.kab_report.lab_results,
                    imaging_findings=request.kab_report.imaging_findings,
                    diagnoses=request.kab_report.diagnoses,
                    risk_factors=request.kab_report.risk_factors
                )
                
                scans = [
                    MedicalScan(
                        scan_id=scan.scan_id,
                        scan_type=scan.scan_type,
                        base64_data=scan.base64_data,
                        description=scan.description
                    )
                    for scan in request.medical_scans
                ]
                
                answer = self.clinical_analyzer.generate_clinical_advice(
                    kab, request.clinical_question, scans
                )
                
                logger.info(f"Clinical question answered for patient {request.kab_report.patient_id}")
                return answer
            
            except Exception as e:
                logger.error(f"Clinical question error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/vectordb/ingest-clinical")
        def ingest_clinical_docs(documents: Dict[str, List[str]]):
            """Ingest clinical documents into vector DB"""
            try:
                count = self.vector_db.ingest_clinical_knowledge(
                    documents.get("documents", [])
                )
                return {"status": "ok", "documents_ingested": count}
            except Exception as e:
                logger.error(f"Error ingesting documents: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/status")
        def system_status():
            """Get system status"""
            return {
                "system": "MedSync AI v0.1.0",
                "llm_available": self.llm_client.is_available(),
                "vector_db_ready": True,
                "active_patients": len(self.patient_sessions),
                "active_roadmaps": len(self.roadmaps),
                "vector_db_stats": self.vector_db.get_collection_stats()
            }
    
    def run(self, host: str = API_HOST, port: int = API_PORT):
        """Start API server"""
        import uvicorn
        logger.info(f"Starting MedSync API on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)


def create_app():
    """Create FastAPI application"""
    api = MedSyncAPI()
    return api.app


if __name__ == "__main__":
    api = MedSyncAPI()
    api.run()

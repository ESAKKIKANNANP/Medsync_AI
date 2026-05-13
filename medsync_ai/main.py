"""
MedSync AI Main Orchestrator
Coordinates all components for complete surgical intelligence workflow
"""
import json
from pathlib import Path
from typing import Dict, Tuple
from medsync_ai.utils.logger import get_logger
from medsync_ai.data_pipeline import DataPipeline
from medsync_ai.vector_db import VectorDatabase
from medsync_ai.rag_engine import SurgeonDoubtRAGEngine
from medsync_ai.surgical_roadmap import SurgicalRoadmapGenerator
from medsync_ai.surgical_vision import VQLAIntegration
from medsync_ai.api.server import create_app

logger = get_logger(__name__)

# Create FastAPI application for deployment
app = create_app()

class MedSyncOrchestrator:
    """Main orchestrator for MedSync AI system"""
    
    def __init__(self):
        """Initialize all components"""
        logger.info("=" * 60)
        logger.info("Initializing MedSync AI Orchestrator")
        logger.info("=" * 60)
        
        # Data pipeline
        self.data_pipeline = DataPipeline()
        
        # Vector database
        self.vector_db = VectorDatabase()
        
        # RAG engine
        self.rag_engine = SurgeonDoubtRAGEngine(self.vector_db)
        
        # Surgical vision
        self.vqla = VQLAIntegration()
        
        # Data storage
        self.clinical_data = None
        self.surgical_data = None
        self.reasoning_data = None
        
        logger.info("MedSync AI Orchestrator ready")
    
    def ingest_and_prepare_data(self) -> bool:
        """
        Ingest all data sources and prepare vector database
        
        Returns:
            Success status
        """
        logger.info("Starting data ingestion and preparation...")
        
        try:
            # Ingest data
            self.clinical_data, self.surgical_data, self.reasoning_data = \
                self.data_pipeline.full_pipeline()
            
            # Prepare clinical knowledge for vector DB
            clinical_texts = []
            clinical_metadata = []
            
            # Extract patient histories
            if 'patients' in self.clinical_data and not self.clinical_data['patients'].empty:
                for idx, row in self.clinical_data['patients'].iterrows():
                    text = f"Patient {row.get('subject_id', idx)}, Age: {row.get('anchor_age', 'N/A')}, Gender: {row.get('gender', 'N/A')}"
                    clinical_texts.append(text)
                    clinical_metadata.append({"source": "patient_demographics"})
            
            # Extract diagnoses
            if 'diagnoses' in self.clinical_data and not self.clinical_data['diagnoses'].empty:
                for idx, row in self.clinical_data['diagnoses'].head(100).iterrows():
                    text = f"Diagnosis: {row.get('icd_code', 'N/A')} - {row.get('icd_version', 'N/A')}"
                    clinical_texts.append(text)
                    clinical_metadata.append({"source": "diagnoses"})
            
            # Ingest to vector DB
            if clinical_texts:
                self.vector_db.ingest_clinical_knowledge(clinical_texts, clinical_metadata)
                logger.info(f"Ingested {len(clinical_texts)} clinical documents")
            
            # Prepare surgical reasoning for vector DB
            surgical_texts = []
            surgical_metadata = []
            
            if self.reasoning_data:
                for idx, example in enumerate(self.reasoning_data[:100]):
                    if isinstance(example, dict):
                        # Extract text from reasoning example
                        instruction = example.get("instruction", "")
                        output = example.get("output", "")
                        text = f"{instruction} {output}"
                    else:
                        text = str(example)
                    
                    if text:
                        surgical_texts.append(text)
                        surgical_metadata.append({"source": "reasoning_sft", "index": idx})
                
                if surgical_texts:
                    self.vector_db.ingest_surgical_reasoning(surgical_texts, surgical_metadata)
                    logger.info(f"Ingested {len(surgical_texts)} surgical reasoning documents")
            
            # Prepare VQLA data
            vqla_texts = []
            vqla_image_ids = []
            
            if self.surgical_data and 'vqla_labels' in self.surgical_data:
                vqla_labels = self.surgical_data['vqla_labels']
                for split in ['train', 'val'][:1]:  # Start with train set
                    for idx, label_item in enumerate(vqla_labels.get(split, [])[:50]):
                        try:
                            label_str = json.dumps(label_item.get('data', label_item))
                            vqla_texts.append(label_str)
                            img_id = label_item.get('file', f'{split}_{idx}')
                            vqla_image_ids.append(img_id)
                        except Exception as e:
                            logger.warning(f"Error processing VQLA label {idx}: {e}")
                
                if vqla_texts:
                    self.vector_db.ingest_vqla_labels(vqla_texts, vqla_image_ids)
                    logger.info(f"Ingested {len(vqla_texts)} VQLA labels")
            
            logger.info("Data ingestion and preparation completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error during data ingestion: {e}")
            return False
    
    def demonstrate_rag_engine(self, patient_id: str = "demo_patient",
                              sample_queries: list = None) -> list:
        """
        Demonstrate RAG engine with sample queries
        
        Args:
            patient_id: Sample patient ID
            sample_queries: List of sample queries
        
        Returns:
            List of responses
        """
        if not sample_queries:
            sample_queries = [
                "What are the risks of hemorrhaging during this step?",
                "Is the patient on any anticoagulant medications?",
                "What surgical instruments should be available for the main dissection phase?"
            ]
        
        logger.info("Demonstrating RAG Engine...")
        
        # Create sample patient data
        demo_patient = {
            "patient_id": patient_id,
            "age": 58,
            "on_anticoagulation": True,
            "on_aspirin": False,
            "platelet_count": 95000,
            "has_coagulopathy": False,
            "mh_family_history": False,
            "has_renal_impairment": False,
            "has_hepatic_impairment": False
        }
        
        responses = []
        
        for query in sample_queries:
            logger.info(f"Query: {query}")
            response = self.rag_engine.answer_surgeon_query(query, demo_patient)
            responses.append(response)
            logger.info(f"Response: {response['response'][:200]}...")
        
        return responses
    
    def demonstrate_surgical_roadmap(self, procedure: str = "cholecystectomy",
                                    patient_id: str = "demo_patient") -> dict:
        """
        Demonstrate surgical roadmap generation
        
        Args:
            procedure: Surgical procedure type
            patient_id: Patient ID
        
        Returns:
            Roadmap report
        """
        logger.info(f"Demonstrating Surgical Roadmap for {procedure}...")
        
        roadmap = SurgicalRoadmapGenerator(procedure)
        
        # Simulate surgery progression
        demo_patient = {
            "age": 58,
            "hemorrhage_risk": 0.6,
            "anesthetic_reaction_risk": 0.5
        }
        
        # Evaluate initial risks
        roadmap.evaluate_risks(demo_patient)
        
        # Progress through steps
        for i in range(min(3, len(roadmap.steps))):
            logger.info(f"Step {i+1}: {roadmap.get_current_step().description}")
            roadmap.advance_to_next_step()
        
        report = roadmap.generate_roadmap_report()
        return report
    
    def demonstrate_surgical_vision(self, frame_id: str = "frame_001") -> dict:
        """
        Demonstrate surgical vision analysis
        
        Args:
            frame_id: Frame ID to analyze
        
        Returns:
            Analysis results
        """
        logger.info(f"Demonstrating Surgical Vision Analysis for frame {frame_id}...")
        
        analysis = self.vqla.analyze_frame(frame_id)
        return analysis
    
    def generate_system_report(self) -> dict:
        """Generate comprehensive system status report"""
        report = {
            "system": "MedSync AI v0.1.0",
            "data_ingestion": {
                "clinical_data_loaded": self.clinical_data is not None,
                "surgical_data_loaded": self.surgical_data is not None,
                "reasoning_data_loaded": self.reasoning_data is not None
            },
            "components": {
                "data_pipeline": "Ready",
                "vector_database": "Ready",
                "rag_engine": "Ready",
                "surgical_vision": "Ready",
                "surgical_roadmap": "Ready"
            },
            "vector_db_stats": self.vector_db.get_collection_stats()
        }
        return report


def main():
    """Main entry point for MedSync AI"""
    logger.info("=" * 60)
    logger.info("MedSync AI - Multi-Modal Surgical Intelligence Framework")
    logger.info("=" * 60)
    
    # Initialize orchestrator
    orchestrator = MedSyncOrchestrator()
    
    # Ingest data
    logger.info("\n[PHASE 1] Ingesting and preparing data...")
    data_ready = orchestrator.ingest_and_prepare_data()
    
    if not data_ready:
        logger.warning("Data ingestion had issues, but continuing with demonstration...")
    
    # Demonstrate RAG engine
    logger.info("\n[PHASE 2] Demonstrating RAG Engine...")
    responses = orchestrator.demonstrate_rag_engine()
    
    for resp in responses:
        print(f"\n--- RAG Engine Sample ---")
        print(f"Query: {resp['query']}")
        print(f"Response: {resp['response'][:150]}...")
        print(f"Hemorrhage Risk: {resp['hemorrhage_risk_score']}")
    
    # Demonstrate surgical roadmap
    logger.info("\n[PHASE 3] Demonstrating Surgical Roadmap...")
    roadmap_report = orchestrator.demonstrate_surgical_roadmap()
    print(f"\n--- Surgical Roadmap Sample ---")
    print(f"Procedure: {roadmap_report['procedure']}")
    print(f"Progress: {roadmap_report['progress']['completion_percentage']:.1f}%")
    print(f"Current Phase: {roadmap_report['current_step_info']['phase']}")
    print(f"Critical Alerts: {roadmap_report['critical_alerts']}")
    
    # System status
    logger.info("\n[PHASE 4] System Status...")
    system_report = orchestrator.generate_system_report()
    print(f"\n--- System Report ---")
    print(f"Data Pipeline: {system_report['data_ingestion']}")
    print(f"Vector DB Stats: {system_report['vector_db_stats']}")
    
    logger.info("\n" + "=" * 60)
    logger.info("MedSync AI Prototype Setup Complete!")
    logger.info("Run 'python -m medsync_ai.api.server' to start the API")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

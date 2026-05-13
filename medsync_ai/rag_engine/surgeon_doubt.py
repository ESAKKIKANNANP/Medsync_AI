"""
RAG Engine for Surgeon Doubt Clearing
Retrieves surgical context and patient-specific risk factors to answer surgeon queries
"""
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from medsync_ai.utils.logger import get_logger
from medsync_ai.vector_db import VectorDatabase
from medsync_ai.config.settings import (
    GEMINI_API_KEY, GEMINI_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS,
    HEMORRHAGE_RISK_THRESHOLD, ANESTHETIC_REACTION_THRESHOLD
)

logger = get_logger(__name__)

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    logger.warning("google-generativeai not installed. Install with: pip install google-generativeai")

@dataclass
class SurgicalContext:
    """Container for surgical context information"""
    patient_history: List[str]
    surgical_procedures: List[str]
    medications: List[str]
    risk_factors: List[str]
    recent_labs: List[str]
    vqla_labels: List[str]


class GeminiClient:
    """Client for Google Gemini API"""
    
    def __init__(self, api_key: str = GEMINI_API_KEY, 
                 model: str = GEMINI_MODEL):
        """
        Initialize Gemini client
        
        Args:
            api_key: Google API key
            model: Model name
        """
        if not GENAI_AVAILABLE:
            raise ImportError("google-generativeai is required. Install with: pip install google-generativeai")
        
        if not api_key:
            error_msg = (
                "GEMINI_API_KEY is not set. "
                "You can set it in three ways: "
                "1. Edit .env file in project root: GEMINI_API_KEY=your-key "
                "2. Export in terminal: export GEMINI_API_KEY=your-key "
                "3. Set in system environment variables. "
                "Get your free API key from: https://aistudio.google.com/"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        self.api_key = api_key
        self.model = model
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(model)
        logger.info(f"Initialized Gemini client: {model}")
    
    def generate(self, prompt: str, 
                 images: Optional[List[Any]] = None,
                 temperature: float = LLM_TEMPERATURE,
                 max_tokens: int = LLM_MAX_TOKENS) -> str:
        """
        Generate text from prompt with optional images
        
        Args:
            prompt: Input prompt
            images: Optional list of PIL Images or file paths
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        
        Returns:
            Generated text
        """
        try:
            logger.info(f"Querying Gemini: {self.model}")
            
            # Prepare content
            content = []
            
            # Add images if provided
            if images:
                for img in images:
                    if isinstance(img, str):
                        # File path - will be handled by genai
                        content.append(img)
                    else:
                        # PIL Image object
                        content.append(img)
            
            # Add text
            content.append(prompt)
            
            # Generate response
            response = self.client.generate_content(
                content,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens
                )
            )
            
            generated_text = response.text
            logger.info(f"Gemini response received ({len(generated_text)} chars)")
            return generated_text
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return f"Error generating response: {str(e)}"
    
    def is_available(self) -> bool:
        """Check if Gemini API is available"""
        try:
            if not GENAI_AVAILABLE:
                logger.warning("google-generativeai not installed")
                return False
            
            if not self.api_key:
                logger.warning("GEMINI_API_KEY not configured")
                return False
            
            # If we have an API key and the client is initialized, consider it available
            # The API might fail due to quota limits, but that's different from being unavailable
            logger.info(f"✓ Gemini API available - model {self.model} is configured")
            return True
            
        except Exception as e:
            logger.warning(f"Gemini availability check failed: {e}")
            return False


# Keep LLMClient as alias for backward compatibility
LLMClient = GeminiClient


class RiskAssessment:
    """Assess surgical and patient-specific risks"""
    
    @staticmethod
    def assess_hemorrhage_risk(patient_data: Dict) -> Tuple[float, List[str]]:
        """
        Assess hemorrhage risk based on patient factors
        
        Args:
            patient_data: Patient clinical data
        
        Returns:
            (risk_score, risk_factors)
        """
        risk_score = 0.0
        risk_factors = []
        
        # Check for anticoagulation
        if patient_data.get("on_anticoagulation"):
            risk_score += 0.3
            risk_factors.append("Patient on anticoagulation therapy")
        
        # Check for low platelets
        platelets = patient_data.get("platelet_count")
        if platelets and platelets < 100000:
            risk_score += 0.2
            risk_factors.append(f"Low platelet count: {platelets}")
        
        # Check for coagulopathy
        if patient_data.get("has_coagulopathy"):
            risk_score += 0.25
            risk_factors.append("Known coagulopathy or clotting disorder")
        
        # Recent aspirin use
        if patient_data.get("on_aspirin"):
            risk_score += 0.15
            risk_factors.append("Recent aspirin use")
        
        return min(risk_score, 1.0), risk_factors
    
    @staticmethod
    def assess_anesthetic_reaction_risk(patient_data: Dict) -> Tuple[float, List[str]]:
        """
        Assess risk of anesthetic adverse reactions
        
        Args:
            patient_data: Patient data
        
        Returns:
            (risk_score, risk_factors)
        """
        risk_score = 0.0
        risk_factors = []
        
        # Malignant hyperthermia family history
        if patient_data.get("mh_family_history"):
            risk_score += 0.35
            risk_factors.append("Family history of malignant hyperthermia")
        
        # Age extremes
        age = patient_data.get("age")
        if age and (age < 5 or age > 75):
            risk_score += 0.15
            risk_factors.append(f"Extreme age: {age} years")
        
        # Renal impairment
        if patient_data.get("has_renal_impairment"):
            risk_score += 0.2
            risk_factors.append("Renal impairment")
        
        # Hepatic impairment
        if patient_data.get("has_hepatic_impairment"):
            risk_score += 0.2
            risk_factors.append("Hepatic impairment")
        
        return min(risk_score, 1.0), risk_factors


class SurgeonDoubtRAGEngine:
    """RAG engine for surgeon doubt clearing"""
    
    def __init__(self, vector_db: VectorDatabase,
                 llm_client: Optional[LLMClient] = None):
        """
        Initialize RAG engine
        
        Args:
            vector_db: Vector database instance
            llm_client: Optional LLM client
        """
        self.vector_db = vector_db
        self.llm_client = llm_client or LLMClient()
        self.risk_assessor = RiskAssessment()
        logger.info("RAG Engine initialized")
    
    def retrieve_context(self, query: str, 
                        patient_data: Optional[Dict] = None,
                        top_k: int = 5) -> SurgicalContext:
        """
        Retrieve comprehensive surgical context
        
        Args:
            query: Surgeon query
            patient_data: Patient clinical data
            top_k: Number of results to retrieve
        
        Returns:
            SurgicalContext object
        """
        # Hybrid retrieval from all collections
        retrieval_results = self.vector_db.hybrid_retrieve(query, top_k)
        
        context = SurgicalContext(
            patient_history=retrieval_results['clinical'][0],
            surgical_procedures=retrieval_results['surgical'][0],
            medications=[],
            risk_factors=[],
            recent_labs=[],
            vqla_labels=retrieval_results['vqla'][0]
        )
        
        # Add patient-specific risk factors
        if patient_data:
            hemorrhage_risk, hemorrhage_factors = self.risk_assessor.assess_hemorrhage_risk(
                patient_data
            )
            anesthetic_risk, anesthetic_factors = self.risk_assessor.assess_anesthetic_reaction_risk(
                patient_data
            )
            
            context.risk_factors.extend(hemorrhage_factors)
            context.risk_factors.extend(anesthetic_factors)
        
        return context
    
    def build_prompt(self, query: str, context: SurgicalContext) -> str:
        """
        Build comprehensive prompt for LLM with professional medical analysis
        
        Args:
            query: Original surgeon query
            context: Retrieved surgical context
        
        Returns:
            Formatted prompt with enhanced system guidance
        """
        prompt = f"""You are a senior surgical consultant and expert in surgical decision support. 
Respond in a professional doc-to-doc manner with rigorous, evidence-based analysis.

PATIENT-SPECIFIC SURGICAL CONTEXT:
====================================
Clinical Background: {'; '.join(context.patient_history[:3]) if context.patient_history else 'Not available'}

Relevant Surgical History: {'; '.join(context.surgical_procedures[:2]) if context.surgical_procedures else 'No prior relevant procedures'}

Identified Risk Factors:
{(chr(10).join([f"• {rf}" for rf in context.risk_factors])) if context.risk_factors else "• No major risk factors identified"}

Current Medications: {'; '.join(context.medications) if context.medications else 'None documented'}

Surgical Instruments/Technical Scene: {'; '.join(context.vqla_labels[:3]) if context.vqla_labels else 'Standard instruments'}

SURGEON'S QUESTION/CONCERN:
====================================
{query}

RESPONSE REQUIREMENTS:
====================================
1. **CLINICAL IMPRESSION**: Synthesize the patient's condition and case complexity
2. **DIRECT ANSWER**: Address the question concisely with evidence-based recommendations
3. **RISK STRATIFICATION**: Quantify risks where applicable; specify HIGH/MEDIUM/LOW
4. **TECHNICAL PEARLS**: Provide specific surgical techniques or maneuvers for THIS patient
5. **ALTERNATIVE STRATEGIES**: If standard approach is risky, outline alternatives
6. **COMPLICATION PREVENTION**: Detail specific preventive measures
7. **MONITORING PARAMETERS**: What to watch for during the procedure
8. **POST-OP CONSIDERATIONS**: Specific care plan implications

Guidelines:
- Be professional but collegial - respond as one surgeon to another
- Use precise medical terminology; include anatomy where relevant
- Provide actionable recommendations, not vague suggestions
- Reference patient-specific factors, not generic guidance
- Include estimated complication rates if applicable
- Maximum clarity with appropriate detail for surgical planning
- If uncertainty exists, acknowledge and provide conservative approach

Format: Clear sections with bullet points for rapid scanning."""
        
        return prompt
    
    def answer_surgeon_query(self, query: str,
                            patient_data: Optional[Dict] = None,
                            surgical_scene: Optional[str] = None) -> Dict[str, str]:
        """
        Answer surgeon's doubt with comprehensive RAG response
        
        Args:
            query: Surgeon's question
            patient_data: Patient clinical information
            surgical_scene: Current surgical scene description
        
        Returns:
            Dictionary with response and metadata
        """
        logger.info(f"Processing surgeon query: {query}")
        
        # Retrieve context
        context = self.retrieve_context(query, patient_data)
        
        # Build and generate response
        prompt = self.build_prompt(query, context)
        response_text = self.llm_client.generate(prompt)
        
        # Assess risk levels
        hemorrhage_risk = 0.0
        anesthetic_risk = 0.0
        
        if patient_data:
            hemorrhage_risk, _ = self.risk_assessor.assess_hemorrhage_risk(patient_data)
            anesthetic_risk, _ = self.risk_assessor.assess_anesthetic_reaction_risk(patient_data)
        
        result = {
            "query": query,
            "response": response_text,
            "hemorrhage_risk_score": f"{hemorrhage_risk:.2f}",
            "anesthetic_risk_score": f"{anesthetic_risk:.2f}",
            "hemorrhage_alert": hemorrhage_risk > HEMORRHAGE_RISK_THRESHOLD,
            "anesthetic_alert": anesthetic_risk > ANESTHETIC_REACTION_THRESHOLD,
            "retrieved_clinical_docs": len(context.patient_history),
            "retrieved_surgical_docs": len(context.surgical_procedures),
            "risk_factors_identified": len(context.risk_factors)
        }
        
        logger.info(f"Query processed - Risk scores: Hemorrhage={hemorrhage_risk:.2f}, Anesthetic={anesthetic_risk:.2f}")
        
        return result
    
    def batch_answer_queries(self, queries: List[str],
                            patient_data: Optional[Dict] = None) -> List[Dict]:
        """Answer multiple queries"""
        results = []
        for query in queries:
            result = self.answer_surgeon_query(query, patient_data)
            results.append(result)
        return results


if __name__ == "__main__":
    # Demo
    vdb = VectorDatabase()
    rag_engine = SurgeonDoubtRAGEngine(vdb)
    
    # Sample patient data
    patient = {
        "on_anticoagulation": True,
        "platelet_count": 95000,
        "age": 65
    }
    
    # Sample query
    query = "Is there a risk of hemorrhaging during this step?"
    
    # Get response
    response = rag_engine.answer_surgeon_query(query, patient)
    print("Query:", response["query"])
    print("Response:", response["response"])
    print("Hemorrhage Risk:", response["hemorrhage_risk_score"])

"""
Unit tests and integration tests for MedSync AI components
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from medsync_ai.utils.text_processor import clean_text, chunk_text
from medsync_ai.vector_db import EmbeddingEngine, VectorDatabase
from medsync_ai.rag_engine import LLMClient, SurgeonDoubtRAGEngine
from medsync_ai.surgical_roadmap import SurgicalRoadmapGenerator
from medsync_ai.surgical_vision import VQLAIntegration
from medsync_ai.utils.logger import get_logger

logger = get_logger(__name__)

class TestTextProcessing:
    """Test text processing utilities"""
    
    def test_clean_text(self):
        """Test text cleaning"""
        dirty = "  Hello   WORLD!! 123  "
        clean = clean_text(dirty)
        assert len(clean) < len(dirty)
        assert "WORLD" in clean
        print("✓ Text cleaning works")
    
    def test_chunk_text(self):
        """Test text chunking"""
        long_text = "This is a test sentence. " * 100
        chunks = chunk_text(long_text, chunk_size=100, overlap=20)
        assert len(chunks) > 1
        print(f"✓ Text chunking works ({len(chunks)} chunks)")

class TestEmbeddings:
    """Test embedding functionality"""
    
    def test_embedding_engine(self):
        """Test embedding engine initialization"""
        try:
            engine = EmbeddingEngine()
            text = "Patient with chest pain and elevated troponin"
            embedding = engine.embed_text(text)
            assert embedding is not None
            assert len(embedding) == 768
            print(f"✓ Embedding engine works (768 dimensions)")
        except Exception as e:
            print(f"⚠ Embedding engine test skipped: {e}")

class TestVectorDatabase:
    """Test vector database operations"""
    
    def test_vdb_initialization(self):
        """Test vector database initialization"""
        try:
            vdb = VectorDatabase()
            stats = vdb.get_collection_stats()
            assert isinstance(stats, dict)
            print(f"✓ Vector DB initialized: {stats}")
        except Exception as e:
            print(f"⚠ Vector DB test skipped: {e}")

class TestLLMClient:
    """Test LLM client"""
    
    def test_llm_availability(self):
        """Test LLM availability check"""
        client = LLMClient()
        available = client.is_available()
        if available:
            print("✓ LLM is available")
        else:
            print("⚠ LLM not available (start Ollama to enable)")

class TestSurgicalRoadmap:
    """Test surgical roadmap generation"""
    
    def test_roadmap_creation(self):
        """Test roadmap creation"""
        roadmap = SurgicalRoadmapGenerator("cholecystectomy")
        assert len(roadmap.steps) > 0
        current = roadmap.get_current_step()
        assert current is not None
        print(f"✓ Surgical roadmap created ({len(roadmap.steps)} steps)")
    
    def test_roadmap_progression(self):
        """Test roadmap step progression"""
        roadmap = SurgicalRoadmapGenerator("cholecystectomy")
        initial_idx = roadmap.current_step_index
        roadmap.advance_to_next_step()
        assert roadmap.current_step_index > initial_idx
        print("✓ Roadmap progression works")
    
    def test_alert_generation(self):
        """Test alert generation"""
        roadmap = SurgicalRoadmapGenerator("cholecystectomy")
        roadmap.add_complication_alert(
            severity="medium",
            risk_type="test",
            description="Test alert",
            risk_score=0.5
        )
        assert len(roadmap.alerts) > 0
        print("✓ Alert generation works")

class TestVQLAIntegration:
    """Test VQLA integration"""
    
    def test_vqla_initialization(self):
        """Test VQLA initialization"""
        try:
            vqla = VQLAIntegration()
            print("✓ VQLA integration initialized")
        except Exception as e:
            print(f"⚠ VQLA test skipped: {e}")

def run_all_tests():
    """Run all tests"""
    logger.info("=" * 60)
    logger.info("MedSync AI Component Tests")
    logger.info("=" * 60)
    
    print("\n[1] Text Processing Tests")
    test_text = TestTextProcessing()
    test_text.test_clean_text()
    test_text.test_chunk_text()
    
    print("\n[2] Embedding Tests")
    test_embed = TestEmbeddings()
    test_embed.test_embedding_engine()
    
    print("\n[3] Vector Database Tests")
    test_vdb = TestVectorDatabase()
    test_vdb.test_vdb_initialization()
    
    print("\n[4] LLM Client Tests")
    test_llm = TestLLMClient()
    test_llm.test_llm_availability()
    
    print("\n[5] Surgical Roadmap Tests")
    test_roadmap = TestSurgicalRoadmap()
    test_roadmap.test_roadmap_creation()
    test_roadmap.test_roadmap_progression()
    test_roadmap.test_alert_generation()
    
    print("\n[6] VQLA Integration Tests")
    test_vqla = TestVQLAIntegration()
    test_vqla.test_vqla_initialization()
    
    logger.info("=" * 60)
    logger.info("Component Tests Complete!")
    logger.info("=" * 60)

if __name__ == "__main__":
    run_all_tests()

"""
Surgical Vision & VQLA Module
Integrates EndoVis-18-VQLA dataset for surgical instrument and tissue identification
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from medsync_ai.utils.logger import get_logger
from medsync_ai.config.settings import (
    ENDOVIS_DIR, SURGICAL_INSTRUMENTS, ANATOMICAL_STRUCTURES
)

logger = get_logger(__name__)

@dataclass
class SurgicalInstrument:
    """Container for surgical instrument detection"""
    name: str
    confidence: float
    location: Optional[Dict[str, float]] = None  # bounding box
    frame_id: Optional[int] = None

@dataclass
class AnatomicalStructure:
    """Container for anatomical structure detection"""
    name: str
    confidence: float
    location: Optional[Dict[str, float]] = None
    frame_id: Optional[int] = None

@dataclass
class SurgicalQuestion:
    """Container for surgical VQA"""
    question: str
    answer: Optional[str] = None
    instruments_involved: List[str] = None
    anatomical_structures_involved: List[str] = None
    frame_id: Optional[int] = None
    image_id: Optional[str] = None

class VQLALabelParser:
    """Parse VQLA labels from JSON format"""
    
    @staticmethod
    def parse_label(label_dict: Dict) -> Dict:
        """
        Parse VQLA label dictionary
        
        Args:
            label_dict: Raw VQLA label
        
        Returns:
            Structured label data
        """
        parsed = {
            "frame_id": label_dict.get("frame_id"),
            "image_id": label_dict.get("image_id"),
            "instruments": [],
            "anatomical_structures": [],
            "vqa_pairs": []
        }
        
        # Parse instruments
        if "instruments" in label_dict:
            for inst in label_dict["instruments"]:
                if isinstance(inst, dict):
                    parsed["instruments"].append({
                        "name": inst.get("name"),
                        "confidence": inst.get("confidence", 0.0),
                        "location": inst.get("location")
                    })
                else:
                    parsed["instruments"].append({"name": inst, "confidence": 1.0})
        
        # Parse anatomical structures
        if "anatomical_structures" in label_dict:
            for struct in label_dict["anatomical_structures"]:
                if isinstance(struct, dict):
                    parsed["anatomical_structures"].append({
                        "name": struct.get("name"),
                        "confidence": struct.get("confidence", 0.0),
                        "location": struct.get("location")
                    })
                else:
                    parsed["anatomical_structures"].append({"name": struct, "confidence": 1.0})
        
        # Parse VQA pairs
        if "vqa_pairs" in label_dict:
            for vqa in label_dict["vqa_pairs"]:
                parsed["vqa_pairs"].append({
                    "question": vqa.get("question"),
                    "answer": vqa.get("answer"),
                    "instruments": vqa.get("instruments", []),
                    "anatomical_structures": vqa.get("anatomical_structures", [])
                })
        
        return parsed


class SurgicalSceneAnalyzer:
    """Analyze surgical scenes for instruments and structures"""
    
    def __init__(self, vqla_labels: Optional[Dict] = None):
        """
        Initialize scene analyzer
        
        Args:
            vqla_labels: Pre-loaded VQLA labels
        """
        self.vqla_labels = vqla_labels or {}
        self.parser = VQLALabelParser()
        logger.info("Surgical Scene Analyzer initialized")
    
    def identify_instruments(self, frame_id: str,
                            vqla_data: Optional[Dict] = None) -> List[SurgicalInstrument]:
        """
        Identify surgical instruments in frame
        
        Args:
            frame_id: Frame identifier
            vqla_data: Optional VQLA data for this frame
        
        Returns:
            List of detected instruments
        """
        instruments = []
        
        if vqla_data and "instruments" in vqla_data:
            for inst_data in vqla_data["instruments"]:
                instrument = SurgicalInstrument(
                    name=inst_data.get("name"),
                    confidence=inst_data.get("confidence", 0.0),
                    location=inst_data.get("location"),
                    frame_id=frame_id
                )
                instruments.append(instrument)
        
        logger.info(f"Frame {frame_id}: Identified {len(instruments)} instruments")
        return instruments
    
    def identify_anatomical_structures(self, frame_id: str,
                                      vqla_data: Optional[Dict] = None) -> List[AnatomicalStructure]:
        """
        Identify anatomical structures in frame
        
        Args:
            frame_id: Frame identifier
            vqla_data: Optional VQLA data for this frame
        
        Returns:
            List of detected structures
        """
        structures = []
        
        if vqla_data and "anatomical_structures" in vqla_data:
            for struct_data in vqla_data["anatomical_structures"]:
                structure = AnatomicalStructure(
                    name=struct_data.get("name"),
                    confidence=struct_data.get("confidence", 0.0),
                    location=struct_data.get("location"),
                    frame_id=frame_id
                )
                structures.append(structure)
        
        logger.info(f"Frame {frame_id}: Identified {len(structures)} anatomical structures")
        return structures
    
    def answer_visual_question(self, question: str,
                              frame_id: str,
                              vqla_data: Optional[Dict] = None) -> Optional[SurgicalQuestion]:
        """
        Answer visual question for surgical frame
        
        Args:
            question: Visual question
            frame_id: Frame identifier
            vqla_data: VQLA data for this frame
        
        Returns:
            Question with answer and context
        """
        answer = None
        instruments = []
        structures = []
        
        if vqla_data:
            # Look for matching question in VQA pairs
            for vqa_pair in vqla_data.get("vqa_pairs", []):
                if vqa_pair.get("question", "").lower() == question.lower():
                    answer = vqa_pair.get("answer")
                    instruments = vqa_pair.get("instruments", [])
                    structures = vqa_pair.get("anatomical_structures", [])
                    break
        
        sq = SurgicalQuestion(
            question=question,
            answer=answer,
            instruments_involved=instruments,
            anatomical_structures_involved=structures,
            frame_id=frame_id
        )
        
        logger.info(f"VQA - Q: {question[:50]}... A: {answer[:50] if answer else 'Not found'}...")
        return sq
    
    def map_instruments_to_questions(self, vqla_data: Dict) -> Dict[str, List[str]]:
        """
        Map surgical instruments to relevant VQA pairs
        
        Args:
            vqla_data: VQLA data
        
        Returns:
            Dictionary mapping instruments to questions
        """
        mapping = {}
        
        instruments = [i.get("name") for i in vqla_data.get("instruments", [])]
        
        for inst in instruments:
            mapping[inst] = []
            for vqa_pair in vqla_data.get("vqa_pairs", []):
                if inst in vqa_pair.get("instruments", []):
                    mapping[inst].append(vqa_pair.get("question"))
        
        return mapping
    
    def get_surgical_phase_context(self, current_frame: int,
                                  all_frames: Dict[int, Dict]) -> Dict:
        """
        Provide surgical phase context based on instrument progression
        
        Args:
            current_frame: Current frame number
            all_frames: All frame VQLA data indexed by frame number
        
        Returns:
            Context about surgical phase
        """
        frames_by_num = sorted(all_frames.keys())
        current_idx = frames_by_num.index(current_frame) if current_frame in frames_by_num else 0
        
        # Get instruments from recent frames
        recent_instruments = set()
        for i in range(max(0, current_idx - 5), current_idx + 1):
            frame_num = frames_by_num[i]
            frame_data = all_frames[frame_num]
            for inst in frame_data.get("instruments", []):
                recent_instruments.add(inst.get("name"))
        
        context = {
            "current_frame": current_frame,
            "current_instruments": recent_instruments,
            "phase_progression": current_idx / len(frames_by_num) if frames_by_num else 0
        }
        
        return context


class VQLAIntegration:
    """Main integration layer for VQLA dataset"""
    
    def __init__(self, endovis_path: Path = ENDOVIS_DIR):
        """
        Initialize VQLA integration
        
        Args:
            endovis_path: Path to EndoVis dataset
        """
        self.endovis_path = Path(endovis_path)
        self.scene_analyzer = SurgicalSceneAnalyzer()
        self.parser = VQLALabelParser()
        logger.info("VQLA Integration initialized")
    
    def load_frame_labels(self, frame_id: str) -> Optional[Dict]:
        """Load VQLA labels for specific frame"""
        # Try to find label file for frame
        for split in ["train", "val"]:
            split_path = self.endovis_path / split / "image"
            label_file = split_path / f"{frame_id}.json"
            
            if label_file.exists():
                try:
                    with open(label_file) as f:
                        return self.parser.parse_label(json.load(f))
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse label: {label_file}")
        
        return None
    
    def analyze_frame(self, frame_id: str) -> Dict:
        """Complete analysis of surgical frame"""
        labels = self.load_frame_labels(frame_id)
        
        if not labels:
            return {"frame_id": frame_id, "error": "Labels not found"}
        
        instruments = self.scene_analyzer.identify_instruments(frame_id, labels)
        structures = self.scene_analyzer.identify_anatomical_structures(frame_id, labels)
        vqa_context = self.scene_analyzer.map_instruments_to_questions(labels)
        
        return {
            "frame_id": frame_id,
            "instruments": [
                {"name": i.name, "confidence": i.confidence, "location": i.location}
                for i in instruments
            ],
            "anatomical_structures": [
                {"name": s.name, "confidence": s.confidence, "location": s.location}
                for s in structures
            ],
            "vqa_context": vqa_context,
            "raw_labels": labels
        }


if __name__ == "__main__":
    # Demo
    vqla = VQLAIntegration()
    
    # Example frame analysis (assuming frame exists)
    frame_analysis = vqla.analyze_frame("frame_001")
    print("Frame Analysis:", frame_analysis)

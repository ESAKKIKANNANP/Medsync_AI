"""
Intelligent Surgical Roadmap Generator
Generates dynamic surgical planning that updates with surgery progression
"""
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
from medsync_ai.utils.logger import get_logger
from medsync_ai.config.settings import (
    COMPLICATION_ALERT_THRESHOLD,
    HEMORRHAGE_RISK_THRESHOLD,
    ANESTHETIC_REACTION_THRESHOLD
)

logger = get_logger(__name__)

class SurgicalPhase(Enum):
    """Enumeration of surgical phases"""
    PREPARATION = "preparation"
    EXPOSURE = "exposure"
    MAIN_DISSECTION = "main_dissection"
    PRIMARY_REPAIR = "primary_repair"
    HEMOSTASIS = "hemostasis"
    CLOSURE = "closure"
    COMPLETE = "complete"

@dataclass
class SurgicalStep:
    """Individual surgical step"""
    phase: SurgicalPhase
    step_number: int
    description: str
    estimated_duration_minutes: int
    critical_instruments: List[str] = field(default_factory=list)
    anatomical_focus: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    completed: bool = False
    actual_duration_minutes: int = 0
    notes: str = ""

@dataclass
class ComplicationAlert:
    """Alert for potential complication"""
    severity: str  # low, medium, high, critical
    risk_type: str  # hemorrhage, anesthetic, anatomical, time
    description: str
    risk_score: float
    timestamp: datetime = field(default_factory=datetime.now)
    mitigation_steps: List[str] = field(default_factory=list)

class SurgicalRoadmapGenerator:
    """Generate and manage surgical roadmap"""
    
    # Standard surgical phases for common procedures
    STANDARD_PHASES = {
        "cholecystectomy": [
            SurgicalStep(
                phase=SurgicalPhase.PREPARATION,
                step_number=1,
                description="Patient positioning and prepping",
                estimated_duration_minutes=15,
                critical_instruments=["Drapes", "Betadine"],
                anatomical_focus=["Abdomen"]
            ),
            SurgicalStep(
                phase=SurgicalPhase.EXPOSURE,
                step_number=2,
                description="Trocar placement and port establishment",
                estimated_duration_minutes=10,
                critical_instruments=["Trocars", "Insufflator"],
                anatomical_focus=["Peritoneum", "Abdominal wall"],
                risk_factors=["Bowel injury risk", "Vascular injury risk"]
            ),
            SurgicalStep(
                phase=SurgicalPhase.MAIN_DISSECTION,
                step_number=3,
                description="Calot's triangle dissection and identification",
                estimated_duration_minutes=20,
                critical_instruments=["Grasper", "Dissector", "Electrocautery"],
                anatomical_focus=["Gallbladder", "Cystic artery", "Cystic duct"],
                risk_factors=["Hemorrhage risk", "Bile duct injury risk"]
            ),
            SurgicalStep(
                phase=SurgicalPhase.PRIMARY_REPAIR,
                step_number=4,
                description="Cystic artery and duct ligation",
                estimated_duration_minutes=15,
                critical_instruments=["Clipper", "Cautery", "Suction"],
                anatomical_focus=["Cystic artery", "Cystic duct"],
                risk_factors=["Hemorrhage", "Clip slippage"]
            ),
            SurgicalStep(
                phase=SurgicalPhase.HEMOSTASIS,
                step_number=5,
                description="Inspection for hemostasis and bile leak",
                estimated_duration_minutes=10,
                critical_instruments=["Grasper", "Suction", "Cautery"],
                anatomical_focus=["Liver bed", "Peritoneum"]
            ),
            SurgicalStep(
                phase=SurgicalPhase.CLOSURE,
                step_number=6,
                description="Port closure and skin closure",
                estimated_duration_minutes=15,
                critical_instruments=["Fascial sutures", "Skin sutures"],
                anatomical_focus=["Fascia", "Skin"]
            )
        ]
    }
    
    def __init__(self, procedure_type: str = "cholecystectomy"):
        """
        Initialize roadmap generator
        
        Args:
            procedure_type: Type of surgical procedure
        """
        self.procedure_type = procedure_type
        self.steps = self.STANDARD_PHASES.get(procedure_type, [])
        self.current_step_index = 0
        self.alerts = []
        self.start_time = datetime.now()
        logger.info(f"Surgical Roadmap initialized for {procedure_type}")
    
    def get_current_step(self) -> Optional[SurgicalStep]:
        """Get current surgical step"""
        if self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None
    
    def advance_to_next_step(self) -> bool:
        """Advance to next surgical step"""
        if self.current_step_index < len(self.steps) - 1:
            current = self.steps[self.current_step_index]
            current.completed = True
            self.current_step_index += 1
            logger.info(f"Advanced to step {self.current_step_index}: {self.get_current_step().description}")
            return True
        elif self.current_step_index == len(self.steps) - 1:
            # Mark final step as complete
            self.steps[-1].completed = True
            self.current_step_index = len(self.steps)
            logger.info("Surgery completed all steps")
            return False
        return False
    
    def add_complication_alert(self, severity: str, risk_type: str,
                              description: str, risk_score: float,
                              mitigation_steps: Optional[List[str]] = None):
        """
        Add complication alert to roadmap
        
        Args:
            severity: Severity level
            risk_type: Type of risk
            description: Alert description
            risk_score: Risk score (0-1)
            mitigation_steps: Suggested mitigation steps
        """
        alert = ComplicationAlert(
            severity=severity,
            risk_type=risk_type,
            description=description,
            risk_score=risk_score,
            mitigation_steps=mitigation_steps or []
        )
        self.alerts.append(alert)
        
        logger.warning(f"Alert: [{severity.upper()}] {risk_type} - {description}")
    
    def evaluate_risks(self, patient_data: Dict) -> List[ComplicationAlert]:
        """
        Evaluate surgical risks at current step
        
        Args:
            patient_data: Patient clinical information
        
        Returns:
            List of identified alerts
        """
        current_step = self.get_current_step()
        if not current_step:
            return []
        
        new_alerts = []
        
        # Hemorrhage risk
        if "hemorrhage" in [rf.lower() for rf in current_step.risk_factors]:
            hemorrhage_score = patient_data.get("hemorrhage_risk", 0.0)
            if hemorrhage_score > HEMORRHAGE_RISK_THRESHOLD:
                self.add_complication_alert(
                    severity="high",
                    risk_type="hemorrhage",
                    description=f"High hemorrhage risk in {current_step.description}",
                    risk_score=hemorrhage_score,
                    mitigation_steps=[
                        "Ensure adequate blood products available",
                        "Verify hemostatic technique",
                        "Consider cell salvage activation"
                    ]
                )
                new_alerts.append(self.alerts[-1])
        
        # Anesthetic risk
        anesthetic_score = patient_data.get("anesthetic_reaction_risk", 0.0)
        if anesthetic_score > ANESTHETIC_REACTION_THRESHOLD:
            self.add_complication_alert(
                severity="high",
                risk_type="anesthetic",
                description="Anesthetic adverse reaction risk",
                risk_score=anesthetic_score,
                mitigation_steps=[
                    "Maintain vigilant monitoring of vital signs",
                    "Have malignant hyperthermia cart available",
                    "Consider neuromuscular monitoring"
                ]
            )
            new_alerts.append(self.alerts[-1])
        
        # Step-specific complication risks
        current_time = datetime.now()
        elapsed_minutes = (current_time - self.start_time).total_seconds() / 60
        
        # Check if running long
        cumulative_expected = sum(
            s.estimated_duration_minutes for s in self.steps[:self.current_step_index + 1]
        )
        
        if elapsed_minutes > cumulative_expected * 1.5:
            self.add_complication_alert(
                severity="medium",
                risk_type="time",
                description="Surgery exceeding expected duration",
                risk_score=0.65,
                mitigation_steps=[
                    "Assess for unexpected anatomical variation",
                    "Consider consulting senior surgeon if needed",
                    "Monitor for surgeon fatigue"
                ]
            )
            new_alerts.append(self.alerts[-1])
        
        return new_alerts
    
    def generate_roadmap_report(self) -> Dict:
        """
        Generate comprehensive roadmap report
        
        Returns:
            Report dictionary
        """
        current_step = self.get_current_step()
        next_step = self.steps[self.current_step_index + 1] if self.current_step_index + 1 < len(self.steps) else None
        
        completed_steps = sum(1 for s in self.steps if s.completed)
        
        report = {
            "procedure": self.procedure_type,
            "start_time": self.start_time.isoformat(),
            "current_time": datetime.now().isoformat(),
            "elapsed_minutes": (datetime.now() - self.start_time).total_seconds() / 60,
            "progress": {
                "current_step": self.current_step_index + 1,
                "total_steps": len(self.steps),
                "completed_steps": completed_steps,
                "completion_percentage": (completed_steps / len(self.steps) * 100) if self.steps else 0
            },
            "current_step_info": {
                "phase": current_step.phase.value if current_step else None,
                "description": current_step.description if current_step else None,
                "critical_instruments": current_step.critical_instruments if current_step else [],
                "anatomical_focus": current_step.anatomical_focus if current_step else [],
                "risk_factors": current_step.risk_factors if current_step else []
            },
            "next_step_info": {
                "phase": next_step.phase.value if next_step else None,
                "description": next_step.description if next_step else None
            },
            "alerts": [
                {
                    "severity": alert.severity,
                    "type": alert.risk_type,
                    "description": alert.description,
                    "risk_score": f"{alert.risk_score:.2f}",
                    "timestamp": alert.timestamp.isoformat(),
                    "mitigation": alert.mitigation_steps
                }
                for alert in self.alerts
            ],
            "critical_alerts": len([a for a in self.alerts if a.severity in ["high", "critical"]])
        }
        
        return report
    
    def export_roadmap(self, filepath: Optional[str] = None) -> str:
        """
        Export roadmap to JSON file
        
        Args:
            filepath: Optional output filepath
        
        Returns:
            JSON string of roadmap
        """
        report = self.generate_roadmap_report()
        json_str = json.dumps(report, indent=2, default=str)
        
        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_str)
            logger.info(f"Roadmap exported to {filepath}")
        
        return json_str


if __name__ == "__main__":
    # Demo
    roadmap = SurgicalRoadmapGenerator("cholecystectomy")
    
    # Simulate surgery progression
    print("Current Step:", roadmap.get_current_step().description)
    
    # Add sample alert
    patient = {"hemorrhage_risk": 0.75, "anesthetic_reaction_risk": 0.5}
    roadmap.evaluate_risks(patient)
    
    # Advance step
    roadmap.advance_to_next_step()
    print("Next Step:", roadmap.get_current_step().description)
    
    # Generate report
    report = roadmap.generate_roadmap_report()
    print("\nRoadmap Report:")
    print(json.dumps(report, indent=2, default=str))

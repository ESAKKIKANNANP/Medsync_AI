"""
Data Ingestion Pipeline
Handles loading and cleaning MIMIC-IV and EndoVis-18-VQLA data
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from medsync_ai.utils.logger import get_logger
from medsync_ai.config.settings import (
    MIMIC_DIR, ENDOVIS_DIR, MEDICAL_O1_PATH, 
    DATA_DIR
)

logger = get_logger(__name__)

class MIMICDataLoader:
    """Load and process MIMIC-IV clinical data"""
    
    def __init__(self, mimic_path: Path = MIMIC_DIR):
        self.mimic_path = mimic_path
        self.hosp_path = mimic_path / "hosp"
        self.icu_path = mimic_path / "icu"
    
    def load_patients(self) -> pd.DataFrame:
        """Load patient demographics"""
        patients_file = self.hosp_path / "patients.csv"
        if not patients_file.exists():
            logger.warning(f"Patients file not found: {patients_file}")
            return pd.DataFrame()
        
        df = pd.read_csv(patients_file)
        logger.info(f"Loaded {len(df)} patient records")
        return df
    
    def load_admissions(self) -> pd.DataFrame:
        """Load patient admissions with clinical context"""
        admissions_file = self.hosp_path / "admissions.csv"
        if not admissions_file.exists():
            logger.warning(f"Admissions file not found: {admissions_file}")
            return pd.DataFrame()
        
        df = pd.read_csv(admissions_file)
        logger.info(f"Loaded {len(df)} admission records")
        return df
    
    def load_diagnoses(self) -> pd.DataFrame:
        """Load ICD diagnoses"""
        diagnoses_file = self.hosp_path / "diagnoses_icd.csv"
        if not diagnoses_file.exists():
            logger.warning(f"Diagnoses file not found: {diagnoses_file}")
            return pd.DataFrame()
        
        df = pd.read_csv(diagnoses_file)
        logger.info(f"Loaded {len(df)} diagnosis records")
        return df
    
    def load_lab_events(self) -> pd.DataFrame:
        """Load lab events with timestamps"""
        labevents_file = self.icu_path / "labevents.csv"
        if not labevents_file.exists():
            logger.warning(f"Lab events file not found: {labevents_file}")
            return pd.DataFrame()
        
        df = pd.read_csv(labevents_file)
        logger.info(f"Loaded {len(df)} lab event records")
        return df
    
    def load_prescriptions(self) -> pd.DataFrame:
        """Load medication prescriptions"""
        prescriptions_file = self.hosp_path / "prescriptions.csv"
        if not prescriptions_file.exists():
            logger.warning(f"Prescriptions file not found: {prescriptions_file}")
            return pd.DataFrame()
        
        df = pd.read_csv(prescriptions_file)
        logger.info(f"Loaded {len(df)} prescription records")
        return df
    
    def load_procedures(self) -> pd.DataFrame:
        """Load surgical procedures"""
        procedures_file = self.hosp_path / "procedures_icd.csv"
        if not procedures_file.exists():
            logger.warning(f"Procedures file not found: {procedures_file}")
            return pd.DataFrame()
        
        df = pd.read_csv(procedures_file)
        logger.info(f"Loaded {len(df)} procedure records")
        return df


class EndoVisDataLoader:
    """Load and process EndoVis-18-VQLA dataset"""
    
    def __init__(self, endovis_path: Path = ENDOVIS_DIR):
        self.endovis_path = endovis_path
        self.train_path = endovis_path / "train"
        self.val_path = endovis_path / "val"
    
    def load_vqla_labels(self) -> Dict[str, List[Dict]]:
        """Load VQLA labels from JSON files in image directories"""
        vqla_data = {"train": [], "val": []}
        
        for split in ["train", "val"]:
            split_path = self.train_path if split == "train" else self.val_path
            image_dir = split_path / "image"
            
            if not image_dir.exists():
                logger.warning(f"Image directory not found: {image_dir}")
                continue
            
            # Look for VQLA label files
            for json_file in image_dir.glob("*.json"):
                try:
                    with open(json_file) as f:
                        data = json.load(f)
                        vqla_data[split].append({
                            "file": json_file.name,
                            "data": data
                        })
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse JSON: {json_file}")
        
        logger.info(f"Loaded {len(vqla_data['train'])} training and "
                   f"{len(vqla_data['val'])} validation VQLA labels")
        return vqla_data
    
    def get_image_paths(self, split: str = "train") -> List[Path]:
        """Get all image file paths"""
        split_path = self.train_path if split == "train" else self.val_path
        image_dir = split_path / "image"
        
        if not image_dir.exists():
            return []
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        images = [f for f in image_dir.iterdir() 
                  if f.suffix.lower() in image_extensions]
        
        logger.info(f"Found {len(images)} images in {split} split")
        return images


class MedicalO1DataLoader:
    """Load and process Medical-o1 Reasoning dataset"""
    
    def __init__(self, o1_path: Path = MEDICAL_O1_PATH):
        self.o1_path = o1_path
    
    def load_sft_data(self) -> List[Dict]:
        """Load Medical-o1 Supervised Fine-Tuning data"""
        if not self.o1_path.exists():
            logger.warning(f"Medical-o1 file not found: {self.o1_path}")
            return []
        
        try:
            with open(self.o1_path) as f:
                data = json.load(f)
                # Handle both single list and wrapped format
                if isinstance(data, dict) and "data" in data:
                    data = data["data"]
                elif not isinstance(data, list):
                    data = [data]
        except json.JSONDecodeError:
            logger.error("Failed to parse Medical-o1 JSON")
            return []
        
        logger.info(f"Loaded {len(data)} Medical-o1 training examples")
        return data


class DataCleaner:
    """Handle data cleaning and missing value imputation"""
    
    @staticmethod
    def clean_lab_results(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean lab results - handle missing values and outliers
        
        Args:
            df: Lab events dataframe
        
        Returns:
            Cleaned dataframe
        """
        df = df.copy()
        
        # Remove rows with all NaN values
        df = df.dropna(how='all')
        
        # For numeric columns, use median imputation
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
        
        # For categorical columns, use mode or 'Unknown'
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                mode_val = df[col].mode()
                fill_val = mode_val[0] if len(mode_val) > 0 else 'Unknown'
                df[col].fillna(fill_val, inplace=True)
        
        logger.info(f"Cleaned dataframe: {len(df)} rows, {df.shape[1]} columns")
        return df
    
    @staticmethod
    def link_patient_history(patient_df: pd.DataFrame, 
                            clinical_df: pd.DataFrame,
                            on: str = 'subject_id') -> pd.DataFrame:
        """
        Link patient demographics with clinical history
        
        Args:
            patient_df: Patient demographics
            clinical_df: Clinical data
            on: Column to join on
        
        Returns:
            Merged dataframe
        """
        if on not in patient_df.columns or on not in clinical_df.columns:
            logger.warning(f"Column '{on}' not found in one of the dataframes")
            return clinical_df
        
        merged = pd.merge(clinical_df, patient_df[[on, 'anchor_age', 'gender']], 
                         on=on, how='left')
        
        logger.info(f"Linked {len(merged)} records across datasets")
        return merged
    
    @staticmethod
    def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column naming conventions"""
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]
        return df


class DataPipeline:
    """Main data pipeline orchestrating all loading and cleaning"""
    
    def __init__(self):
        self.mimic_loader = MIMICDataLoader()
        self.endovis_loader = EndoVisDataLoader()
        self.o1_loader = MedicalO1DataLoader()
        self.cleaner = DataCleaner()
    
    def ingest_clinical_data(self) -> Dict[str, pd.DataFrame]:
        """Load and clean MIMIC-IV clinical data"""
        logger.info("Starting clinical data ingestion...")
        
        clinical_data = {
            'patients': self.cleaner.normalize_column_names(
                self.cleaner.clean_lab_results(self.mimic_loader.load_patients())
            ),
            'admissions': self.cleaner.normalize_column_names(
                self.mimic_loader.load_admissions()
            ),
            'diagnoses': self.cleaner.normalize_column_names(
                self.mimic_loader.load_diagnoses()
            ),
            'lab_events': self.cleaner.normalize_column_names(
                self.cleaner.clean_lab_results(self.mimic_loader.load_lab_events())
            ),
            'prescriptions': self.cleaner.normalize_column_names(
                self.mimic_loader.load_prescriptions()
            ),
            'procedures': self.cleaner.normalize_column_names(
                self.mimic_loader.load_procedures()
            )
        }
        
        # Link patient histories
        if not clinical_data['patients'].empty and not clinical_data['lab_events'].empty:
            clinical_data['lab_events'] = self.cleaner.link_patient_history(
                clinical_data['patients'],
                clinical_data['lab_events']
            )
        
        logger.info("Clinical data ingestion completed")
        return clinical_data
    
    def ingest_surgical_data(self) -> Dict:
        """Load EndoVis-18-VQLA surgical data"""
        logger.info("Starting surgical data ingestion...")
        
        surgical_data = {
            'vqla_labels': self.endovis_loader.load_vqla_labels(),
            'train_images': self.endovis_loader.get_image_paths('train'),
            'val_images': self.endovis_loader.get_image_paths('val')
        }
        
        logger.info("Surgical data ingestion completed")
        return surgical_data
    
    def ingest_reasoning_data(self) -> List[Dict]:
        """Load Medical-o1 reasoning dataset"""
        logger.info("Starting reasoning data ingestion...")
        
        reasoning_data = self.o1_loader.load_sft_data()
        
        logger.info("Reasoning data ingestion completed")
        return reasoning_data
    
    def full_pipeline(self) -> Tuple[Dict[str, pd.DataFrame], Dict, List[Dict]]:
        """Execute complete data ingestion pipeline"""
        logger.info("=" * 50)
        logger.info("MedSync AI - Data Ingestion Pipeline Started")
        logger.info("=" * 50)
        
        clinical_data = self.ingest_clinical_data()
        surgical_data = self.ingest_surgical_data()
        reasoning_data = self.ingest_reasoning_data()
        
        logger.info("=" * 50)
        logger.info("Data Ingestion Pipeline Completed Successfully")
        logger.info("=" * 50)
        
        return clinical_data, surgical_data, reasoning_data


if __name__ == "__main__":
    pipeline = DataPipeline()
    clinical, surgical, reasoning = pipeline.full_pipeline()
    
    print("\n--- Clinical Data Summary ---")
    for key, df in clinical.items():
        print(f"{key}: {len(df)} records, {df.shape[1]} columns")
    
    print("\n--- Surgical Data Summary ---")
    for key in surgical.keys():
        if isinstance(surgical[key], list):
            print(f"{key}: {len(surgical[key])} items")
        elif isinstance(surgical[key], dict):
            print(f"{key}: {len(surgical[key])} keys")
    
    print(f"\n--- Reasoning Data ---")
    print(f"total examples: {len(reasoning)}")

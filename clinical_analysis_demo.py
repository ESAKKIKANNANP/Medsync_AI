#!/usr/bin/env python3
"""
MedSync AI Clinical Analysis - Professional Demonstration
Shows how to use the clinical analysis endpoints with the Gemini API
"""

import requests
import json
import base64
from pathlib import Path

# API Endpoint
API_BASE_URL = "http://localhost:8000"

def create_sample_kab_report():
    """Create a sample KAB (Knowledge and Brief) report"""
    return {
        "patient_id": "P12345",
        "age": 68,
        "gender": "Male",
        "chief_complaint": "Difficulty swallowing for 3 weeks",
        "presenting_symptoms": [
            "Progressive dysphagia to solids and liquids",
            "Chest pain on swallowing",
            "Unintentional weight loss of 5kg",
            "Regurgitation of food",
            "Mild hoarseness"
        ],
        "medical_history": [
            "Hypertension (10 years)",
            "Type 2 diabetes (8 years)",
            "GERD (5 years)",
            "Previous smoking history (quit 10 years ago)"
        ],
        "medications": [
            "Lisinopril 10mg daily",
            "Metformin 850mg BID",
            "Omeprazole 20mg daily",
            "Atorvastatin 20mg daily"
        ],
        "allergies": [
            "Penicillin (anaphylaxis)"
        ],
        "vital_signs": {
            "BP": "145/88 mmHg",
            "HR": "92 bpm",
            "RR": "18 breaths/min",
            "Temperature": "37.1°C",
            "Weight": "72 kg",
            "BMI": "24"
        },
        "lab_results": {
            "Hemoglobin": "13.5 g/dL",
            "WBC": "7.2 × 10³/μL",
            "Platelets": "245 × 10³/μL",
            "Fasting glucose": "156 mg/dL",
            "HbA1c": "7.8%",
            "Albumin": "3.2 g/dL (low)",
            "Alkaline phosphatase": "92 U/L",
            "AST": "28 U/L",
            "ALT": "32 U/L"
        },
        "imaging_findings": [
            "Barium esophagography: Irregular narrowing in mid-esophagus (7-10cm), shouldering, apple-core appearance",
            "CT chest: Focal esophageal wall thickening at 28cm from incisors, no obvious distant metastases",
            "Regional lymph nodes: Mildly enlarged at 1.2cm"
        ],
        "diagnoses": [
            "Suspected esophageal carcinoma",
            "Dysphagia secondary to malignancy",
            "Malnutrition"
        ],
        "risk_factors": [
            "Age >65",
            "Male gender",
            "Previous smoking",
            "GERD history",
            "Alcohol use (moderate)",
            "Diabetes"
        ]
    }

def create_sample_medical_scans():
    """Create sample medical scan information"""
    return [
        {
            "scan_id": "SCAN_001",
            "scan_type": "Barium Esophagography",
            "description": "Irregular narrowing in mid-esophagus (7-10cm from incisors) with shouldering and apple-core appearance, suspicious for malignancy. No obvious proximal obstruction.",
            "base64_data": None  # In production, include base64 image data
        },
        {
            "scan_id": "SCAN_002",
            "scan_type": "CT",
            "description": "Focal esophageal wall thickening at 28cm from incisors, approximately 4cm in length. Regional lymphadenopathy with largest node measuring 1.2cm in short axis. No obvious distant metastases on current study.",
            "base64_data": None  # In production, include base64 image data
        }
    ]

def demonstrate_comprehensive_analysis():
    """Demonstrate comprehensive clinical analysis"""
    print("\n" + "="*70)
    print(" COMPREHENSIVE CLINICAL ANALYSIS")
    print("="*70 + "\n")
    
    kab_report = create_sample_kab_report()
    scans = create_sample_medical_scans()
    
    request_data = {
        "kab_report": kab_report,
        "medical_scans": scans
    }
    
    try:
        print("Sending comprehensive analysis request to Gemini API...")
        print(f"Patient: {kab_report['patient_id']} | Chief Complaint: {kab_report['chief_complaint']}\n")
        
        response = requests.post(
            f"{API_BASE_URL}/api/clinical/analyze",
            json=request_data,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Analysis Status: {result.get('status')}")
            print(f"📊 Model Used: {result.get('model')}")
            print(f"💾 Tokens Used: ~{result.get('input_tokens_used', 0)} tokens\n")
            print("PROFESSIONAL CLINICAL ANALYSIS:")
            print("-" * 70)
            print(result.get('analysis', 'No analysis generated'))
            print("-" * 70)
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
    
    except Exception as e:
        print(f"❌ Request failed: {e}")

def demonstrate_diagnosis_analysis():
    """Demonstrate diagnosis-specific analysis"""
    print("\n" + "="*70)
    print(" DIAGNOSIS-SPECIFIC ANALYSIS")
    print("="*70 + "\n")
    
    kab_report = create_sample_kab_report()
    scans = create_sample_medical_scans()
    
    request_data = {
        "kab_report": kab_report,
        "suspected_diagnosis": "Esophageal Squamous Cell Carcinoma",
        "medical_scans": scans
    }
    
    try:
        print("Analyzing suspected diagnosis: Esophageal Squamous Cell Carcinoma\n")
        
        response = requests.post(
            f"{API_BASE_URL}/api/clinical/diagnosis-analysis",
            json=request_data,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Analysis Status: {result.get('status')}")
            print(f"📊 Model Used: {result.get('model')}")
            print(f"💾 Tokens Used: ~{result.get('tokens_used', 0)} tokens\n")
            print("DIAGNOSTIC ASSESSMENT:")
            print("-" * 70)
            print(result.get('analysis', 'No analysis generated'))
            print("-" * 70)
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
    
    except Exception as e:
        print(f"❌ Request failed: {e}")

def demonstrate_clinical_question():
    """Demonstrate clinical question answering"""
    print("\n" + "="*70)
    print(" CLINICAL QUESTION ANSWERING")
    print("="*70 + "\n")
    
    kab_report = create_sample_kab_report()
    scans = create_sample_medical_scans()
    
    clinical_question = "What is the recommended next step for staging and treatment planning?"
    
    request_data = {
        "kab_report": kab_report,
        "clinical_question": clinical_question,
        "medical_scans": scans
    }
    
    try:
        print(f"Clinical Question: {clinical_question}\n")
        
        response = requests.post(
            f"{API_BASE_URL}/api/clinical/question",
            json=request_data,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Status: {result.get('status')}")
            print(f"💾 Tokens Used: ~{result.get('tokens_used', 0)} tokens\n")
            print("PROFESSIONAL RESPONSE:")
            print("-" * 70)
            print(result.get('answer', 'No answer generated'))
            print("-" * 70)
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
    
    except Exception as e:
        print(f"❌ Request failed: {e}")

def demonstrate_multiple_questions():
    """Demonstrate multiple clinical questions"""
    print("\n" + "="*70)
    print(" MULTIPLE CLINICAL QUESTIONS")
    print("="*70 + "\n")
    
    kab_report = create_sample_kab_report()
    scans = create_sample_medical_scans()
    
    questions = [
        "What are the differential diagnoses to consider?",
        "What additional investigations would you recommend?",
        "What is the role of endoscopy in this patient?",
        "What are the management options and their respective prognosis?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n📋 Question {i}: {question}")
        print("-" * 70)
        
        request_data = {
            "kab_report": kab_report,
            "clinical_question": question,
            "medical_scans": scans
        }
        
        try:
            response = requests.post(
                f"{API_BASE_URL}/api/clinical/question",
                json=request_data,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get('answer', 'No answer')
                # Print first 500 chars of answer
                print(answer[:500] + "..." if len(answer) > 500 else answer)
                print(f"\n💾 Tokens: ~{result.get('tokens_used', 0)}")
            else:
                print(f"❌ Error: {response.status_code}")
        
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main demonstration"""
    print("""
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║        MedSync AI - Professional Clinical Analysis System          ║
║           Real-time AI-Powered Medical Intelligence                ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
    """)
    
    # Check if API is running
    try:
        health = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if health.status_code == 200:
            print("✅ API Server Status: RUNNING")
            print(f"📊 Gemini API Available: {health.json().get('gemini_api_available', False)}\n")
        else:
            print("❌ API server not responding correctly")
            return
    except Exception as e:
        print(f"❌ Cannot connect to API server: {e}")
        print(f"   Make sure the server is running: python -m medsync_ai.api.server")
        return
    
    print("Demonstrating clinical analysis capabilities with real Gemini API...\n")
    
    # Run demonstrations
    demonstrate_comprehensive_analysis()
    demonstrate_diagnosis_analysis()
    demonstrate_clinical_question()
    demonstrate_multiple_questions()
    
    print("\n" + "="*70)
    print(" DEMONSTRATION COMPLETE")
    print("="*70)
    print("""
✅ Key Features Demonstrated:
   • Comprehensive clinical analysis from KAB reports
   • Diagnosis-specific deep-dive analysis
   • Clinical question answering
   • Multi-modal data integration
   • Professional medical-grade responses
   • Gemini API token utilization

📊 Token Usage:
   The system generates detailed professional medical analyses
   requiring substantial token usage for:
   • Complete clinical reasoning
   • Differential diagnosis generation
   • Evidence-based recommendations
   • Professional documentation

🎯 Use Cases:
   • Pre-surgical evaluation and planning
   • Patient risk stratification
   • Diagnostic support
   • Clinical decision support
   • Medical education and training
   """)

if __name__ == "__main__":
    main()

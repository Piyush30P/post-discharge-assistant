"""
Generate dummy patient discharge reports for nephrology patients
Creates 30 realistic patient records with various kidney-related conditions
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict

from src.config import DATA_DIR
from src.utils.logger import get_logger

logger = get_logger()


class DummyPatientGenerator:
    """Generate realistic dummy patient data for nephrology cases"""
    
    # Patient demographics
    FIRST_NAMES = [
        "John", "Mary", "James", "Patricia", "Robert", "Jennifer", "Michael", "Linda",
        "William", "Barbara", "David", "Elizabeth", "Richard", "Susan", "Joseph", "Jessica",
        "Thomas", "Sarah", "Charles", "Karen", "Christopher", "Nancy", "Daniel", "Lisa",
        "Matthew", "Betty", "Anthony", "Margaret", "Mark", "Sandra", "Donald", "Ashley"
    ]
    
    LAST_NAMES = [
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
        "Rodriguez", "Martinez", "Hernandez", "Lopez", "Wilson", "Anderson", "Thomas",
        "Taylor", "Moore", "Jackson", "Martin", "Lee", "Thompson", "White", "Harris",
        "Clark", "Lewis", "Robinson", "Walker", "Young", "Allen", "King"
    ]
    
    # Nephrology diagnoses
    DIAGNOSES = [
        {
            "name": "Chronic Kidney Disease Stage 3",
            "icd10": "N18.3",
            "severity": "moderate",
            "medications": ["Lisinopril 10mg daily", "Furosemide 20mg twice daily", "Calcium carbonate 500mg with meals"],
            "restrictions": "Low sodium (2g/day), fluid restriction (1.5L/day), protein restriction (0.8g/kg/day)"
        },
        {
            "name": "Acute Kidney Injury (AKI)",
            "icd10": "N17.9",
            "severity": "severe",
            "medications": ["IV fluids", "Discontinue NSAIDs", "Monitor electrolytes"],
            "restrictions": "Strict fluid monitoring, avoid nephrotoxic agents"
        },
        {
            "name": "Chronic Kidney Disease Stage 4",
            "icd10": "N18.4",
            "severity": "severe",
            "medications": ["Lisinopril 20mg daily", "Furosemide 40mg twice daily", "Sodium bicarbonate 650mg TID", "Erythropoietin 2000 units weekly"],
            "restrictions": "Low sodium (2g/day), fluid restriction (1L/day), protein restriction (0.6g/kg/day), potassium restriction"
        },
        {
            "name": "Nephrotic Syndrome",
            "icd10": "N04.9",
            "severity": "moderate",
            "medications": ["Prednisone 60mg daily", "Furosemide 40mg daily", "Lisinopril 10mg daily", "Atorvastatin 40mg daily"],
            "restrictions": "Low sodium diet, monitor for infection, avoid NSAIDs"
        },
        {
            "name": "Diabetic Nephropathy",
            "icd10": "E11.21",
            "severity": "moderate",
            "medications": ["Lisinopril 20mg daily", "Metformin 1000mg twice daily", "Insulin glargine 20 units at bedtime"],
            "restrictions": "Diabetic diet, blood glucose monitoring, protein restriction"
        },
        {
            "name": "Hypertensive Nephropathy",
            "icd10": "I12.9",
            "severity": "moderate",
            "medications": ["Amlodipine 10mg daily", "Lisinopril 20mg daily", "Hydrochlorothiazide 25mg daily"],
            "restrictions": "Low sodium diet (2g/day), DASH diet recommended"
        },
        {
            "name": "Glomerulonephritis",
            "icd10": "N05.9",
            "severity": "moderate",
            "medications": ["Prednisone 40mg daily", "Cyclophosphamide 100mg daily", "Lisinopril 10mg daily"],
            "restrictions": "Low sodium diet, monitor for infection, immunosuppression precautions"
        },
        {
            "name": "Polycystic Kidney Disease",
            "icd10": "Q61.3",
            "severity": "moderate",
            "medications": ["Tolvaptan 45mg/15mg daily", "Lisinopril 20mg daily", "Pain management as needed"],
            "restrictions": "Adequate hydration, blood pressure control, avoid contact sports"
        },
        {
            "name": "Kidney Stones (Nephrolithiasis)",
            "icd10": "N20.0",
            "severity": "mild",
            "medications": ["Tamsulosin 0.4mg daily", "Ketorolac 10mg PRN pain", "Potassium citrate 10mEq twice daily"],
            "restrictions": "Increase fluid intake (3L/day), low sodium diet, avoid high oxalate foods"
        },
        {
            "name": "End-Stage Renal Disease (ESRD) on Hemodialysis",
            "icd10": "N18.6",
            "severity": "severe",
            "medications": ["Sevelamer 800mg with meals", "Epoetin alfa 4000 units with dialysis", "Calcium acetate 667mg with meals"],
            "restrictions": "Fluid restriction (750mL/day), low potassium, low phosphorus, dialysis 3x/week"
        }
    ]
    
    WARNING_SIGNS = [
        "Decreased urine output or dark urine",
        "Swelling in legs, ankles, or feet",
        "Shortness of breath or difficulty breathing",
        "Chest pain or pressure",
        "Severe headache",
        "Confusion or altered mental status",
        "Severe nausea or vomiting",
        "Fever above 101°F",
        "Uncontrolled blood pressure (>180/110)",
        "Blood in urine"
    ]
    
    FOLLOW_UP_TYPES = [
        "Nephrology clinic in 1 week",
        "Nephrology clinic in 2 weeks",
        "Nephrology clinic in 1 month",
        "Primary care physician in 1 week",
        "Dialysis center 3 times per week",
        "Emergency department if symptoms worsen",
        "Lab work (BMP, CBC) in 3 days"
    ]
    
    def __init__(self):
        self.generated_names = set()
    
    def generate_unique_name(self) -> str:
        """Generate a unique patient name"""
        while True:
            first = random.choice(self.FIRST_NAMES)
            last = random.choice(self.LAST_NAMES)
            name = f"{first} {last}"
            if name not in self.generated_names:
                self.generated_names.add(name)
                return name
    
    def generate_discharge_date(self) -> str:
        """Generate a discharge date within the last 3 months"""
        days_ago = random.randint(1, 90)
        discharge_date = datetime.now() - timedelta(days=days_ago)
        return discharge_date.strftime("%Y-%m-%d")
    
    def generate_patient_report(self) -> Dict:
        """Generate a single patient discharge report"""
        diagnosis_info = random.choice(self.DIAGNOSES)
        
        # Select warning signs based on severity
        num_warnings = random.randint(3, 5)
        warning_signs = ", ".join(random.sample(self.WARNING_SIGNS, num_warnings))
        
        # Select follow-up appointments
        follow_ups = random.sample(self.FOLLOW_UP_TYPES, random.randint(1, 2))
        
        # Generate discharge instructions based on diagnosis
        discharge_instructions = self._generate_discharge_instructions(diagnosis_info)
        
        return {
            "patient_name": self.generate_unique_name(),
            "discharge_date": self.generate_discharge_date(),
            "primary_diagnosis": diagnosis_info["name"],
            "icd10_code": diagnosis_info["icd10"],
            "severity": diagnosis_info["severity"],
            "medications": diagnosis_info["medications"],
            "dietary_restrictions": diagnosis_info["restrictions"],
            "follow_up": ", ".join(follow_ups),
            "warning_signs": warning_signs,
            "discharge_instructions": discharge_instructions,
            "emergency_contact": self._generate_emergency_contact()
        }
    
    def _generate_discharge_instructions(self, diagnosis_info: Dict) -> str:
        """Generate specific discharge instructions"""
        base_instructions = [
            "Take all medications as prescribed",
            "Monitor blood pressure daily and record in log",
            "Weigh yourself daily at the same time",
            "Keep all follow-up appointments",
            "Report any warning signs immediately"
        ]
        
        if "Stage 4" in diagnosis_info["name"] or "ESRD" in diagnosis_info["name"]:
            base_instructions.extend([
                "Attend all dialysis sessions",
                "Strict fluid restriction compliance",
                "Monitor for signs of infection at dialysis access site"
            ])
        
        if "Diabetic" in diagnosis_info["name"]:
            base_instructions.extend([
                "Monitor blood glucose 4 times daily",
                "Follow diabetic meal plan",
                "Check feet daily for sores or injuries"
            ])
        
        return "; ".join(base_instructions)
    
    def _generate_emergency_contact(self) -> str:
        """Generate emergency contact information"""
        relationships = ["Spouse", "Adult child", "Sibling", "Parent", "Friend"]
        return f"{random.choice(relationships)} - ({random.randint(200, 999)}) {random.randint(200, 999)}-{random.randint(1000, 9999)}"
    
    def generate_multiple_patients(self, count: int = 30) -> List[Dict]:
        """Generate multiple patient reports"""
        logger.info(f"Generating {count} dummy patient records...")
        patients = []
        
        for i in range(count):
            patient = self.generate_patient_report()
            patients.append(patient)
            logger.debug(f"Generated patient {i+1}/{count}: {patient['patient_name']}")
        
        logger.info(f"✓ Successfully generated {count} patient records")
        return patients
    
    def save_to_json(self, patients: List[Dict], filename: str = "patient_reports.json"):
        """Save patient reports to JSON file"""
        filepath = DATA_DIR / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(patients, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Saved patient reports to: {filepath}")
        return filepath


def main():
    """Main function to generate and save patient data"""
    print("="*80)
    print("Dummy Patient Data Generator")
    print("="*80)
    
    generator = DummyPatientGenerator()
    
    # Generate 30 patients
    patients = generator.generate_multiple_patients(count=30)
    
    # Save to JSON
    filepath = generator.save_to_json(patients)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total Patients Generated: {len(patients)}")
    print(f"File Location: {filepath}")
    print("\nSample Patient Record:")
    print(json.dumps(patients[0], indent=2))
    
    # Statistics
    diagnoses_count = {}
    for patient in patients:
        diagnosis = patient["primary_diagnosis"]
        diagnoses_count[diagnosis] = diagnoses_count.get(diagnosis, 0) + 1
    
    print("\nDiagnosis Distribution:")
    for diagnosis, count in sorted(diagnoses_count.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {diagnosis}: {count} patient(s)")
    
    print("\n✓ Dummy data generation complete!")


if __name__ == "__main__":
    main()
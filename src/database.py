"""
Database Manager for Patient Records and Interactions
Uses SQLite for storing patient discharge reports and system interactions
"""

import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path

from src.config import DATABASE_PATH, DATA_DIR
from src.utils.logger import get_logger

logger = get_logger()


class DatabaseManager:
    """Manage patient data and interaction logging in SQLite"""
    
    def __init__(self, db_path: Path = DATABASE_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()
    
    def _initialize_database(self):
        """Create database tables if they don't exist"""
        logger.info(f"Initializing database at: {self.db_path}")
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Patients table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS patients (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_name TEXT UNIQUE NOT NULL,
                    discharge_date TEXT NOT NULL,
                    primary_diagnosis TEXT NOT NULL,
                    icd10_code TEXT,
                    severity TEXT,
                    medications TEXT,
                    dietary_restrictions TEXT,
                    follow_up TEXT,
                    warning_signs TEXT,
                    discharge_instructions TEXT,
                    emergency_contact TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Interactions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_name TEXT NOT NULL,
                    agent_type TEXT NOT NULL,
                    query TEXT NOT NULL,
                    response TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (patient_name) REFERENCES patients (patient_name)
                )
            """)
            
            # Create indexes for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_patient_name 
                ON patients(patient_name)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_interaction_patient 
                ON interactions(patient_name)
            """)
            
            conn.commit()
        
        logger.info("✓ Database initialized successfully")
    
    def add_patient(self, patient_data: Dict) -> bool:
        """
        Add a new patient to the database
        
        Args:
            patient_data: Dictionary containing patient information
            
        Returns:
            True if successful, False if patient already exists
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO patients (
                        patient_name, discharge_date, primary_diagnosis, icd10_code,
                        severity, medications, dietary_restrictions, follow_up,
                        warning_signs, discharge_instructions, emergency_contact
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    patient_data['patient_name'],
                    patient_data['discharge_date'],
                    patient_data['primary_diagnosis'],
                    patient_data.get('icd10_code', ''),
                    patient_data.get('severity', ''),
                    json.dumps(patient_data.get('medications', [])),
                    patient_data.get('dietary_restrictions', ''),
                    patient_data.get('follow_up', ''),
                    patient_data.get('warning_signs', ''),
                    patient_data.get('discharge_instructions', ''),
                    patient_data.get('emergency_contact', '')
                ))
                
                conn.commit()
                logger.log_database_operation(
                    "INSERT", 
                    f"Added patient: {patient_data['patient_name']}"
                )
                return True
                
        except sqlite3.IntegrityError:
            logger.warning(f"Patient already exists: {patient_data['patient_name']}")
            return False
        except Exception as e:
            logger.log_error("DatabaseManager.add_patient", e)
            raise
    
    def get_patient_by_name(self, patient_name: str) -> Optional[Dict]:
        """
        Retrieve patient data by name
        
        Args:
            patient_name: Name of the patient
            
        Returns:
            Dictionary with patient data or None if not found
        """
        logger.log_database_operation("SELECT", f"Looking up patient: {patient_name}")
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM patients 
                WHERE LOWER(patient_name) = LOWER(?)
            """, (patient_name,))
            
            row = cursor.fetchone()
            
            if row:
                patient_data = dict(row)
                # Parse JSON medications
                patient_data['medications'] = json.loads(patient_data['medications'])
                logger.info(f"✓ Found patient: {patient_name}")
                return patient_data
            else:
                logger.warning(f"Patient not found: {patient_name}")
                return None
    
    def search_patients(self, search_term: str) -> List[Dict]:
        """
        Search for patients by partial name match
        
        Args:
            search_term: Partial name to search for
            
        Returns:
            List of matching patient dictionaries
        """
        logger.log_database_operation("SEARCH", f"Searching for: {search_term}")
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM patients 
                WHERE LOWER(patient_name) LIKE LOWER(?)
            """, (f"%{search_term}%",))
            
            rows = cursor.fetchall()
            patients = []
            
            for row in rows:
                patient_data = dict(row)
                patient_data['medications'] = json.loads(patient_data['medications'])
                patients.append(patient_data)
            
            logger.info(f"✓ Found {len(patients)} matching patient(s)")
            return patients
    
    def log_interaction(
        self, 
        patient_name: str, 
        agent_type: str, 
        query: str, 
        response: str
    ):
        """
        Log a patient interaction
        
        Args:
            patient_name: Name of the patient
            agent_type: Type of agent (receptionist/clinical)
            query: Patient's query
            response: Agent's response
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO interactions (patient_name, agent_type, query, response)
                    VALUES (?, ?, ?, ?)
                """, (patient_name, agent_type, query, response))
                
                conn.commit()
                logger.log_database_operation(
                    "LOG_INTERACTION",
                    f"Patient: {patient_name}, Agent: {agent_type}"
                )
        except Exception as e:
            logger.log_error("DatabaseManager.log_interaction", e)
    
    def get_patient_interaction_history(self, patient_name: str) -> List[Dict]:
        """
        Get all interactions for a specific patient
        
        Args:
            patient_name: Name of the patient
            
        Returns:
            List of interaction dictionaries
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM interactions 
                WHERE patient_name = ?
                ORDER BY timestamp DESC
            """, (patient_name,))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    def get_all_patients(self) -> List[Dict]:
        """Get all patients in the database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM patients ORDER BY patient_name")
            rows = cursor.fetchall()
            
            patients = []
            for row in rows:
                patient_data = dict(row)
                patient_data['medications'] = json.loads(patient_data['medications'])
                patients.append(patient_data)
            
            return patients
    
    def bulk_add_patients(self, patients_list: List[Dict]) -> int:
        """
        Add multiple patients to database
        
        Args:
            patients_list: List of patient dictionaries
            
        Returns:
            Number of patients successfully added
        """
        logger.info(f"Bulk adding {len(patients_list)} patients...")
        
        success_count = 0
        for patient in patients_list:
            if self.add_patient(patient):
                success_count += 1
        
        logger.info(f"✓ Successfully added {success_count}/{len(patients_list)} patients")
        return success_count
    
    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Count patients
            cursor.execute("SELECT COUNT(*) FROM patients")
            patient_count = cursor.fetchone()[0]
            
            # Count interactions
            cursor.execute("SELECT COUNT(*) FROM interactions")
            interaction_count = cursor.fetchone()[0]
            
            # Get diagnoses distribution
            cursor.execute("""
                SELECT primary_diagnosis, COUNT(*) as count 
                FROM patients 
                GROUP BY primary_diagnosis
                ORDER BY count DESC
            """)
            diagnoses = cursor.fetchall()
            
            return {
                'total_patients': patient_count,
                'total_interactions': interaction_count,
                'diagnoses_distribution': diagnoses,
                'database_path': str(self.db_path)
            }
    
    def reset_database(self):
        """Delete and recreate all tables"""
        logger.warning("Resetting database - all data will be lost!")
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("DROP TABLE IF EXISTS interactions")
            cursor.execute("DROP TABLE IF EXISTS patients")
            
            conn.commit()
        
        self._initialize_database()
        logger.info("✓ Database reset complete")


def populate_database_from_json(json_file: Path = DATA_DIR / "patient_reports.json"):
    """Helper function to populate database from JSON file"""
    logger.info(f"Loading patient data from: {json_file}")
    
    if not json_file.exists():
        logger.error(f"JSON file not found: {json_file}")
        return
    
    with open(json_file, 'r', encoding='utf-8') as f:
        patients = json.load(f)
    
    db = DatabaseManager()
    success_count = db.bulk_add_patients(patients)
    
    logger.info(f"✓ Database populated with {success_count} patients")
    return success_count


def main():
    """Test database functionality"""
    print("\n" + "="*80)
    print("Database Manager - Setup and Test")
    print("="*80 + "\n")
    
    db = DatabaseManager()
    
    # Check if database needs population
    stats = db.get_database_stats()
    print(f"Current Database Stats:")
    print(f"  Total Patients: {stats['total_patients']}")
    print(f"  Total Interactions: {stats['total_interactions']}")
    print(f"  Database Path: {stats['database_path']}")
    
    if stats['total_patients'] == 0:
        print("\n" + "="*80)
        print("Populating database from JSON...")
        print("="*80)
        
        json_file = DATA_DIR / "patient_reports.json"
        if json_file.exists():
            count = populate_database_from_json(json_file)
            print(f"\n✓ Added {count} patients to database")
        else:
            print(f"\n❌ Patient data JSON not found at: {json_file}")
            print("Please run generate_dummy_data.py first!")
            return
    
    # Test patient retrieval
    print("\n" + "="*80)
    print("Testing Patient Retrieval")
    print("="*80)
    
    all_patients = db.get_all_patients()
    if all_patients:
        test_patient = all_patients[0]
        print(f"\nTest Patient: {test_patient['patient_name']}")
        
        # Retrieve by name
        patient = db.get_patient_by_name(test_patient['patient_name'])
        if patient:
            print(f"\n✓ Successfully retrieved patient data:")
            print(f"  Diagnosis: {patient['primary_diagnosis']}")
            print(f"  Discharge Date: {patient['discharge_date']}")
            print(f"  Medications: {len(patient['medications'])} prescribed")
        
        # Test search
        search_term = test_patient['patient_name'].split()[0]  # First name
        results = db.search_patients(search_term)
        print(f"\n✓ Search for '{search_term}' found {len(results)} patient(s)")
        
        # Test interaction logging
        db.log_interaction(
            patient_name=test_patient['patient_name'],
            agent_type="receptionist",
            query="How are you feeling today?",
            response="I'm feeling better, thank you!"
        )
        print(f"\n✓ Logged test interaction")
        
        # Get interaction history
        history = db.get_patient_interaction_history(test_patient['patient_name'])
        print(f"✓ Patient has {len(history)} interaction(s) in history")
    
    # Final stats
    stats = db.get_database_stats()
    print("\n" + "="*80)
    print("Final Database Stats:")
    print("="*80)
    print(f"  Total Patients: {stats['total_patients']}")
    print(f"  Total Interactions: {stats['total_interactions']}")
    
    if stats['diagnoses_distribution']:
        print("\n  Diagnoses Distribution:")
        for diagnosis, count in stats['diagnoses_distribution'][:5]:
            print(f"    - {diagnosis}: {count}")
    
    print("\n" + "="*80)
    print("✓ Database test complete!")
    print("="*80)


if __name__ == "__main__":
    main()
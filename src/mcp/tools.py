"""
MCP Tools for Post-Discharge Assistant
Provides tools for patient data retrieval, RAG search, and web search
"""

from typing import Dict, List, Optional
from src.database import DatabaseManager
from src.pinecone_manager import PineconeManager
from duckduckgo_search import DDGS
from src.utils.logger import get_logger

logger = get_logger()


class MCPTools:
    """Collection of tools for the multi-agent system"""
    
    def __init__(self):
        self.db = DatabaseManager()
        self.pinecone = PineconeManager()
        self.pinecone.connect_to_index()
        logger.info("✓ MCP Tools initialized")
    
    def get_patient_data(self, patient_name: str) -> Dict:
        """
        Retrieve patient discharge data from database
        
        Args:
            patient_name: Name of the patient
            
        Returns:
            Patient data dictionary or error message
        """
        logger.log_tool_call(
            "get_patient_data",
            {"patient_name": patient_name},
            "Retrieving patient data"
        )
        
        try:
            patient = self.db.get_patient_by_name(patient_name)
            
            if patient:
                # Format for agent consumption
                formatted_data = {
                    "success": True,
                    "patient_name": patient["patient_name"],
                    "discharge_date": patient["discharge_date"],
                    "diagnosis": patient["primary_diagnosis"],
                    "icd10_code": patient["icd10_code"],
                    "severity": patient["severity"],
                    "medications": patient["medications"],
                    "dietary_restrictions": patient["dietary_restrictions"],
                    "follow_up": patient["follow_up"],
                    "warning_signs": patient["warning_signs"],
                    "discharge_instructions": patient["discharge_instructions"],
                    "emergency_contact": patient["emergency_contact"]
                }
                
                logger.info(f"✓ Retrieved patient data for: {patient_name}")
                return formatted_data
            else:
                logger.warning(f"Patient not found: {patient_name}")
                return {
                    "success": False,
                    "error": f"Patient '{patient_name}' not found in database",
                    "suggestion": "Please verify the patient name spelling"
                }
                
        except Exception as e:
            logger.log_error("get_patient_data", e)
            return {
                "success": False,
                "error": str(e)
            }
    
    def search_medical_knowledge(self, query: str, top_k: int = 5) -> Dict:
        """
        Search medical knowledge base using RAG (Pinecone)
        
        Args:
            query: Medical question or topic
            top_k: Number of relevant chunks to retrieve
            
        Returns:
            Retrieved knowledge with sources
        """
        logger.log_rag_query(query, top_k, ["Nephrology Textbook"])
        
        try:
            results = self.pinecone.search(query, top_k=top_k)
            
            if results:
                # Format results for agent
                formatted_results = {
                    "success": True,
                    "query": query,
                    "num_results": len(results),
                    "results": []
                }
                
                for i, result in enumerate(results):
                    formatted_results["results"].append({
                        "rank": i + 1,
                        "relevance_score": round(result["score"], 4),
                        "text": result["metadata"].get("text", ""),
                        "source": result["metadata"].get("source", "Unknown"),
                        "page": result["metadata"].get("page", "N/A")
                    })
                
                logger.info(f"✓ Found {len(results)} relevant chunks for: {query}")
                return formatted_results
            else:
                logger.warning(f"No results found for: {query}")
                return {
                    "success": False,
                    "query": query,
                    "error": "No relevant information found in knowledge base"
                }
                
        except Exception as e:
            logger.log_error("search_medical_knowledge", e)
            return {
                "success": False,
                "error": str(e)
            }
    
    def web_search(self, query: str, max_results: int = 3) -> Dict:
        """
        Search the web for current medical information
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            Web search results
        """
        logger.log_web_search(query, max_results)
        
        try:
            ddgs = DDGS()
            results = list(ddgs.text(query, max_results=max_results))
            
            if results:
                formatted_results = {
                    "success": True,
                    "query": query,
                    "num_results": len(results),
                    "results": []
                }
                
                for i, result in enumerate(results):
                    formatted_results["results"].append({
                        "rank": i + 1,
                        "title": result.get("title", ""),
                        "snippet": result.get("body", ""),
                        "url": result.get("href", "")
                    })
                
                logger.info(f"✓ Found {len(results)} web results for: {query}")
                return formatted_results
            else:
                return {
                    "success": False,
                    "query": query,
                    "error": "No web results found"
                }
                
        except Exception as e:
            logger.log_error("web_search", e)
            return {
                "success": False,
                "error": str(e)
            }


def get_tool_descriptions() -> List[Dict]:
    """Get tool descriptions for agent function calling"""
    return [
        {
            "name": "get_patient_data",
            "description": "Retrieves patient discharge information from the database including diagnosis, medications, restrictions, and follow-up instructions. Use this when a patient asks about their discharge information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_name": {
                        "type": "string",
                        "description": "Full name of the patient (e.g., 'John Smith')"
                    }
                },
                "required": ["patient_name"]
            }
        },
        {
            "name": "search_medical_knowledge",
            "description": "Searches the comprehensive nephrology medical knowledge base for specific medical information. Use this to answer clinical questions about kidney diseases, treatments, and medical procedures.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Medical question or topic to search for (e.g., 'chronic kidney disease treatment options')"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of relevant chunks to retrieve (default: 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        },
        {
            "name": "web_search",
            "description": "Searches the web for current medical information. Use this as a fallback when the knowledge base doesn't have recent information or for current medical guidelines.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Web search query"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 3)",
                        "default": 3
                    }
                },
                "required": ["query"]
            }
        }
    ]
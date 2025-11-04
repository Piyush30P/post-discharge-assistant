"""
MCP Tools for Post-Discharge Assistant
Enhanced with Model Context Protocol integration
Provides tools for patient data retrieval, RAG search, and web search
UPDATED: Using both direct access and MCP server integration
"""

from typing import Dict, List, Optional
from src.database import DatabaseManager
from src.pinecone_manager import PineconeManager
from src.config import TAVILY_API_KEY
from src.utils.logger import get_logger

logger = get_logger()

# Try to import Tavily client
try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
    logger.info("✓ Tavily client available")
except ImportError as e:
    logger.warning(f"Tavily client not available: {e}")
    TavilyClient = None
    TAVILY_AVAILABLE = False

# Import MCP client for enhanced functionality
try:
    from src.mcp.client import (
        get_mcp_client,
        get_patient_data_sync,
        search_medical_knowledge_sync,
        web_search_medical_sync
    )
    MCP_AVAILABLE = True
    logger.info("✓ MCP client integration available")
except ImportError as e:
    logger.warning(f"MCP client not available: {e}")
    MCP_AVAILABLE = False


class MCPTools:
    """Collection of tools for the multi-agent system with MCP integration"""
    
    def __init__(self, use_mcp: bool = True):
        self.use_mcp = use_mcp and MCP_AVAILABLE
        
        # Initialize direct access components (fallback)
        self.db = DatabaseManager()
        self.pinecone = PineconeManager()
        self.pinecone.connect_to_index()
        
        # Initialize Tavily client if available
        if TAVILY_AVAILABLE and TAVILY_API_KEY:
            try:
                self.tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
                logger.info("✓ Tavily client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Tavily: {e}")
                self.tavily_client = None
        else:
            self.tavily_client = None
            if not TAVILY_AVAILABLE:
                logger.warning("Tavily client not available - web search will be limited")
            elif not TAVILY_API_KEY:
                logger.warning("TAVILY_API_KEY not configured - web search disabled")
        
        # Initialize MCP client if available
        if self.use_mcp:
            try:
                self.mcp_client = get_mcp_client()
                logger.info("✓ MCP Tools initialized with MCP server integration")
            except Exception as e:
                logger.warning(f"MCP initialization failed, falling back to direct access: {e}")
                self.use_mcp = False
        
        if not self.use_mcp:
            logger.info("✓ MCP Tools initialized with direct access mode")
    
    def get_patient_data(self, patient_name: str) -> Dict:
        """
        Retrieve patient discharge data from database
        Uses MCP server if available, falls back to direct access
        
        Args:
            patient_name: Name of the patient
            
        Returns:
            Patient data dictionary or error message
        """
        logger.log_tool_call(
            "get_patient_data",
            {"patient_name": patient_name, "method": "MCP" if self.use_mcp else "Direct"},
            "Retrieving patient data"
        )
        
        try:
            if self.use_mcp:
                # Use MCP server
                result = get_patient_data_sync(patient_name)
                logger.info(f"✓ Retrieved patient data via MCP for: {patient_name}")
                return result
            else:
                # Direct database access (original implementation)
                return self._get_patient_data_direct(patient_name)
                
        except Exception as e:
            logger.log_error("get_patient_data", e)
            # Fallback to direct access if MCP fails
            if self.use_mcp:
                logger.warning("MCP failed, falling back to direct access")
                return self._get_patient_data_direct(patient_name)
            else:
                return {"success": False, "error": str(e)}
    
    def _get_patient_data_direct(self, patient_name: str) -> Dict:
        """Direct database access for patient data"""
    def _get_patient_data_direct(self, patient_name: str) -> Dict:
        """Direct database access for patient data"""
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
                
                logger.info(f"✓ Retrieved patient data directly for: {patient_name}")
                return formatted_data
            else:
                logger.warning(f"Patient not found: {patient_name}")
                return {
                    "success": False,
                    "error": f"Patient '{patient_name}' not found in database",
                    "suggestion": "Please verify the patient name spelling"
                }
                
        except Exception as e:
            logger.log_error("get_patient_data_direct", e)
            return {
                "success": False,
                "error": str(e)
            }

    def search_medical_knowledge(self, query: str, top_k: int = 5) -> Dict:
        """
        Search medical knowledge base using RAG (Pinecone)
        Uses MCP server if available, falls back to direct access
        
        Args:
            query: Medical question or topic
            top_k: Number of relevant chunks to retrieve
            
        Returns:
            Retrieved knowledge with sources
        """
        logger.log_rag_query(query, top_k, ["Nephrology Textbook", "MCP" if self.use_mcp else "Direct"])
        
        try:
            if self.use_mcp:
                # Use MCP server
                result = search_medical_knowledge_sync(query, top_k)
                logger.info(f"✓ Medical knowledge search via MCP completed")
                return result
            else:
                # Direct Pinecone access (original implementation)
                return self._search_medical_knowledge_direct(query, top_k)
                
        except Exception as e:
            logger.log_error("search_medical_knowledge", e)
            # Fallback to direct access if MCP fails
            if self.use_mcp:
                logger.warning("MCP search failed, falling back to direct access")
                return self._search_medical_knowledge_direct(query, top_k)
            else:
                return {"success": False, "error": str(e)}

    def _search_medical_knowledge_direct(self, query: str, top_k: int = 5) -> Dict:
        """Direct Pinecone access for medical knowledge search"""
        
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
        Search the web for current medical information using Tavily API
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            Web search results
        """
        logger.log_web_search(query, max_results)
        
        if not self.tavily_client:
            logger.error("Tavily client not initialized")
            return {
                "success": False,
                "error": "Web search service not available. Please check TAVILY_API_KEY in .env file"
            }
        
        try:
            # Use Tavily search with medical context
            response = self.tavily_client.search(
                query=query,
                max_results=max_results,
                search_depth="advanced",  # Use advanced search for better results
                include_domains=[],  # You can specify trusted medical domains
                exclude_domains=[]
            )
            
            results = response.get("results", [])
            
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
                        "snippet": result.get("content", ""),
                        "url": result.get("url", ""),
                        "score": result.get("score", 0)
                    })
                
                logger.info(f"✓ Found {len(results)} web results for: {query}")
                return formatted_results
            else:
                logger.warning(f"No web results found for: {query}")
                return {
                    "success": False,
                    "query": query,
                    "error": "No web results found"
                }
                
        except Exception as e:
            logger.log_error("web_search", e)
            return {
                "success": False,
                "error": f"Web search failed: {str(e)}"
            }

    # Additional MCP-enhanced methods
    def list_patients(self) -> Dict:
        """List all patients (MCP enhanced)"""
        try:
            if self.use_mcp:
                from src.mcp.client import get_mcp_client
                client = get_mcp_client()
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                return loop.run_until_complete(client.list_patients())
            else:
                patients = self.db.get_all_patients()
                return {
                    "success": True,
                    "total_patients": len(patients),
                    "patients": [
                        {
                            "name": p["patient_name"],
                            "diagnosis": p["primary_diagnosis"],
                            "discharge_date": p["discharge_date"],
                            "severity": p["severity"]
                        }
                        for p in patients
                    ]
                }
        except Exception as e:
            logger.log_error("list_patients", e)
            return {"success": False, "error": str(e)}

    def get_patient_medications(self, patient_name: str) -> Dict:
        """Get detailed patient medications (MCP enhanced)"""
        try:
            if self.use_mcp:
                from src.mcp.client import get_mcp_client
                client = get_mcp_client()
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                return loop.run_until_complete(client.get_patient_medications(patient_name))
            else:
                patient = self.db.get_patient_by_name(patient_name)
                if patient:
                    return {
                        "success": True,
                        "patient_name": patient_name,
                        "medications": patient["medications"],
                        "dietary_restrictions": patient["dietary_restrictions"],
                        "warning_signs": patient["warning_signs"]
                    }
                else:
                    return {"success": False, "error": f"Patient '{patient_name}' not found"}
        except Exception as e:
            logger.log_error("get_patient_medications", e)
            return {"success": False, "error": str(e)}

    def check_drug_interactions(self, medications: List[str]) -> Dict:
        """Check for drug interactions (MCP enhanced)"""
        try:
            if self.use_mcp:
                from src.mcp.client import get_mcp_client
                client = get_mcp_client()
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                return loop.run_until_complete(client.check_drug_interactions(medications))
            else:
                # Search knowledge base for drug interaction information
                query = f"drug interactions {' '.join(medications)}"
                results = self.pinecone.search(query, top_k=3)
                
                return {
                    "success": True,
                    "medications_checked": medications,
                    "interaction_info": [
                        {
                            "content": result["metadata"].get("text", ""),
                            "relevance_score": result["score"],
                            "source": result["metadata"].get("source", "Unknown")
                        }
                        for result in results
                    ] if results else [],
                    "disclaimer": "Always consult healthcare professionals for drug interaction advice"
                }
        except Exception as e:
            logger.log_error("check_drug_interactions", e)
            return {"success": False, "error": str(e)}


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
            "description": "Searches the web for current medical information using Tavily API. Use this as a fallback when the knowledge base doesn't have recent information or for current medical guidelines.",
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
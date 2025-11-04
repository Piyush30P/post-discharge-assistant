"""
MCP Server for Post-Discharge Medical AI Assistant
Implements Model Context Protocol server with medical tools
"""

import asyncio
import json
from typing import Any, Dict, List, Optional, Sequence
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    CallToolRequest,
    ListResourcesRequest,
    ListToolsRequest,
    ReadResourceRequest,
)
import logging
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.database import DatabaseManager
from src.pinecone_manager import PineconeManager
from src.config import TAVILY_API_KEY
from src.utils.logger import get_logger
from tavily import TavilyClient

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = get_logger()

# Initialize server
server = Server("post-discharge-assistant")

# Initialize components
db_manager = DatabaseManager()
pinecone_manager = PineconeManager()
pinecone_manager.connect_to_index()

# Initialize Tavily client
try:
    tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
except Exception as e:
    logger.error(f"Failed to initialize Tavily: {e}")
    tavily_client = None


@server.list_resources()
async def handle_list_resources() -> List[Resource]:
    """List available resources"""
    return [
        Resource(
            uri="file://patient-database",
            name="Patient Database",
            description="SQLite database containing patient discharge records",
            mimeType="application/json"
        ),
        Resource(
            uri="file://medical-knowledge",
            name="Medical Knowledge Base",
            description="Nephrology textbook knowledge stored in Pinecone vector database",
            mimeType="application/json"
        ),
        Resource(
            uri="file://patient-list",
            name="Available Patients",
            description="List of all patients in the database",
            mimeType="application/json"
        )
    ]


@server.read_resource()
async def handle_read_resource(uri: str) -> str:
    """Read resource content"""
    if uri == "file://patient-database":
        patients = db_manager.get_all_patients()
        return json.dumps({
            "description": "Complete patient database",
            "total_patients": len(patients),
            "patients": patients
        }, indent=2)
    
    elif uri == "file://patient-list":
        patients = db_manager.get_all_patients()
        patient_names = [p["patient_name"] for p in patients]
        return json.dumps({
            "available_patients": patient_names,
            "total_count": len(patient_names)
        }, indent=2)
    
    elif uri == "file://medical-knowledge":
        return json.dumps({
            "description": "Medical knowledge base from Comprehensive Clinical Nephrology",
            "vector_database": "Pinecone",
            "total_chunks": "~4000",
            "embedding_model": "all-MiniLM-L6-v2",
            "capabilities": [
                "Semantic search",
                "Medical question answering",
                "Evidence-based responses"
            ]
        }, indent=2)
    
    else:
        raise ValueError(f"Unknown resource: {uri}")


@server.list_tools()
async def handle_list_tools() -> List[Tool]:
    """List available tools"""
    return [
        Tool(
            name="get_patient_data",
            description="Retrieve patient discharge information from database",
            inputSchema={
                "type": "object",
                "properties": {
                    "patient_name": {
                        "type": "string",
                        "description": "Name of the patient to retrieve data for"
                    }
                },
                "required": ["patient_name"]
            }
        ),
        Tool(
            name="search_medical_knowledge",
            description="Search medical knowledge base using RAG (Retrieval Augmented Generation)",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Medical question or search query"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of relevant chunks to retrieve (default: 5)",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="web_search_medical",
            description="Search web for current medical information using Tavily API",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Medical search query for current information"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of search results (default: 3)",
                        "minimum": 1,
                        "maximum": 10,
                        "default": 3
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="list_patients",
            description="Get list of all available patients in the database",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="get_patient_medications",
            description="Get detailed medication information for a specific patient",
            inputSchema={
                "type": "object",
                "properties": {
                    "patient_name": {
                        "type": "string",
                        "description": "Name of the patient"
                    }
                },
                "required": ["patient_name"]
            }
        ),
        Tool(
            name="check_drug_interactions",
            description="Check for potential drug interactions using medical knowledge",
            inputSchema={
                "type": "object",
                "properties": {
                    "medications": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of medications to check for interactions"
                    }
                },
                "required": ["medications"]
            }
        )
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls"""
    try:
        if name == "get_patient_data":
            patient_name = arguments.get("patient_name")
            logger.info(f"MCP: Getting patient data for {patient_name}")
            
            patient = db_manager.get_patient_by_name(patient_name)
            
            if patient:
                result = {
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
            else:
                result = {
                    "success": False,
                    "error": f"Patient '{patient_name}' not found in database",
                    "available_patients": [p["patient_name"] for p in db_manager.get_all_patients()[:5]]
                }
            
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "search_medical_knowledge":
            query = arguments.get("query")
            top_k = arguments.get("top_k", 5)
            logger.info(f"MCP: Searching medical knowledge for: {query}")
            
            results = pinecone_manager.search(query, top_k=top_k)
            
            if results:
                formatted_results = {
                    "success": True,
                    "query": query,
                    "results_count": len(results),
                    "knowledge_chunks": [
                        {
                            "content": result["content"],
                            "page": result["page"],
                            "score": result["score"]
                        }
                        for result in results
                    ]
                }
            else:
                formatted_results = {
                    "success": False,
                    "error": "No relevant medical knowledge found",
                    "query": query
                }
            
            return [TextContent(type="text", text=json.dumps(formatted_results, indent=2))]
        
        elif name == "web_search_medical":
            query = arguments.get("query")
            max_results = arguments.get("max_results", 3)
            logger.info(f"MCP: Web searching for: {query}")
            
            if not tavily_client:
                return [TextContent(type="text", text=json.dumps({
                    "success": False,
                    "error": "Tavily API not available"
                }))]
            
            try:
                search_results = tavily_client.search(
                    query=f"medical {query}",
                    max_results=max_results,
                    search_depth="advanced",
                    include_domains=["pubmed.ncbi.nlm.nih.gov", "nejm.org", "bmj.com", "nature.com"]
                )
                
                formatted_results = {
                    "success": True,
                    "query": query,
                    "results": [
                        {
                            "title": result.get("title", ""),
                            "content": result.get("content", ""),
                            "url": result.get("url", ""),
                            "score": result.get("score", 0)
                        }
                        for result in search_results.get("results", [])
                    ]
                }
            except Exception as e:
                formatted_results = {
                    "success": False,
                    "error": f"Web search failed: {str(e)}"
                }
            
            return [TextContent(type="text", text=json.dumps(formatted_results, indent=2))]
        
        elif name == "list_patients":
            logger.info("MCP: Listing all patients")
            patients = db_manager.get_all_patients()
            
            result = {
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
            
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "get_patient_medications":
            patient_name = arguments.get("patient_name")
            logger.info(f"MCP: Getting medications for {patient_name}")
            
            patient = db_manager.get_patient_by_name(patient_name)
            
            if patient:
                result = {
                    "success": True,
                    "patient_name": patient_name,
                    "medications": patient["medications"],
                    "dietary_restrictions": patient["dietary_restrictions"],
                    "warning_signs": patient["warning_signs"]
                }
            else:
                result = {
                    "success": False,
                    "error": f"Patient '{patient_name}' not found"
                }
            
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "check_drug_interactions":
            medications = arguments.get("medications", [])
            logger.info(f"MCP: Checking drug interactions for: {medications}")
            
            # Search knowledge base for drug interaction information
            query = f"drug interactions {' '.join(medications)}"
            results = pinecone_manager.search(query, top_k=3)
            
            formatted_results = {
                "success": True,
                "medications_checked": medications,
                "interaction_info": [
                    {
                        "content": result["content"],
                        "relevance_score": result["score"]
                    }
                    for result in results
                ] if results else [],
                "disclaimer": "Always consult healthcare professionals for drug interaction advice"
            }
            
            return [TextContent(type="text", text=json.dumps(formatted_results, indent=2))]
        
        else:
            raise ValueError(f"Unknown tool: {name}")
    
    except Exception as e:
        logger.error(f"MCP tool error: {e}")
        error_result = {
            "success": False,
            "error": str(e),
            "tool": name
        }
        return [TextContent(type="text", text=json.dumps(error_result, indent=2))]


async def main():
    """Run the MCP server"""
    logger.info("Starting Post-Discharge Assistant MCP Server...")
    
    # Initialize database and vector store
    try:
        pinecone_manager.connect_to_index()
        logger.info("✓ MCP Server initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize MCP server: {e}")
        return
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="post-discharge-assistant",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=None,
                    experimental_capabilities=None,
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
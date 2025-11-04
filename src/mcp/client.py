"""
MCP Client for Post-Discharge Assistant
Integrates MCP server with the existing agent system
"""

import asyncio
import json
from typing import Dict, List, Optional, Any
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client
from contextlib import asynccontextmanager
import subprocess
import sys
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger()


class MCPClient:
    """MCP Client for connecting to the Post-Discharge Assistant MCP Server"""
    
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.server_process: Optional[subprocess.Popen] = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize MCP client connection"""
        if self._initialized:
            return
        
        try:
            logger.info("Initializing MCP Client...")
            
            # Start MCP server process
            server_path = Path(__file__).parent / "server.py"
            self.server_process = subprocess.Popen(
                [sys.executable, str(server_path)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Connect to server
            async with stdio_client(
                self.server_process.stdin,
                self.server_process.stdout
            ) as streams:
                self.session = ClientSession(*streams)
                await self.session.initialize()
                
                logger.info("✓ MCP Client initialized successfully")
                self._initialized = True
                
        except Exception as e:
            logger.error(f"Failed to initialize MCP client: {e}")
            if self.server_process:
                self.server_process.terminate()
            raise
    
    async def close(self):
        """Close MCP client connection"""
        if self.server_process:
            self.server_process.terminate()
            self.server_process = None
        self._initialized = False
        logger.info("MCP Client closed")
    
    @asynccontextmanager
    async def get_session(self):
        """Context manager for MCP session"""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Create new session for each request
            server_path = Path(__file__).parent / "server.py"
            server_process = subprocess.Popen(
                [sys.executable, str(server_path)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            async with stdio_client(
                server_process.stdin,
                server_process.stdout
            ) as streams:
                session = ClientSession(*streams)
                await session.initialize()
                yield session
                
        finally:
            if 'server_process' in locals():
                server_process.terminate()
    
    async def get_patient_data(self, patient_name: str) -> Dict:
        """Get patient data via MCP"""
        try:
            async with self.get_session() as session:
                result = await session.call_tool(
                    "get_patient_data",
                    {"patient_name": patient_name}
                )
                
                if result and result.content:
                    return json.loads(result.content[0].text)
                else:
                    return {"success": False, "error": "No response from MCP server"}
                    
        except Exception as e:
            logger.error(f"MCP get_patient_data error: {e}")
            return {"success": False, "error": str(e)}
    
    async def search_medical_knowledge(self, query: str, top_k: int = 5) -> Dict:
        """Search medical knowledge via MCP"""
        try:
            async with self.get_session() as session:
                result = await session.call_tool(
                    "search_medical_knowledge",
                    {"query": query, "top_k": top_k}
                )
                
                if result and result.content:
                    return json.loads(result.content[0].text)
                else:
                    return {"success": False, "error": "No response from MCP server"}
                    
        except Exception as e:
            logger.error(f"MCP search_medical_knowledge error: {e}")
            return {"success": False, "error": str(e)}
    
    async def web_search_medical(self, query: str, max_results: int = 3) -> Dict:
        """Web search via MCP"""
        try:
            async with self.get_session() as session:
                result = await session.call_tool(
                    "web_search_medical",
                    {"query": query, "max_results": max_results}
                )
                
                if result and result.content:
                    return json.loads(result.content[0].text)
                else:
                    return {"success": False, "error": "No response from MCP server"}
                    
        except Exception as e:
            logger.error(f"MCP web_search_medical error: {e}")
            return {"success": False, "error": str(e)}
    
    async def list_patients(self) -> Dict:
        """List all patients via MCP"""
        try:
            async with self.get_session() as session:
                result = await session.call_tool("list_patients", {})
                
                if result and result.content:
                    return json.loads(result.content[0].text)
                else:
                    return {"success": False, "error": "No response from MCP server"}
                    
        except Exception as e:
            logger.error(f"MCP list_patients error: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_patient_medications(self, patient_name: str) -> Dict:
        """Get patient medications via MCP"""
        try:
            async with self.get_session() as session:
                result = await session.call_tool(
                    "get_patient_medications",
                    {"patient_name": patient_name}
                )
                
                if result and result.content:
                    return json.loads(result.content[0].text)
                else:
                    return {"success": False, "error": "No response from MCP server"}
                    
        except Exception as e:
            logger.error(f"MCP get_patient_medications error: {e}")
            return {"success": False, "error": str(e)}
    
    async def check_drug_interactions(self, medications: List[str]) -> Dict:
        """Check drug interactions via MCP"""
        try:
            async with self.get_session() as session:
                result = await session.call_tool(
                    "check_drug_interactions",
                    {"medications": medications}
                )
                
                if result and result.content:
                    return json.loads(result.content[0].text)
                else:
                    return {"success": False, "error": "No response from MCP server"}
                    
        except Exception as e:
            logger.error(f"MCP check_drug_interactions error: {e}")
            return {"success": False, "error": str(e)}
    
    async def list_resources(self) -> List[Dict]:
        """List available MCP resources"""
        try:
            async with self.get_session() as session:
                resources = await session.list_resources()
                return [
                    {
                        "uri": resource.uri,
                        "name": resource.name,
                        "description": resource.description,
                        "mimeType": resource.mimeType
                    }
                    for resource in resources
                ]
                
        except Exception as e:
            logger.error(f"MCP list_resources error: {e}")
            return []
    
    async def list_tools(self) -> List[Dict]:
        """List available MCP tools"""
        try:
            async with self.get_session() as session:
                tools = await session.list_tools()
                return [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "inputSchema": tool.inputSchema
                    }
                    for tool in tools
                ]
                
        except Exception as e:
            logger.error(f"MCP list_tools error: {e}")
            return []


# Singleton instance
_mcp_client = None

def get_mcp_client() -> MCPClient:
    """Get singleton MCP client instance"""
    global _mcp_client
    if _mcp_client is None:
        _mcp_client = MCPClient()
    return _mcp_client


# Sync wrapper functions for backward compatibility
def sync_wrapper(async_func):
    """Wrapper to run async functions synchronously"""
    def wrapper(*args, **kwargs):
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(async_func(*args, **kwargs))
    return wrapper


# Sync versions of MCP functions
def get_patient_data_sync(patient_name: str) -> Dict:
    """Sync version of get_patient_data"""
    client = get_mcp_client()
    return sync_wrapper(client.get_patient_data)(patient_name)


def search_medical_knowledge_sync(query: str, top_k: int = 5) -> Dict:
    """Sync version of search_medical_knowledge"""
    client = get_mcp_client()
    return sync_wrapper(client.search_medical_knowledge)(query, top_k)


def web_search_medical_sync(query: str, max_results: int = 3) -> Dict:
    """Sync version of web_search_medical"""
    client = get_mcp_client()
    return sync_wrapper(client.web_search_medical)(query, max_results)
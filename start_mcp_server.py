#!/usr/bin/env python3
"""
MCP Server Startup Script
Starts the Post-Discharge Assistant MCP Server
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.mcp.server import main

if __name__ == "__main__":
    try:
        print("🏥 Starting Post-Discharge Assistant MCP Server...")
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n✅ MCP Server stopped")
    except Exception as e:
        print(f"❌ MCP Server error: {e}")
        sys.exit(1)
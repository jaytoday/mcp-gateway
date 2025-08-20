#!/usr/bin/env python3
"""
Demo MCP Server - Works without any API keys
Provides simple tools for testing the gateway
"""

import asyncio
import json
from datetime import datetime
from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool
import mcp.types as types

app = Server("demo-server")

@app.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available tools"""
    return [
        types.Tool(
            name="get_current_time",
            description="Get the current date and time",
            inputSchema={
                "type": "object",
                "properties": {
                    "format": {
                        "type": "string",
                        "description": "Time format (12h or 24h)",
                        "default": "24h"
                    }
                }
            }
        ),
        types.Tool(
            name="calculate",
            description="Perform basic arithmetic calculations",
            inputSchema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate (e.g., '2+2', '10*5')"
                    }
                },
                "required": ["expression"]
            }
        ),
        types.Tool(
            name="echo",
            description="Echo back the provided message",
            inputSchema={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Message to echo back"
                    }
                },
                "required": ["message"]
            }
        )
    ]

@app.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Handle tool calls"""
    
    if name == "get_current_time":
        time_format = arguments.get("format", "24h")
        now = datetime.now()
        if time_format == "12h":
            time_str = now.strftime("%Y-%m-%d %I:%M:%S %p")
        else:
            time_str = now.strftime("%Y-%m-%d %H:%M:%S")
        
        return [types.TextContent(
            type="text",
            text=f"Current time: {time_str}"
        )]
    
    elif name == "calculate":
        expression = arguments.get("expression", "")
        try:
            # Safe evaluation of basic math expressions
            allowed_chars = set('0123456789+-*/.() ')
            if not all(c in allowed_chars for c in expression):
                raise ValueError("Invalid characters in expression")
            
            result = eval(expression)
            return [types.TextContent(
                type="text",
                text=f"{expression} = {result}"
            )]
        except Exception as e:
            return [types.TextContent(
                type="text",
                text=f"Error calculating '{expression}': {str(e)}"
            )]
    
    elif name == "echo":
        message = arguments.get("message", "")
        return [types.TextContent(
            type="text",
            text=f"Echo: {message}"
        )]
    
    else:
        raise ValueError(f"Unknown tool: {name}")

async def main():
    # Run the server using stdin/stdout streams
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="demo-server",
                server_version="0.1.0",
                capabilities=app.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())

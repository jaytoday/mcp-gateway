import logging
from typing import Any, Dict, List, Optional
import httpx
import json
import os
from mcp.server.fastmcp import FastMCP
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.types import Receive, Scope, Send
import click
import uvicorn
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

# Configure logging
logger = logging.getLogger(__name__)

# Create the FastMCP server
mcp = FastMCP("gateway")

# Gateway URL - can be configured via environment variable
GATEWAY_URL = os.getenv("GATEWAY_URL", "http://localhost:8080")


@mcp.tool()
async def search_tools(query: str, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Search for MCP tools across all connected servers
    
    Args:
        query: Search query string
        session_id: Optional session ID to search within a specific session context
    """
    logger.info(f"search_tools called with query={repr(query)}, session_id={repr(session_id)}")
    logger.info(f"query type: {type(query)}")
    async with httpx.AsyncClient() as client:
        try:
            headers = {}
            if session_id:
                headers["X-Session-ID"] = session_id
                
            response = await client.post(
                f"{GATEWAY_URL}/mcp/search",
                json={"query": query},
                headers=headers,
                timeout=30.0
            )
            response.raise_for_status()

            result = response.json()

            # Extract the actual tools from the result
            if "result" in result and isinstance(result["result"], list):
                return result["result"]
            elif "result" in result and isinstance(result["result"], str):
                # Try to parse if it's a JSON string
                try:
                    parsed = json.loads(result["result"])
                    if isinstance(parsed, list):
                        return parsed
                except json.JSONDecodeError:
                    pass

            # Return empty list if no valid result
            return []

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error searching tools: {e}")
            return []
        except Exception as e:
            logger.error(f"Error searching tools: {e}")
            return []


@mcp.tool()
async def execute_tool(tool_name: str, args: Dict[str, Any], session_id: Optional[str] = None) -> str:
    """Execute a specific MCP tool on the appropriate server
    
    Args:
        tool_name: Name of the tool to execute
        args: Arguments to pass to the tool
        session_id: Optional session ID to execute within a specific session context
    """
    async with httpx.AsyncClient() as client:
        try:
            headers = {}
            if session_id:
                headers["X-Session-ID"] = session_id
                
            response = await client.post(
                f"{GATEWAY_URL}/mcp/execute",
                json={
                    "tool_name": tool_name,
                    "args": args
                },
                headers=headers,
                timeout=60.0  # Longer timeout for tool execution
            )
            response.raise_for_status()

            result = response.json()

            # Extract the actual result
            if "result" in result:
                if isinstance(result["result"], dict):
                    # If it's a dict with content, return the content
                    if "content" in result["result"]:
                        return result["result"]["content"]
                    else:
                        return json.dumps(result["result"])
                elif isinstance(result["result"], str):
                    # Try to parse if it's a JSON string
                    try:
                        parsed = json.loads(result["result"])
                        if isinstance(parsed, dict) and "content" in parsed:
                            return parsed["content"]
                        else:
                            return result["result"]
                    except json.JSONDecodeError:
                        return result["result"]
                else:
                    return str(result["result"])

            return "No result returned"

        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP error executing tool: {e}"
            logger.error(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Error executing tool: {e}"
            logger.error(error_msg)
            return error_msg


@click.command()
@click.option("--port", default=3000, help="Port to listen on for HTTP")
@click.option(
    "--log-level",
    default="INFO",
    help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
)
def main(port: int, log_level: str) -> int:
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create the session manager without event store (simple setup)
    session_manager = StreamableHTTPSessionManager(
        app=mcp._mcp_server,
        json_response=False,  # Use SSE streaming
    )

    # ASGI handler for streamable HTTP connections
    async def handle_streamable_http(
        scope: Scope, receive: Receive, send: Send
    ) -> None:
        await session_manager.handle_request(scope, receive, send)

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        """Context manager for managing session manager lifecycle."""
        async with session_manager.run():
            logger.info(
                "Application started with StreamableHTTP session manager!")
            try:
                yield
            finally:
                logger.info("Application shutting down...")

    # Create an ASGI application
    starlette_app = Starlette(
        debug=True,
        routes=[
            Mount("/mcp", app=handle_streamable_http),
        ],
        lifespan=lifespan,
    )

    logger.info(f"Starting MCP server on http://127.0.0.1:{port}/mcp")
    uvicorn.run(starlette_app, host="127.0.0.1", port=port)

    return 0


if __name__ == "__main__":
    main()

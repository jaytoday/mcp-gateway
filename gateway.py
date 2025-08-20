"""
MCP Gateway Service
"""

import asyncio
import json
import logging
import signal
import sys
import time
import os
import uuid
from typing import Dict, Any, Optional, Set, List
import redis.asyncio as redis
from datetime import datetime
from contextlib import AsyncExitStack
from dataclasses import dataclass
from dotenv import load_dotenv

# Import aiohttp for HTTP server
from aiohttp import web
import aiohttp_cors

# Import MCP client libraries
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.sse import sse_client

# Import AI providers for routing
import openai
import anthropic
import google.generativeai as genai
from pydantic import BaseModel, Field

# Import session management
from session import Session, SessionManager, InMemorySessionStore

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ServerConnection:
    """Holds connection information for a single MCP server"""
    uuid: str
    name: str
    session: ClientSession
    tools: List[Dict[str, Any]]
    transport: str = "stdio"  # Track transport type for heartbeat purposes


class ToolMatch(BaseModel):
    """Represents a single tool match from the routing system"""
    mcp_server: str = Field(
        description="The name of the MCP server containing this tool")
    tool_name: str = Field(description="The exact name of the tool")
    relevance_score: float = Field(
        description="Relevance score from 0.0 to 1.0", ge=0.0, le=1.0)
    reasoning: str = Field(
        description="Brief explanation of why this tool is relevant")


class ToolRoutingResult(BaseModel):
    """Result of tool routing based on a user query"""
    tools: List[ToolMatch] = Field(
        description="List of relevant tools, ordered by relevance score", max_items=5)


class MCPClient:
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        # Load environment variables from .env file
        load_dotenv()

        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        self.running = False
        self.queue_names = [
            "agent_queue:urgent",
            "agent_queue:high",
            "agent_queue:normal",
            "agent_queue:low",
        ]

        # MCP server connections (global, legacy support)
        self.connections: Dict[str, ServerConnection] = {}
        self.exit_stack = AsyncExitStack()

        # Session management
        self.session_store = InMemorySessionStore()
        self.session_manager = SessionManager(self.session_store)
        # session_id -> {server_uuid -> connection}
        self.session_connections: Dict[str, Dict[str, ServerConnection]] = {}
        # session_id -> exit_stack
        self.session_exit_stacks: Dict[str, AsyncExitStack] = {}

        # Register cleanup callback for expired sessions
        self.session_manager.register_cleanup_callback(
            self._cleanup_session_resources)

        # Initialize AI clients for routing
        self.ai_client = None
        self.ai_provider = None

        # Check for available API keys and initialize the appropriate client
        openai_api_key = os.getenv("OPENAI_API_KEY")
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        gemini_api_key = os.getenv("GEMINI_API_KEY")

        if anthropic_api_key:
            self.ai_client = anthropic.Anthropic(api_key=anthropic_api_key)
            self.ai_provider = "anthropic"
            logger.info("Using Anthropic for tool routing")
        elif gemini_api_key:
            genai.configure(api_key=gemini_api_key)
            self.ai_client = genai.GenerativeModel('gemini-1.5-flash')
            self.ai_provider = "gemini"
            logger.info("Using Gemini for tool routing")
        elif openai_api_key:
            self.ai_client = openai.OpenAI(api_key=openai_api_key)
            self.ai_provider = "openai"
            logger.info("Using OpenAI for tool routing")
        else:
            logger.warning(
                "No AI API keys found (ANTHROPIC_API_KEY, GEMINI_API_KEY, or OPENAI_API_KEY) - tool routing will use fallback string matching")
            self.ai_client = None
            self.ai_provider = None

        # HTTP server app
        self.app = None
        self.http_port = int(os.getenv("GATEWAY_HTTP_PORT", "8080"))

    async def connect(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = await redis.from_url(self.redis_url)
            await self.redis_client.ping()
            logger.info("Connected to Redis successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    async def disconnect(self):
        """Close Redis connection and MCP connections"""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("Disconnected from Redis")

        # Clean up MCP connections
        await self.exit_stack.aclose()
        logger.info("Closed all MCP connections")

        # Clean up session exit stacks
        for exit_stack in self.session_exit_stacks.values():
            await exit_stack.aclose()

        # Stop session cleanup task
        await self.session_manager.stop_cleanup_task()

    async def connect_to_server(self, server_input: str):
        """Connect to an MCP server based on input

        Args:
            server_input: Either an npx command or path to executable
        """
        # Parse the input to determine connection type
        parts = server_input.strip().split()
        if not parts:
            logger.error("Empty server input")
            return

        # Handle npx commands
        if parts[0] == "npx":
            # Extract package name and args from npx command
            if len(parts) < 2:
                logger.error("Invalid npx command: missing package name")
                return

            package_name = parts[1]
            additional_args = parts[2:] if len(parts) > 2 else []

            # Create server name from package name
            server_name = package_name.split('/')[-1].replace('@', '')

            # Set up npx parameters with -y flag
            args = ["-y", package_name] + additional_args

            # Create environment
            env = os.environ.copy()

            await self._connect_with_params(server_name, "npx", args, env, "stdio")

        # Handle executable paths (.py or .js files)
        elif parts[0].endswith('.py') or parts[0].endswith('.js'):
            server_script_path = parts[0]

            # Validate file exists
            if not os.path.exists(server_script_path):
                logger.error(f"Server script not found: {server_script_path}")
                return

            is_python = server_script_path.endswith('.py')
            command = "python" if is_python else "node"

            # Extract server name from file path
            server_name = os.path.basename(
                server_script_path).rsplit('.', 1)[0]

            # Create environment
            env = os.environ.copy()

            await self._connect_with_params(server_name, command, [server_script_path], env, "stdio")

        else:
            logger.error(f"Unsupported server input format: {server_input}")
            logger.info(
                "Expected formats: 'npx <package-name> [args...]' or '/path/to/server.py|.js'")

    async def _connect_with_params(self, name: str, command: str, args: List[str], env: Dict[str, str] = None, transport: str = "stdio"):
        """Internal method to connect with specific parameters"""
        try:
            logger.info(
                f"Connecting to MCP server '{name}' with command: {command} {args} using transport: {transport}")

            # Handle different transport types
            if transport == "stdio":
                # Set up server parameters for stdio
                server_params = StdioServerParameters(
                    command=command,
                    args=args,
                    env=env
                )

                # Create and connect to the server
                stdio_transport = await self.exit_stack.enter_async_context(
                    stdio_client(server_params)
                )
                stdio, write = stdio_transport
                session = await self.exit_stack.enter_async_context(
                    ClientSession(stdio, write)
                )

            elif transport == "http":
                # For HTTP transport, the last arg should be the URL
                if args and args[-1].startswith("http"):
                    url = args[-1]
                    logger.info(f"Connecting to HTTP MCP server at: {url}")

                    # Create HTTP transport
                    streams_context = await self.exit_stack.enter_async_context(
                        streamablehttp_client(url=url, headers={})
                    )
                    read_stream, write_stream, _ = streams_context
                    session = await self.exit_stack.enter_async_context(
                        ClientSession(read_stream, write_stream)
                    )
                else:
                    raise ValueError(f"Invalid HTTP URL in args: {args}")

            elif transport == "sse":
                # For SSE transport, the last arg should be the URL
                if args and args[-1].startswith("http"):
                    url = args[-1]
                    logger.info(f"Connecting to SSE MCP server at: {url}")

                    # Create SSE transport
                    streams_context = await self.exit_stack.enter_async_context(
                        sse_client(url=url, headers={})
                    )
                    # SSE client returns only 2 values (read_stream, write_stream)
                    read_stream, write_stream = streams_context
                    session = await self.exit_stack.enter_async_context(
                        ClientSession(read_stream, write_stream)
                    )
                else:
                    raise ValueError(f"Invalid SSE URL in args: {args}")

            else:
                raise ValueError(f"Unsupported transport type: {transport}")

            # Initialize the session
            await session.initialize()

            # List available tools
            response = await session.list_tools()
            tools = []
            for tool in response.tools:
                tool_info = {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.inputSchema
                }
                tools.append(tool_info)

            # Generate UUID for this server
            server_uuid = str(uuid.uuid4())

            # Store connection with transport type
            self.connections[server_uuid] = ServerConnection(
                uuid=server_uuid,
                name=name,
                session=session,
                tools=tools,
                transport=transport
            )

            logger.info(
                f"Connected to '{name}' server (UUID: {server_uuid}) with {len(tools)} tools: {[t['name'] for t in tools]}")

            # Update tool inventory in Redis
            await self.update_tool_inventory()

        except Exception as e:
            logger.error(f"Failed to connect to MCP server '{name}': {e}")
            raise

    def search_tools(self, query: str) -> list:
        """Search for available MCP tools matching the query using LLM routing"""

        # Check if AI client is available
        if not hasattr(self, 'ai_client') or self.ai_client is None:
            logger.error("No AI client available for tool routing")
            return []

        # Prepare tool inventory for the LLM
        tool_inventory = []
        for server_uuid, connection in self.connections.items():
            for tool in connection.tools:
                tool_inventory.append({
                    "mcp_server": connection.name,
                    "mcp_server_uuid": server_uuid,
                    "tool_name": tool["name"],
                    "description": tool.get("description", ""),
                    "input_schema": tool.get("inputSchema", {})
                })

        if not tool_inventory:
            logger.warning("No tools available from connected MCP servers")
            return []

        return self._perform_tool_search(query, tool_inventory)

    def _perform_tool_search(self, query: str, tool_inventory: list) -> list:
        """Perform the actual tool search

        NOTE: Feel free to modify this function to implement custom routing logic or fallback mechanisms

        Args:
            query: The user's search query
            tool_inventory: List of available tools from connected MCP servers

        Returns:
            List of matching tools with relevance scores and reasoning
        """
        # Create the routing prompt
        system_prompt = """You are a tool routing system for MCP (Model Context Protocol) servers. Your job is to analyze a user query and select the most relevant tools that could help answer or fulfill that query.

                        You will be provided with:
                        1. A user query
                        2. A list of available tools with their descriptions and schemas
                        
                        Your task is to:
                        - Analyze the semantic intent of the query
                        - Match it against available tools based on their capabilities
                        - Return the most relevant tools (up to 5) ordered by relevance
                        
                        Focus on semantic understanding - for example:
                        - "search the web" should match tools like "exa_search", "web_search", or "search"
                        - "find information about X" should match search tools
                        - "what's the weather" should match weather-related tools
                        - Be smart about synonyms and related concepts
                        
                        Only return tools that are actually relevant to the query."""

        user_prompt = f"""Query: {query}

                        Available tools:
                        {json.dumps(tool_inventory, indent=2)}

                        Select the most relevant tools for this query.
                        """

        try:
            logger.info(
                f"Calling {self.ai_provider} API for tool routing with {len(tool_inventory)} tools")

            # Call the appropriate AI provider
            routing_result = self._call_ai_provider(system_prompt, user_prompt)

            logger.info(
                f"{self.ai_provider} returned {len(routing_result.tools)} tool matches")

            # Format results
            results = []
            for tool_match in routing_result.tools:
                # Find the full tool info from our inventory
                matching_tool = None
                for t in tool_inventory:
                    if t["mcp_server"] == tool_match.mcp_server and t["tool_name"] == tool_match.tool_name:
                        matching_tool = t
                        break

                if matching_tool:
                    results.append({
                        "mcp_server": matching_tool["mcp_server"],
                        "mcp_server_uuid": matching_tool["mcp_server_uuid"],
                        "tool_name": matching_tool["tool_name"],
                        "description": matching_tool["description"],
                        "input_schema": matching_tool["input_schema"],
                        "match_score": tool_match.relevance_score,
                        "reasoning": tool_match.reasoning
                    })
                else:
                    logger.warning(
                        f"Tool not found in inventory: {tool_match.mcp_server}/{tool_match.tool_name}")

            logger.info(
                f"LLM router found {len(results)} relevant tools for query: {query}")
            return results

        except Exception as e:
            logger.error(f"Error in LLM-based tool routing: {e}")
            return []

    def _call_ai_provider(self, system_prompt: str, user_prompt: str) -> ToolRoutingResult:
        """Call the appropriate AI provider based on configuration"""
        if self.ai_provider == "openai":
            return self._call_openai(system_prompt, user_prompt)
        elif self.ai_provider == "anthropic":
            return self._call_anthropic(system_prompt, user_prompt)
        elif self.ai_provider == "gemini":
            return self._call_gemini(system_prompt, user_prompt)
        else:
            raise ValueError(f"Unsupported AI provider: {self.ai_provider}")

    def _call_openai(self, system_prompt: str, user_prompt: str) -> ToolRoutingResult:
        """Call OpenAI API for tool routing"""
        completion = self.ai_client.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format=ToolRoutingResult,
            temperature=0.2,
            max_tokens=1000
        )
        return completion.choices[0].message.parsed

    def _call_anthropic(self, system_prompt: str, user_prompt: str) -> ToolRoutingResult:
        """Call Anthropic API for tool routing"""
        enhanced_prompt = f"""
        {user_prompt}\n\nPlease respond with a JSON object containing a 'tools' array with objects that have 'mcp_server', 'tool_name', 'relevance_score' (float between 0.0 and 1.0), and 'reasoning' fields.
        your response should be a JSON which can be converted into the following type:

        class ToolMatch(BaseModel):
            mcp_server: str = Field(
                description="The name of the MCP server containing this tool")
            tool_name: str = Field(description="The exact name of the tool")
            relevance_score: float = Field(
                description="Relevance score from 0.0 to 1.0", ge=0.0, le=1.0)
            reasoning: str = Field(
                description="Brief explanation of why this tool is relevant")
        """

        message = self.ai_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            temperature=0.2,
            system=system_prompt,
            messages=[{"role": "user", "content": enhanced_prompt}]
        )

        # Parse the response as JSON
        response_text = message.content[0].text
        try:
            import json
            response_data = json.loads(response_text)

            # Convert to ToolRoutingResult format
            tools = []
            for tool_data in response_data.get("tools", []):
                tools.append(ToolMatch(
                    mcp_server=tool_data.get("mcp_server", ""),
                    tool_name=tool_data.get("tool_name", ""),
                    relevance_score=float(
                        tool_data.get("relevance_score", 0.0)),
                    reasoning=tool_data.get("reasoning", "")
                ))

            return ToolRoutingResult(tools=tools)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to parse Anthropic response: {e}")
            return ToolRoutingResult(tools=[])

    def _call_gemini(self, system_prompt: str, user_prompt: str) -> ToolRoutingResult:
        """Call Gemini API for tool routing"""
        prompt = f"""
        {system_prompt}\n\n{user_prompt}\n\nPlease respond with a JSON object containing a 'tools' array with objects that have 'mcp_server', 'tool_name', 'relevance_score' (float between 0.0 and 1.0), and 'reasoning' fields.
        your response should be a JSON which can be converted into the following type:

        class ToolMatch(BaseModel):
            mcp_server: str = Field(
                description="The name of the MCP server containing this tool")
            tool_name: str = Field(description="The exact name of the tool")
            relevance_score: float = Field(
                description="Relevance score from 0.0 to 1.0", ge=0.0, le=1.0)
            reasoning: str = Field(
                description="Brief explanation of why this tool is relevant")
        """
        response = self.ai_client.generate_content(prompt)

        try:
            import json
            # Extract JSON from response
            response_text = response.text
            # Try to find JSON in the response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                response_data = json.loads(json_str)

                # Convert to ToolRoutingResult format
                tools = []
                for tool_data in response_data.get("tools", []):
                    tools.append(ToolMatch(
                        mcp_server=tool_data.get("mcp_server", ""),
                        tool_name=tool_data.get("tool_name", ""),
                        relevance_score=float(
                            tool_data.get("relevance_score", 0.0)),
                        reasoning=tool_data.get("reasoning", "")
                    ))

                return ToolRoutingResult(tools=tools)
            else:
                logger.error("No JSON found in Gemini response")
                return ToolRoutingResult(tools=[])
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to parse Gemini response: {e}")
            return ToolRoutingResult(tools=[])

    async def update_tool_inventory(self):
        """Update the tool inventory in Redis for external services"""
        if not self.redis_client:
            return

        try:
            # Build complete tool inventory
            tool_inventory = []
            for server_uuid, connection in self.connections.items():
                for tool in connection.tools:
                    tool_inventory.append({
                        "mcp_server": connection.name,
                        "mcp_server_uuid": server_uuid,
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "inputSchema": tool.get("inputSchema", {})
                    })

            # Store in Redis
            await self.redis_client.set(
                "gateway:tool_inventory",
                json.dumps(tool_inventory),
                ex=3600  # Expire after 1 hour
            )

            # Also update gateway status
            servers_info = {
                server_uuid: connection.name
                for server_uuid, connection in self.connections.items()
            }
            await self.redis_client.set(
                "gateway:status",
                json.dumps({
                    "status": "running",
                    "connected_servers": servers_info,
                    "total_tools": len(tool_inventory),
                    "timestamp": str(datetime.now())
                }),
                ex=60  # Expire after 1 minute (heartbeat)
            )

            logger.info(
                f"Updated tool inventory in Redis: {len(tool_inventory)} tools from {len(self.connections)} servers")

        except Exception as e:
            logger.error(f"Failed to update tool inventory in Redis: {e}")

    def fetch_tool_specs(self, tool_matches: list) -> list:
        """Given a list of tool matches from search_tools, return formatted specs"""
        results = []

        for match in tool_matches:
            tool_spec = self.fetch_single_tool_spec(
                # Try UUID first, fallback to name
                match.get("mcp_server_uuid", match.get("mcp_server")),
                match["tool_name"]
            )
            if tool_spec:
                # Add match metadata to the spec
                tool_spec["match_score"] = match.get("match_score", 1.0)
                tool_spec["reasoning"] = match.get("reasoning", "")
                results.append(tool_spec)

        return results

    def fetch_single_tool_spec(self, server_uuid: str, tool_name: str) -> Optional[Dict[str, Any]]:
        """Fetch the spec for a specific tool from a specific server (by UUID)"""
        # Check if server exists
        if server_uuid not in self.connections:
            logger.warning(
                f"Server UUID '{server_uuid}' not found in connections")
            return None

        connection = self.connections[server_uuid]

        # Find the tool in this server's tools
        for tool in connection.tools:
            if tool["name"] == tool_name:
                # Format the tool spec for the agent
                return {
                    "mcp_server": connection.name,
                    "mcp_server_uuid": server_uuid,
                    "tool_name": tool["name"],
                    "description": tool.get("description", ""),
                    "input_schema": tool.get("inputSchema", {})
                }

        logger.warning(
            f"Tool '{tool_name}' not found in server '{connection.name}' (UUID: {server_uuid})")
        return None

    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single message and return response"""
        request_id = message.get("request_id")
        action = message.get("action")
        payload = message.get("payload", {})

        print(f"printing message ... \n {str(message)}")
        # Parse payload if it's a JSON string
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse payload JSON: {payload}")
                # Keep payload as string if it can't be parsed
                pass

        logger.info(f"Processing message: {request_id} - Action: {action}")
        logger.info(f"Payload: {json.dumps(payload)}")

        # Base response
        response = {
            "request_id": request_id,
            "status": "completed",
            "timestamp": time.time(),
        }

        # Handle specific MCP actions
        if action == "mcp_search":
            query = payload.get("query", "")

            # Use the actual search_tools method to find relevant tools
            logger.info(f"Searching for tools matching query: '{query}'")
            relevant_tools = self.search_tools(query)
            logger.info(f"search_tools returned {len(relevant_tools)} matches")

            # Fetch detailed specs for the matching tools
            relevant_tool_specs = self.fetch_tool_specs(relevant_tools)
            logger.info(
                f"fetch_tool_specs returned {len(relevant_tool_specs)} specs")

            logger.info(
                f"Found {len(relevant_tool_specs)} tool specs for query '{query}':\n{json.dumps(relevant_tool_specs, indent=2)}"
            )

            # Return the tool specs as the result - serialize to JSON string for Redis
            response["result"] = json.dumps(relevant_tool_specs)
            logger.info(
                f"MCP Search completed: query='{query}' -> {len(relevant_tool_specs)} tools found"
            )

        elif action == "mcp_tool_call":
            tool_name = payload.get("tool_name", "")
            args = payload.get("args", {})

            logger.info(
                f"MCP Tool Call request: tool='{tool_name}' args={json.dumps(args)}")

            # Find which server has this tool
            server_uuid = self._get_service_for_tool(tool_name)
            if not server_uuid:
                error_msg = f"Tool '{tool_name}' not found in any connected server"
                logger.error(error_msg)
                response["status"] = "error"
                response["result"] = json.dumps({"error": error_msg})
                return response

            # Get the connection and session
            connection = self.connections[server_uuid]
            session = connection.session

            # Execute the tool call
            try:
                logger.info(
                    f"Executing tool '{tool_name}' on server '{connection.name}' (UUID: {server_uuid})")

                # Call the tool through the MCP session
                tool_result = await session.call_tool(tool_name, args)

                logger.info(f"Tool execution completed successfully")

                # Extract content from tool result
                if hasattr(tool_result, 'content'):
                    # Handle different content types
                    if isinstance(tool_result.content, list):
                        # Extract text content from list of content items
                        content_texts = []
                        for item in tool_result.content:
                            if hasattr(item, 'text'):
                                content_texts.append(item.text)
                            else:
                                content_texts.append(str(item))
                        result_content = "\n".join(content_texts)
                    else:
                        result_content = str(tool_result.content)
                else:
                    result_content = str(tool_result)

                response["result"] = json.dumps({
                    "success": True,
                    "server": connection.name,
                    "server_uuid": server_uuid,
                    "tool": tool_name,
                    "content": result_content
                })

            except Exception as e:
                error_msg = f"Failed to execute tool '{tool_name}': {str(e)}"
                logger.error(error_msg, exc_info=True)  # Add stack trace
                response["status"] = "error"
                response["result"] = json.dumps({"error": error_msg})

            logger.info(
                f"MCP Tool Call completed: tool='{tool_name}' status='{response['status']}'"
            )

        else:
            # Default echo for other actions
            response["result"] = f"ECHO: {json.dumps(message)}"

        return response

    def _get_service_for_tool(self, tool_name: str) -> Optional[str]:
        """Determine which service a tool belongs to (returns UUID)"""
        # Search through all connections to find which server has this tool
        for server_uuid, connection in self.connections.items():
            for tool in connection.tools:
                if tool["name"] == tool_name:
                    logger.info(
                        f"Tool '{tool_name}' found in server '{connection.name}' (UUID: {server_uuid})")
                    return server_uuid

        logger.warning(f"Tool '{tool_name}' not found in any connected server")
        return None

    async def send_sse_heartbeats(self):
        """Send heartbeat messages to SSE connections"""
        for server_uuid, connection in self.connections.items():
            if connection.transport == "sse":
                try:
                    # For SSE connections, we need to keep the connection alive
                    # The MCP protocol over SSE might handle this differently
                    # Try to list resources as a lightweight operation
                    await connection.session.list_resources()
                    logger.debug(
                        f"Sent heartbeat (list_resources) to SSE server '{connection.name}' (UUID: {server_uuid})")
                except AttributeError:
                    try:
                        # If list_resources is not available, try list_prompts
                        await connection.session.list_prompts()
                        logger.debug(
                            f"Sent heartbeat (list_prompts) to SSE server '{connection.name}' (UUID: {server_uuid})")
                    except:
                        # If neither works, the connection might handle heartbeats internally
                        logger.debug(
                            f"SSE server '{connection.name}' heartbeat handled by transport layer")
                except Exception as e:
                    logger.warning(
                        f"Failed to send heartbeat to SSE server '{connection.name}' (UUID: {server_uuid}): {e}")

    async def poll_queues(self):
        """Poll Redis queues for messages"""
        last_heartbeat = 0
        last_sse_heartbeat = 0
        while self.running:
            try:
                # Try each queue in priority order
                for queue_name in self.queue_names:
                    # First, check if there are any messages without removing them
                    messages = await self.redis_client.lrange(queue_name, 0, -1)

                    if messages:
                        logger.debug(
                            f"Found {len(messages)} messages in {queue_name}")
                        # Look for MCP-specific messages from the end (oldest first)
                        for i in range(len(messages) - 1, -1, -1):
                            try:
                                message = json.loads(messages[i])
                                action = message.get("action", "")

                                # Only process MCP-specific actions
                                if action in ["mcp_search", "mcp_tool_call"]:
                                    logger.info(
                                        f"Found MCP message in {queue_name}: {action}"
                                    )
                                    # Remove this specific message from the queue
                                    # Use LREM to remove exactly this message
                                    removed = await self.redis_client.lrem(
                                        queue_name, 1, messages[i]
                                    )

                                    if removed:
                                        logger.info(
                                            f"Processing MCP message: {action}")

                                        # Process the message
                                        response = await self.process_message(message)

                                        # Store response in Redis hash
                                        request_id = message.get("request_id")
                                        if request_id:
                                            response_key = f"request:{request_id}"

                                            # Update the request with response data
                                            result_value = response.get(
                                                "result", "")

                                            # Ensure result is a string for Redis storage
                                            if isinstance(result_value, (list, dict)):
                                                result_str = json.dumps(
                                                    result_value)
                                                logger.debug(
                                                    f"Serialized {type(result_value).__name__} result to JSON string")
                                            else:
                                                result_str = str(result_value)

                                            await self.redis_client.hset(
                                                response_key,
                                                mapping={
                                                    "status": response["status"],
                                                    "result": result_str,
                                                    "response_timestamp": str(
                                                        response["timestamp"]
                                                    ),
                                                },
                                            )

                                            logger.info(
                                                f"Stored response for request {request_id}"
                                            )
                                            logger.info(
                                                f"Response details: status={response['status']}, result_type={type(result_value).__name__}, result_length={len(result_str)}"
                                            )

                                        # Break after processing one message to re-check priorities
                                        break

                            except json.JSONDecodeError as e:
                                logger.error(f"Failed to parse message: {e}")
                            except Exception as e:
                                logger.error(f"Error processing message: {e}")

                    # Small delay between queue checks
                    await asyncio.sleep(0.1)

                # After checking all queues, wait 1 second before next poll cycle
                await asyncio.sleep(1.0)

            except Exception as e:
                logger.error(f"Error in poll_queues: {e}")
                await asyncio.sleep(1)  # Wait before retrying

            # Update heartbeat every 30 seconds
            current_time = time.time()
            if current_time - last_heartbeat > 30:
                await self.update_tool_inventory()
                last_heartbeat = current_time

            # Send SSE heartbeats every 5 seconds
            if current_time - last_sse_heartbeat > 5:
                await self.send_sse_heartbeats()
                last_sse_heartbeat = current_time

    async def handle_terminal_input(self):
        """Handle terminal input commands"""
        logger.info("Ready for terminal input. Commands:")
        logger.info(
            "  add_mcp_npx: <npx command> - Connect to an npx MCP server")
        logger.info(
            "  add_mcp_exec: <path> - Connect to an executable MCP server")
        logger.info("  list - List connected servers")
        logger.info("  list_all - List all tools from all servers")
        logger.info("  quit - Exit the gateway")

        while self.running:
            try:
                # Use asyncio to read stdin without blocking
                loop = asyncio.get_event_loop()
                future = loop.create_future()

                def stdin_callback():
                    line = sys.stdin.readline().strip()
                    if not future.done():
                        future.set_result(line)

                loop.add_reader(sys.stdin.fileno(), stdin_callback)

                try:
                    line = await asyncio.wait_for(future, timeout=0.1)
                    loop.remove_reader(sys.stdin.fileno())

                    if line:
                        await self.process_terminal_command(line)

                except asyncio.TimeoutError:
                    # No input available, continue
                    loop.remove_reader(sys.stdin.fileno())

            except Exception as e:
                logger.error(f"Error reading terminal input: {e}")

            await asyncio.sleep(0.1)

    async def process_terminal_command(self, command: str):
        """Process a terminal command"""
        command = command.strip()

        if command.startswith("add_mcp_npx:"):
            # Extract npx command
            npx_cmd = command[len("add_mcp_npx:"):].strip()
            if npx_cmd:
                await self.connect_to_server(npx_cmd)
            else:
                logger.error("Missing npx command")

        elif command.startswith("add_mcp_exec:"):
            # Extract executable path
            exec_path = command[len("add_mcp_exec:"):].strip()
            if exec_path:
                await self.connect_to_server(exec_path)
            else:
                logger.error("Missing executable path")

        elif command == "list":
            # List connected servers
            if self.connections:
                logger.info(f"Connected servers ({len(self.connections)}):")
                for server_uuid, conn in self.connections.items():
                    logger.info(
                        f"  - {conn.name} (UUID: {server_uuid}): {len(conn.tools)} tools")
                    for tool in conn.tools[:3]:  # Show first 3 tools
                        logger.info(f"    • {tool['name']}")
                    if len(conn.tools) > 3:
                        logger.info(f"    ... and {len(conn.tools) - 3} more")
            else:
                logger.info("No servers connected")

        elif command == "list_all":
            # List all tools from all servers
            if self.connections:
                total_tools = sum(len(conn.tools)
                                  for conn in self.connections.values())
                logger.info(
                    f"\nAll tools from {len(self.connections)} servers (Total: {total_tools} tools):\n")

                for server_uuid, conn in self.connections.items():
                    logger.info(
                        f"[{conn.name}] (UUID: {server_uuid}) ({len(conn.tools)} tools):")
                    for tool in conn.tools:
                        desc = tool.get('description', 'No description')
                        # Truncate long descriptions
                        if len(desc) > 80:
                            desc = desc[:77] + "..."
                        logger.info(f"  • {tool['name']}: {desc}")
                    logger.info("")  # Empty line between servers
            else:
                logger.info("No servers connected")

        elif command == "quit":
            logger.info("Shutting down...")
            self.running = False

        else:
            logger.warning(f"Unknown command: {command}")
            logger.info(
                "Available commands: add_mcp_npx:, add_mcp_exec:, list, list_all, quit")

    async def load_servers_from_file(self, filepath: str = "servers.txt"):
        """Load server configurations from servers.txt file"""
        try:
            if not os.path.exists(filepath):
                logger.warning(f"Servers file not found: {filepath}")
                return

            with open(filepath, 'r') as f:
                content = f.read().strip()

            if not content:
                logger.info("Servers file is empty")
                return

            logger.info(f"Loading servers from {filepath}")

            # Parse JSON configuration
            try:
                data = json.loads(content)
                if isinstance(data, dict) and "mcpServers" in data:
                    logger.info(
                        f"Found {len(data['mcpServers'])} servers in configuration")

                    # Handle each server configuration
                    for server_name, config in data["mcpServers"].items():
                        if config.get("command") == "npx" and "args" in config:
                            # Get transport type, default to stdio
                            transport = config.get("transport", "stdio")

                            # Extract args
                            args = config['args']

                            # Create environment
                            env = os.environ.copy()

                            logger.info(
                                f"Loading {server_name}: npx {' '.join(args)} with transport={transport}")

                            # Connect with transport parameter
                            await self._connect_with_params(server_name, "npx", args, env, transport)
                        elif config.get("command") == "python" and "args" in config:
                            # Get transport type, default to stdio
                            transport = config.get("transport", "stdio")

                            # Extract args
                            args = config['args']

                            # Create environment
                            env = os.environ.copy()

                            logger.info(
                                f"Loading {server_name}: python {' '.join(args)} with transport={transport}")

                            # Connect with transport parameter
                            await self._connect_with_params(server_name, "python", args, env, transport)
                        else:
                            logger.warning(
                                f"Invalid configuration for server '{server_name}'")
                else:
                    logger.error(
                        "Invalid JSON format: missing 'mcpServers' key")

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse servers.txt as JSON: {e}")

        except Exception as e:
            logger.error(f"Error loading servers from file: {e}")

    async def load_servers_from_file_session(self, session_id: str, filepath: str = "servers.txt"):
        """Load server configurations from servers.txt file for a specific session"""
        try:
            # Verify session exists
            session = await self.session_manager.get_session(session_id)
            if not session:
                raise ValueError(f"Session {session_id} not found")

            if not os.path.exists(filepath):
                logger.warning(f"Servers file not found: {filepath}")
                return

            with open(filepath, 'r') as f:
                content = f.read().strip()

            if not content:
                logger.info("Servers file is empty")
                return

            logger.info(
                f"Loading servers from {filepath} for session {session_id}")

            # Get exit stack for this session
            if session_id not in self.session_exit_stacks:
                self.session_exit_stacks[session_id] = AsyncExitStack()
            exit_stack = self.session_exit_stacks[session_id]

            # Parse JSON configuration
            try:
                data = json.loads(content)
                if isinstance(data, dict) and "mcpServers" in data:
                    logger.info(
                        f"Found {len(data['mcpServers'])} servers in configuration for session {session_id}")

                    # Handle each server configuration
                    for server_name, config in data["mcpServers"].items():
                        if config.get("command") == "npx" and "args" in config:
                            # Get transport type, default to stdio
                            transport = config.get("transport", "stdio")

                            # Extract args
                            args = config['args']

                            # Create environment
                            env = os.environ.copy()

                            logger.info(
                                f"Loading {server_name} for session {session_id}: npx {' '.join(args)} with transport={transport}")

                            try:
                                # Connect with transport parameter for this session
                                await self._connect_with_params_session(session_id, server_name, "npx", args, env, transport, exit_stack)
                            except Exception as e:
                                logger.error(
                                    f"Failed to load server '{server_name}' for session {session_id}: {e}")
                        elif config.get("command") == "python" and "args" in config:
                            # Get transport type, default to stdio
                            transport = config.get("transport", "stdio")

                            # Extract args
                            args = config['args']

                            # Create environment
                            env = os.environ.copy()

                            logger.info(
                                f"Loading {server_name} for session {session_id}: python {' '.join(args)} with transport={transport}")

                            try:
                                # Connect with transport parameter for this session
                                await self._connect_with_params_session(session_id, server_name, "python", args, env, transport, exit_stack)
                            except Exception as e:
                                logger.error(
                                    f"Failed to load server '{server_name}' for session {session_id}: {e}")
                        else:
                            logger.warning(
                                f"Invalid configuration for server '{server_name}' in session {session_id}")
                else:
                    logger.error(
                        "Invalid JSON format: missing 'mcpServers' key")

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse servers.txt as JSON: {e}")

        except Exception as e:
            logger.error(
                f"Error loading servers from file for session {session_id}: {e}")

    def setup_http_routes(self):
        """Set up HTTP API routes"""
        self.app = web.Application()

        # Set up CORS
        cors = aiohttp_cors.setup(self.app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods="*"
            )
        })

        # Add routes
        self.app.router.add_post('/mcp/search', self.handle_http_search)
        self.app.router.add_post('/mcp/execute', self.handle_http_tool_call)
        self.app.router.add_get('/mcp/tools', self.handle_http_list_tools)
        self.app.router.add_get('/mcp/servers', self.handle_http_list_servers)
        self.app.router.add_get('/health', self.handle_http_health)
        self.app.router.add_post(
            '/mcp/servers/add', self.handle_http_add_servers)
        self.app.router.add_post(
            '/mcp/servers/remove', self.handle_http_remove_server)

        # Session management routes
        self.app.router.add_post(
            '/sessions/create', self.handle_http_create_session)
        self.app.router.add_get(
            '/sessions/{session_id}', self.handle_http_get_session)
        self.app.router.add_get('/sessions', self.handle_http_list_sessions)
        self.app.router.add_delete(
            '/sessions/{session_id}', self.handle_http_delete_session)

        # Configure CORS on all routes
        for route in list(self.app.router.routes()):
            cors.add(route)

        logger.info(f"HTTP API configured on port {self.http_port}")

    async def handle_http_health(self, request):
        """Health check endpoint"""
        return web.json_response({
            "status": "healthy",
            "connected_servers": len(self.connections),
            "total_tools": sum(len(conn.tools) for conn in self.connections.values()),
            "timestamp": str(datetime.now())
        })

    async def handle_http_list_servers(self, request):
        """List all connected MCP servers"""
        # Check for session header
        session_id = request.headers.get("X-Session-ID")

        if session_id:
            # List servers for specific session
            session_connections = self.session_connections.get(session_id, {})
            servers = {}
            for server_uuid, conn in session_connections.items():
                servers[server_uuid] = {
                    "name": conn.name,
                    "transport": conn.transport,
                    "tool_count": len(conn.tools),
                    "tools": [tool["name"] for tool in conn.tools]
                }
            return web.json_response({"servers": servers, "session_id": session_id})
        else:
            # List global servers
            servers = {}
            for server_uuid, conn in self.connections.items():
                servers[server_uuid] = {
                    "name": conn.name,
                    "transport": conn.transport,
                    "tool_count": len(conn.tools),
                    "tools": [tool["name"] for tool in conn.tools]
                }
            return web.json_response({"servers": servers})

    async def handle_http_list_tools(self, request):
        """List all available tools from all servers"""
        # Check for session header
        session_id = request.headers.get("X-Session-ID")

        if session_id:
            # List tools for specific session
            session_connections = self.session_connections.get(session_id, {})
            tools = []
            for server_uuid, connection in session_connections.items():
                for tool in connection.tools:
                    tools.append({
                        "mcp_server": connection.name,
                        "mcp_server_uuid": server_uuid,
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "inputSchema": tool.get("inputSchema", {})
                    })
            return web.json_response({"tools": tools, "total": len(tools), "session_id": session_id})
        else:
            # List global tools
            tools = []
            for server_uuid, connection in self.connections.items():
                for tool in connection.tools:
                    tools.append({
                        "mcp_server": connection.name,
                        "mcp_server_uuid": server_uuid,
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "inputSchema": tool.get("inputSchema", {})
                    })
            return web.json_response({"tools": tools, "total": len(tools)})

    async def handle_http_search(self, request):
        """Handle HTTP search requests"""
        try:
            data = await request.json()
            query = data.get("query", "")

            if not query:
                return web.json_response({"error": "Missing query parameter"}, status=400)

            # Check for session header
            session_id = request.headers.get("X-Session-ID")

            # Create a message structure similar to Redis messages
            message = {
                "request_id": f"http_{int(time.time() * 1000)}",
                "action": "mcp_search",
                "payload": {"query": query}
            }

            # Process using session-aware or global logic
            if session_id:
                result = await self.process_message_session(session_id, message)
            else:
                result = await self.process_message(message)

            # Parse the result if it's a JSON string
            if isinstance(result.get("result"), str):
                try:
                    result["result"] = json.loads(result["result"])
                except json.JSONDecodeError:
                    pass

            return web.json_response(result)

        except Exception as e:
            logger.error(f"Error in HTTP search handler: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def handle_http_tool_call(self, request):
        """Handle HTTP tool call requests"""
        try:
            data = await request.json()
            tool_name = data.get("tool_name", "")
            args = data.get("args", {})

            if not tool_name:
                return web.json_response({"error": "Missing tool_name parameter"}, status=400)

            # Check for session header
            session_id = request.headers.get("X-Session-ID")

            # Create a message structure similar to Redis messages
            message = {
                "request_id": f"http_{int(time.time() * 1000)}",
                "action": "mcp_tool_call",
                "payload": {
                    "tool_name": tool_name,
                    "args": args
                }
            }

            # Process using session-aware or global logic
            if session_id:
                result = await self.process_message_session(session_id, message)
            else:
                result = await self.process_message(message)

            # Parse the result if it's a JSON string
            if isinstance(result.get("result"), str):
                try:
                    result["result"] = json.loads(result["result"])
                except json.JSONDecodeError:
                    pass

            return web.json_response(result)

        except Exception as e:
            logger.error(f"Error in HTTP tool call handler: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def handle_http_add_servers(self, request):
        """Handle HTTP request to add new MCP servers from a config file"""
        try:
            # Check for session header first
            session_id = request.headers.get("X-Session-ID")

            data = await request.json()

            # Get the configuration data
            config_data = data.get("config_data")
            if not config_data:
                return web.json_response({"error": "Missing config_data parameter"}, status=400)

            # If session ID provided, use session-specific handler
            if session_id:
                return await self._handle_http_add_servers_session(request, session_id, data)

            # Parse the configuration
            try:
                servers_config = json.loads(config_data)
                if not isinstance(servers_config, dict) or "mcpServers" not in servers_config:
                    return web.json_response({"error": "Invalid configuration format: missing 'mcpServers' key"}, status=400)
            except json.JSONDecodeError as e:
                return web.json_response({"error": f"Failed to parse configuration: {e}"}, status=400)

            # Track added servers
            added_servers = []
            failed_servers = []

            # Process each server configuration
            for server_name, config in servers_config["mcpServers"].items():
                # Skip if server with same name already exists
                existing_server = any(
                    conn.name == server_name for conn in self.connections.values())
                if existing_server:
                    failed_servers.append({
                        "name": server_name,
                        "error": "Server with this name already connected"
                    })
                    continue

                try:
                    if config.get("command") == "npx" and "args" in config:
                        # Get transport type, default to stdio
                        transport = config.get("transport", "stdio")
                        args = config['args']
                        env = os.environ.copy()

                        logger.info(
                            f"Adding server {server_name}: npx {' '.join(args)} with transport={transport}")

                        # Connect with transport parameter
                        await self._connect_with_params(server_name, "npx", args, env, transport)

                        # Find the UUID for the newly connected server
                        server_uuid = None
                        for uuid, conn in self.connections.items():
                            if conn.name == server_name:
                                server_uuid = uuid
                                break

                        if server_uuid:
                            added_servers.append({
                                "name": server_name,
                                "uuid": server_uuid,
                                "command": f"npx {' '.join(args)}",
                                "transport": transport,
                                "tools_count": len(self.connections[server_uuid].tools)
                            })
                        else:
                            # This shouldn't happen but handle it gracefully
                            added_servers.append({
                                "name": server_name,
                                "command": f"npx {' '.join(args)}",
                                "transport": transport,
                                "tools_count": 0
                            })
                    else:
                        failed_servers.append({
                            "name": server_name,
                            "error": "Invalid configuration format"
                        })
                except Exception as e:
                    failed_servers.append({
                        "name": server_name,
                        "error": str(e)
                    })

            return web.json_response({
                "success": True,
                "added_servers": added_servers,
                "failed_servers": failed_servers,
                "total_servers": len(self.connections)
            })

        except Exception as e:
            logger.error(f"Error in HTTP add servers handler: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def handle_http_remove_server(self, request):
        """Handle HTTP request to remove an MCP server"""
        try:
            data = await request.json()
            server_identifier = data.get(
                "server_uuid") or data.get("server_name")

            if not server_identifier:
                return web.json_response({"error": "Missing server_uuid or server_name parameter"}, status=400)

            # Check for session header
            session_id = request.headers.get("X-Session-ID")

            if session_id:
                # Remove from specific session
                success = await self.remove_server_from_session(session_id, server_identifier)
                if success:
                    session_connections = self.session_connections.get(
                        session_id, {})
                    return web.json_response({
                        "success": True,
                        "removed_server": server_identifier,
                        "session_id": session_id,
                        "remaining_servers_in_session": len(session_connections)
                    })
                else:
                    return web.json_response({"error": f"Server '{server_identifier}' not found in session {session_id}"}, status=404)

            # Find server by UUID or name
            server_uuid = None
            connection = None

            # First try to find by UUID
            if server_identifier in self.connections:
                server_uuid = server_identifier
                connection = self.connections[server_uuid]
            else:
                # Try to find by name
                for uuid, conn in self.connections.items():
                    if conn.name == server_identifier:
                        server_uuid = uuid
                        connection = conn
                        break

            if not server_uuid:
                return web.json_response({"error": f"Server '{server_identifier}' not found"}, status=404)

            # Get server info before removing
            removed_tools_count = len(connection.tools)
            removed_server_name = connection.name

            # Remove the server
            try:
                # Close the session if it exists
                if connection.session:
                    # Session will be closed by exit_stack
                    pass

                # Remove from connections
                del self.connections[server_uuid]

                # Update tool inventory in Redis
                await self.update_tool_inventory()

                logger.info(
                    f"Successfully removed server '{removed_server_name}' (UUID: {server_uuid})")

                return web.json_response({
                    "success": True,
                    "removed_server": removed_server_name,
                    "removed_server_uuid": server_uuid,
                    "removed_tools_count": removed_tools_count,
                    "remaining_servers": {uuid: conn.name for uuid, conn in self.connections.items()},
                    "total_servers": len(self.connections)
                })

            except Exception as e:
                logger.error(
                    f"Error removing server '{removed_server_name}': {e}")
                return web.json_response({"error": f"Failed to remove server: {e}"}, status=500)

        except Exception as e:
            logger.error(f"Error in HTTP remove server handler: {e}")
            return web.json_response({"error": str(e)}, status=500)

    # Session-aware methods
    async def connect_to_server_session(self, session_id: str, server_input: str) -> None:
        """Connect to an MCP server within a specific session context"""
        # Verify session exists
        session = await self.session_manager.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        # Parse the input to determine connection type
        parts = server_input.strip().split()
        if not parts:
            logger.error("Empty server input")
            return

        # Get exit stack for this session
        if session_id not in self.session_exit_stacks:
            self.session_exit_stacks[session_id] = AsyncExitStack()
        exit_stack = self.session_exit_stacks[session_id]

        # Handle npx commands
        if parts[0] == "npx":
            # Extract package name and args from npx command
            if len(parts) < 2:
                logger.error("Invalid npx command: missing package name")
                return

            package_name = parts[1]
            additional_args = parts[2:] if len(parts) > 2 else []
            server_name = package_name.split('/')[-1].replace('@', '')
            args = ["-y", package_name] + additional_args
            env = os.environ.copy()

            await self._connect_with_params_session(session_id, server_name, "npx", args, env, "stdio", exit_stack)

        # Handle executable paths (.py or .js files)
        elif parts[0].endswith('.py') or parts[0].endswith('.js'):
            server_script_path = parts[0]

            # Validate file exists
            if not os.path.exists(server_script_path):
                logger.error(f"Server script not found: {server_script_path}")
                return

            is_python = server_script_path.endswith('.py')
            command = "python" if is_python else "node"
            server_name = os.path.basename(
                server_script_path).rsplit('.', 1)[0]
            env = os.environ.copy()

            await self._connect_with_params_session(session_id, server_name, command, [server_script_path], env, "stdio", exit_stack)

        else:
            logger.error(f"Unsupported server input format: {server_input}")
            logger.info(
                "Expected formats: 'npx <package-name> [args...]' or '/path/to/server.py|.js'")

    async def get_session_connections(self, session_id: str) -> Dict[str, ServerConnection]:
        """Get all server connections for a specific session"""
        return self.session_connections.get(session_id, {})

    async def remove_server_from_session(self, session_id: str, server_identifier: str) -> bool:
        """Remove a server connection from a specific session"""
        try:
            # Check if session exists
            session = await self.session_manager.get_session(session_id)
            if not session:
                logger.error(f"Session {session_id} not found")
                return False

            # Check if server exists in session
            if session_id not in self.session_connections:
                logger.error(f"No connections found for session {session_id}")
                return False

            # Find server by UUID or name
            server_uuid = None
            if server_identifier in self.session_connections[session_id]:
                server_uuid = server_identifier
            else:
                # Try to find by name
                for uuid, conn in self.session_connections[session_id].items():
                    if conn.name == server_identifier:
                        server_uuid = uuid
                        break

            if not server_uuid:
                logger.error(
                    f"Server {server_identifier} not found in session {session_id}")
                return False

            # Remove the connection
            connection = self.session_connections[session_id][server_uuid]
            del self.session_connections[session_id][server_uuid]

            # Update session manager
            await self.session_manager.remove_server_connection(session_id, server_uuid)

            # Update tool inventory for session
            await self.update_tool_inventory_session(session_id)

            logger.info(
                f"Removed server '{connection.name}' (UUID: {server_uuid}) from session {session_id}")
            return True

        except Exception as e:
            logger.error(f"Error removing server from session: {e}")
            return False

    def search_tools_session(self, session_id: str, query: str) -> list:
        """Search for tools within a specific session's context"""
        # Check if AI client is available
        if not hasattr(self, 'ai_client') or self.ai_client is None:
            logger.error("No AI client available for tool routing")
            return []

        # Get session connections
        session_connections = self.session_connections.get(session_id, {})
        if not session_connections:
            logger.warning(f"No connections found for session {session_id}")
            return []

        # Prepare tool inventory for the LLM
        tool_inventory = []
        for server_uuid, connection in session_connections.items():
            for tool in connection.tools:
                tool_inventory.append({
                    "mcp_server": connection.name,
                    "mcp_server_uuid": server_uuid,
                    "tool_name": tool["name"],
                    "description": tool.get("description", ""),
                    "input_schema": tool.get("inputSchema", {})
                })

        if not tool_inventory:
            logger.warning(f"No tools available in session {session_id}")
            return []

        # Use the same routing logic as global search
        return self._perform_tool_search(query, tool_inventory)

    def fetch_tool_specs_session(self, session_id: str, tool_matches: list) -> list:
        """Fetch tool specs within a specific session's context"""
        results = []
        session_connections = self.session_connections.get(session_id, {})

        for match in tool_matches:
            server_uuid = match.get("mcp_server_uuid", match.get("mcp_server"))
            tool_name = match["tool_name"]

            # Check if server exists in session
            if server_uuid in session_connections:
                connection = session_connections[server_uuid]

                # Find the tool in this server's tools
                for tool in connection.tools:
                    if tool["name"] == tool_name:
                        # Format the tool spec for the agent
                        tool_spec = {
                            "mcp_server": connection.name,
                            "mcp_server_uuid": server_uuid,
                            "tool_name": tool["name"],
                            "description": tool.get("description", ""),
                            "input_schema": tool.get("inputSchema", {}),
                            "match_score": match.get("match_score", 1.0),
                            "reasoning": match.get("reasoning", "")
                        }
                        results.append(tool_spec)
                        break

        return results

    async def process_message_session(self, session_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process a message within a specific session's context"""
        # Verify session exists
        session = await self.session_manager.get_session(session_id)
        if not session:
            return {
                "request_id": message.get("request_id"),
                "status": "error",
                "timestamp": time.time(),
                "result": json.dumps({"error": f"Session {session_id} not found"})
            }

        request_id = message.get("request_id")
        action = message.get("action")
        payload = message.get("payload", {})

        # Parse payload if it's a JSON string
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse payload JSON: {payload}")

        logger.info(
            f"Processing message in session {session_id}: {request_id} - Action: {action}")

        # Base response
        response = {
            "request_id": request_id,
            "status": "completed",
            "timestamp": time.time(),
        }

        # Handle specific MCP actions
        if action == "mcp_search":
            query = payload.get("query", "")
            logger.info(
                f"Searching for tools in session {session_id} matching query: '{query}'")
            relevant_tools = self.search_tools_session(session_id, query)
            relevant_tool_specs = self.fetch_tool_specs_session(
                session_id, relevant_tools)
            response["result"] = json.dumps(relevant_tool_specs)

        elif action == "mcp_tool_call":
            tool_name = payload.get("tool_name", "")
            args = payload.get("args", {})

            logger.info(
                f"MCP Tool Call request in session {session_id}: tool='{tool_name}'")

            # Find which server has this tool in this session
            server_uuid = self._get_service_for_tool_session(
                session_id, tool_name)
            if not server_uuid:
                error_msg = f"Tool '{tool_name}' not found in session {session_id}"
                logger.error(error_msg)
                response["status"] = "error"
                response["result"] = json.dumps({"error": error_msg})
                return response

            # Get the connection and session
            connection = self.session_connections[session_id][server_uuid]
            session = connection.session

            # Execute the tool call
            try:
                logger.info(
                    f"Executing tool '{tool_name}' on server '{connection.name}' in session {session_id}")
                tool_result = await session.call_tool(tool_name, args)

                # Extract content from tool result
                if hasattr(tool_result, 'content'):
                    if isinstance(tool_result.content, list):
                        content_texts = []
                        for item in tool_result.content:
                            if hasattr(item, 'text'):
                                content_texts.append(item.text)
                            else:
                                content_texts.append(str(item))
                        result_content = "\n".join(content_texts)
                    else:
                        result_content = str(tool_result.content)
                else:
                    result_content = str(tool_result)

                response["result"] = json.dumps({
                    "success": True,
                    "server": connection.name,
                    "server_uuid": server_uuid,
                    "tool": tool_name,
                    "content": result_content
                })

            except Exception as e:
                error_msg = f"Failed to execute tool '{tool_name}': {str(e)}"
                logger.error(error_msg, exc_info=True)
                response["status"] = "error"
                response["result"] = json.dumps({"error": error_msg})

        else:
            # Default echo for other actions
            response["result"] = f"ECHO: {json.dumps(message)}"

        return response

    async def update_tool_inventory_session(self, session_id: str) -> None:
        """Update tool inventory for a specific session"""
        if not self.redis_client:
            return

        try:
            # Build tool inventory for this session
            tool_inventory = []
            session_connections = self.session_connections.get(session_id, {})

            for server_uuid, connection in session_connections.items():
                for tool in connection.tools:
                    tool_inventory.append({
                        "mcp_server": connection.name,
                        "mcp_server_uuid": server_uuid,
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "inputSchema": tool.get("inputSchema", {})
                    })

            # Store in Redis with session-specific key
            await self.redis_client.set(
                f"gateway:session:{session_id}:tool_inventory",
                json.dumps(tool_inventory),
                ex=3600  # Expire after 1 hour
            )

            # Also update session status
            servers_info = {
                server_uuid: connection.name
                for server_uuid, connection in session_connections.items()
            }
            await self.redis_client.set(
                f"gateway:session:{session_id}:status",
                json.dumps({
                    "status": "active",
                    "connected_servers": servers_info,
                    "total_tools": len(tool_inventory),
                    "timestamp": str(datetime.now())
                }),
                ex=60  # Expire after 1 minute (heartbeat)
            )

            logger.info(
                f"Updated tool inventory for session {session_id}: {len(tool_inventory)} tools from {len(session_connections)} servers")

        except Exception as e:
            logger.error(
                f"Failed to update tool inventory for session {session_id}: {e}")

    async def _connect_with_params_session(self, session_id: str, name: str, command: str, args: List[str], env: Dict[str, str], transport: str, exit_stack: AsyncExitStack):
        """Internal method to connect with specific parameters within a session context"""
        try:
            logger.info(
                f"Connecting to MCP server '{name}' in session {session_id} with command: {command} {args} using transport: {transport}")

            # Handle different transport types
            if transport == "stdio":
                # Set up server parameters for stdio
                server_params = StdioServerParameters(
                    command=command,
                    args=args,
                    env=env
                )

                # Create and connect to the server
                stdio_transport = await exit_stack.enter_async_context(
                    stdio_client(server_params)
                )
                stdio, write = stdio_transport
                session = await exit_stack.enter_async_context(
                    ClientSession(stdio, write)
                )

            elif transport == "http":
                # For HTTP transport, the last arg should be the URL
                if args and args[-1].startswith("http"):
                    url = args[-1]
                    logger.info(f"Connecting to HTTP MCP server at: {url}")

                    # Create HTTP transport
                    streams_context = await exit_stack.enter_async_context(
                        streamablehttp_client(url=url, headers={})
                    )
                    read_stream, write_stream, _ = streams_context
                    session = await exit_stack.enter_async_context(
                        ClientSession(read_stream, write_stream)
                    )
                else:
                    raise ValueError(f"Invalid HTTP URL in args: {args}")

            elif transport == "sse":
                # For SSE transport, the last arg should be the URL
                if args and args[-1].startswith("http"):
                    url = args[-1]
                    logger.info(f"Connecting to SSE MCP server at: {url}")

                    # Create SSE transport
                    streams_context = await exit_stack.enter_async_context(
                        sse_client(url=url, headers={})
                    )
                    # SSE client returns only 2 values (read_stream, write_stream)
                    read_stream, write_stream = streams_context
                    session = await exit_stack.enter_async_context(
                        ClientSession(read_stream, write_stream)
                    )
                else:
                    raise ValueError(f"Invalid SSE URL in args: {args}")

            else:
                raise ValueError(f"Unsupported transport type: {transport}")

            # Initialize the session
            await session.initialize()

            # List available tools
            response = await session.list_tools()
            tools = []
            for tool in response.tools:
                tool_info = {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.inputSchema
                }
                tools.append(tool_info)

            # Generate UUID for this server
            server_uuid = str(uuid.uuid4())

            # Store connection with transport type in session context
            if session_id not in self.session_connections:
                self.session_connections[session_id] = {}

            self.session_connections[session_id][server_uuid] = ServerConnection(
                uuid=server_uuid,
                name=name,
                session=session,
                tools=tools,
                transport=transport
            )

            # Update session manager
            connection_data = {
                "name": name,
                "transport": transport,
                "tools_count": len(tools),
                "connected_at": str(datetime.now())
            }
            await self.session_manager.add_server_connection(session_id, server_uuid, connection_data)

            logger.info(
                f"Connected to '{name}' server (UUID: {server_uuid}) in session {session_id} with {len(tools)} tools: {[t['name'] for t in tools]}")

            # Update tool inventory for this session
            await self.update_tool_inventory_session(session_id)

        except Exception as e:
            logger.error(
                f"Failed to connect to MCP server '{name}' in session {session_id}: {e}")
            raise

    def _get_service_for_tool_session(self, session_id: str, tool_name: str) -> Optional[str]:
        """Determine which service a tool belongs to within a session (returns UUID)"""
        session_connections = self.session_connections.get(session_id, {})

        for server_uuid, connection in session_connections.items():
            for tool in connection.tools:
                if tool["name"] == tool_name:
                    logger.info(
                        f"Tool '{tool_name}' found in server '{connection.name}' (UUID: {server_uuid}) in session {session_id}")
                    return server_uuid

        logger.warning(
            f"Tool '{tool_name}' not found in any server in session {session_id}")
        return None

    # Session management HTTP endpoints
    async def handle_http_create_session(self, request):
        """Create a new session"""
        try:
            data = await request.json() if request.content_length else {}
            metadata = data.get("metadata", {})

            # Create new session
            session = await self.session_manager.create_session(metadata)

            # Automatically load servers from servers.txt for the new session
            try:
                await self.load_servers_from_file_session(session.session_id)
                logger.info(
                    f"Automatically loaded servers from servers.txt for session {session.session_id}")
            except Exception as e:
                logger.warning(
                    f"Failed to automatically load servers for session {session.session_id}: {e}")

            # Get session connections count after loading servers
            session_connections = self.session_connections.get(
                session.session_id, {})

            return web.json_response({
                "success": True,
                "session_id": session.session_id,
                "created_at": session.created_at.isoformat(),
                "expires_at": session.expires_at.isoformat() if session.expires_at else None,
                "metadata": session.metadata,
                "servers_loaded": len(session_connections),
                "servers": {uuid: conn.name for uuid, conn in session_connections.items()}
            })

        except Exception as e:
            logger.error(f"Error creating session: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def handle_http_get_session(self, request):
        """Get session details"""
        try:
            session_id = request.match_info["session_id"]

            # Get session
            session = await self.session_manager.get_session(session_id)
            if not session:
                return web.json_response({"error": f"Session {session_id} not found"}, status=404)

            # Get session connections
            session_connections = self.session_connections.get(session_id, {})
            connections_info = []
            for server_uuid, connection in session_connections.items():
                connections_info.append({
                    "uuid": server_uuid,
                    "name": connection.name,
                    "transport": connection.transport,
                    "tools_count": len(connection.tools),
                    "tools": [tool["name"] for tool in connection.tools]
                })

            return web.json_response({
                "session_id": session.session_id,
                "created_at": session.created_at.isoformat(),
                "last_accessed": session.last_accessed.isoformat(),
                "expires_at": session.expires_at.isoformat() if session.expires_at else None,
                "active": session.active,
                "metadata": session.metadata,
                "connections": connections_info,
                "total_tools": sum(conn["tools_count"] for conn in connections_info)
            })

        except Exception as e:
            logger.error(f"Error getting session: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def handle_http_list_sessions(self, request):
        """List all active sessions"""
        try:
            # Get all active sessions
            sessions = await self.session_manager.list_active_sessions()

            sessions_info = []
            for session in sessions:
                # Get session connections
                session_connections = self.session_connections.get(
                    session.session_id, {})
                connections_count = len(session_connections)
                tools_count = sum(len(conn.tools)
                                  for conn in session_connections.values())

                sessions_info.append({
                    "session_id": session.session_id,
                    "created_at": session.created_at.isoformat(),
                    "last_accessed": session.last_accessed.isoformat(),
                    "expires_at": session.expires_at.isoformat() if session.expires_at else None,
                    "connections_count": connections_count,
                    "tools_count": tools_count,
                    "metadata": session.metadata
                })

            return web.json_response({
                "sessions": sessions_info,
                "total": len(sessions_info)
            })

        except Exception as e:
            logger.error(f"Error listing sessions: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def handle_http_delete_session(self, request):
        """Delete a session"""
        try:
            session_id = request.match_info["session_id"]

            # Check if session exists
            session = await self.session_manager.get_session(session_id)
            if not session:
                return web.json_response({"error": f"Session {session_id} not found"}, status=404)

            # Clean up session connections and their exit stacks (which will close server processes)
            if session_id in self.session_connections:
                # Log the servers being cleaned up
                for server_uuid, conn in self.session_connections[session_id].items():
                    logger.info(
                        f"Cleaning up server '{conn.name}' (UUID: {server_uuid}) from session {session_id}")
                del self.session_connections[session_id]

            # Clean up session exit stack - this will close all server processes
            if session_id in self.session_exit_stacks:
                logger.info(
                    f"Closing all server processes for session {session_id}")
                await self.session_exit_stacks[session_id].aclose()
                del self.session_exit_stacks[session_id]

            # Delete session from manager
            success = await self.session_manager.delete_session(session_id)

            if success:
                return web.json_response({
                    "success": True,
                    "deleted_session_id": session_id
                })
            else:
                return web.json_response({"error": "Failed to delete session"}, status=500)

        except Exception as e:
            logger.error(f"Error deleting session: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def _handle_http_add_servers_session(self, request, session_id: str, data: Dict[str, Any]):
        """Handle adding servers to a specific session"""
        try:
            # Verify session exists
            session = await self.session_manager.get_session(session_id)
            if not session:
                return web.json_response({"error": f"Session {session_id} not found"}, status=404)

            # Get the configuration data
            config_data = data.get("config_data")

            # Parse the configuration
            try:
                servers_config = json.loads(config_data)
                if not isinstance(servers_config, dict) or "mcpServers" not in servers_config:
                    return web.json_response({"error": "Invalid configuration format: missing 'mcpServers' key"}, status=400)
            except json.JSONDecodeError as e:
                return web.json_response({"error": f"Failed to parse configuration: {e}"}, status=400)

            # Get exit stack for this session
            if session_id not in self.session_exit_stacks:
                self.session_exit_stacks[session_id] = AsyncExitStack()
            exit_stack = self.session_exit_stacks[session_id]

            # Track added servers
            added_servers = []
            failed_servers = []

            # Process each server configuration
            for server_name, config in servers_config["mcpServers"].items():
                # Check if server with same name already exists in this session
                session_connections = self.session_connections.get(
                    session_id, {})
                existing_server = any(
                    conn.name == server_name for conn in session_connections.values())
                if existing_server:
                    failed_servers.append({
                        "name": server_name,
                        "error": "Server with this name already connected in this session"
                    })
                    continue

                try:
                    if config.get("command") == "npx" and "args" in config:
                        # Get transport type, default to stdio
                        transport = config.get("transport", "stdio")
                        args = config['args']
                        env = os.environ.copy()

                        logger.info(
                            f"Adding server {server_name} to session {session_id}: npx {' '.join(args)} with transport={transport}")

                        # Connect with transport parameter
                        await self._connect_with_params_session(session_id, server_name, "npx", args, env, transport, exit_stack)

                        # Refresh session_connections after adding server
                        session_connections = self.session_connections.get(
                            session_id, {})

                        # Find the UUID for the newly connected server
                        server_uuid = None
                        for uuid, conn in session_connections.items():
                            if conn.name == server_name:
                                server_uuid = uuid
                                break

                        if server_uuid:
                            added_servers.append({
                                "name": server_name,
                                "uuid": server_uuid,
                                "command": f"npx {' '.join(args)}",
                                "transport": transport,
                                "tools_count": len(session_connections[server_uuid].tools) if server_uuid in session_connections else 0
                            })
                        else:
                            # This shouldn't happen but handle it gracefully
                            added_servers.append({
                                "name": server_name,
                                "command": f"npx {' '.join(args)}",
                                "transport": transport,
                                "tools_count": 0
                            })
                    else:
                        failed_servers.append({
                            "name": server_name,
                            "error": "Invalid configuration format"
                        })
                except Exception as e:
                    failed_servers.append({
                        "name": server_name,
                        "error": str(e)
                    })

            # Get updated session connections count
            session_connections = self.session_connections.get(session_id, {})

            return web.json_response({
                "success": True,
                "session_id": session_id,
                "added_servers": added_servers,
                "failed_servers": failed_servers,
                "total_servers_in_session": len(session_connections)
            })

        except Exception as e:
            logger.error(f"Error adding servers to session: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def _cleanup_session_resources(self, session_id: str) -> None:
        """Cleanup resources for an expired session"""
        logger.info(f"Cleaning up resources for expired session {session_id}")

        # Clean up session connections
        if session_id in self.session_connections:
            for server_uuid, conn in self.session_connections[session_id].items():
                logger.info(
                    f"Cleaning up server '{conn.name}' (UUID: {server_uuid}) from expired session {session_id}")
            del self.session_connections[session_id]

        # Clean up session exit stack - this will close all server processes
        if session_id in self.session_exit_stacks:
            logger.info(
                f"Closing all server processes for expired session {session_id}")
            try:
                await self.session_exit_stacks[session_id].aclose()
            except Exception as e:
                logger.error(
                    f"Error closing exit stack for session {session_id}: {e}")
            del self.session_exit_stacks[session_id]

    async def run(self):
        """Main run loop"""
        self.running = True
        logger.info("MCP Gateway started")

        # Auto-load servers from file
        await self.load_servers_from_file()

        # Start session cleanup task
        await self.session_manager.start_cleanup_task(interval_seconds=300)

        # Set up HTTP routes
        self.setup_http_routes()

        # Create HTTP server runner
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', self.http_port)

        try:
            # Start HTTP server
            await site.start()
            logger.info(f"HTTP API server started on port {self.http_port}")

            # Run terminal input handler, queue polling, and HTTP server concurrently
            await asyncio.gather(
                self.handle_terminal_input(),
                self.poll_queues(),
                return_exceptions=True
            )
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        finally:
            self.running = False
            await runner.cleanup()


def signal_handler(sig, frame):
    """Handle shutdown signals"""
    logger.info("Shutdown signal received")
    sys.exit(0)


async def main():
    """Main entry point"""
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Create and run MCP client
    client = MCPClient()

    try:
        await client.connect()
        await client.run()
    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())

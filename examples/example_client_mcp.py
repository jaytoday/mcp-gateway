#!/usr/bin/env python3
"""
Example MCP client that connects directly to the MCP Gateway server and provides an interactive chat interface.

This client demonstrates how to:
1. Connect directly to the MCP Gateway via MCP protocol (not HTTP)
2. Use the gateway's search_tools and execute_tool functions
3. Implement an LLM loop with linear chat memory
4. Handle tool search -> tool execute -> summary workflow

The main difference from example_client_http is that this client uses native MCP protocol
to communicate with the gateway instead of HTTP endpoints.
"""

import asyncio
import json
import time
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from contextlib import AsyncExitStack
from dotenv import load_dotenv

# MCP imports
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

# Import both AI clients
try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

load_dotenv()


@dataclass
class ChatMessage:
    role: str
    content: str
    timestamp: float


class MCPGatewayClient:
    def __init__(self, gateway_mcp_url: str = "http://localhost:3000/mcp",
                 ai_provider: str = "auto",
                 anthropic_api_key: Optional[str] = None,
                 openai_api_key: Optional[str] = None):
        self.gateway_mcp_url = gateway_mcp_url
        self.chat_history: List[ChatMessage] = []
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

        # Initialize AI client based on provider preference and availability
        self.ai_client = None
        self.ai_provider = None

        if ai_provider == "auto":
            # Auto-detect based on available API keys
            anthropic_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
            openai_key = openai_api_key or os.getenv("OPENAI_API_KEY")

            if anthropic_key and ANTHROPIC_AVAILABLE:
                self.ai_provider = "anthropic"
                self.ai_client = Anthropic(api_key=anthropic_key)
            elif openai_key and OPENAI_AVAILABLE:
                self.ai_provider = "openai"
                self.ai_client = OpenAI(api_key=openai_key)
            else:
                raise ValueError(
                    "No AI API keys found. Please set ANTHROPIC_API_KEY or OPENAI_API_KEY")

        elif ai_provider == "anthropic":
            if not ANTHROPIC_AVAILABLE:
                raise ValueError(
                    "Anthropic not available. Install with: pip install anthropic")
            api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY must be provided")
            self.ai_provider = "anthropic"
            self.ai_client = Anthropic(api_key=api_key)

        elif ai_provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ValueError(
                    "OpenAI not available. Install with: pip install openai")
            api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY must be provided")
            self.ai_provider = "openai"
            self.ai_client = OpenAI(api_key=api_key)

        else:
            raise ValueError(
                "ai_provider must be 'auto', 'anthropic', or 'openai'")

        print(f"✓ {self.ai_provider.title()} client initialized")

    async def connect_to_gateway(self):
        """Connect to the MCP Gateway server"""
        try:
            # Create the transport using streamablehttp_client
            transport = await self.exit_stack.enter_async_context(
                streamablehttp_client(self.gateway_mcp_url)
            )
            self.read_stream, self.write_stream, self.get_session_id = transport
            
            # Create the session
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(self.read_stream, self.write_stream)
            )
            
            # Initialize the session
            await self.session.initialize()
            
            # List available tools (should include search_tools and execute_tool)
            response = await self.session.list_tools()
            tools = response.tools
            tool_names = [tool.name for tool in tools]
            
            print(f"✓ Connected to MCP Gateway at {self.gateway_mcp_url}")
            print(f"✓ Available tools: {tool_names}")
            
            # Verify we have the expected gateway tools
            if "search_tools" not in tool_names or "execute_tool" not in tool_names:
                print("⚠️  Warning: Expected gateway tools (search_tools, execute_tool) not found")
            
        except Exception as e:
            print(f"❌ Failed to connect to MCP Gateway at {self.gateway_mcp_url}: {e}")
            raise

    async def search_tools(self, query: str, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for tools using the gateway's search_tools function"""
        try:
            # Try calling with keyword arguments first
            args = {"query": query}
            if session_id:
                args["session_id"] = session_id
                
            result = await self.session.call_tool("search_tools", args)
            
            # The result content should be a list of tools or JSON string
            if hasattr(result, 'content'):
                content = result.content
                if isinstance(content, str):
                    try:
                        return json.loads(content)
                    except json.JSONDecodeError:
                        return []
                elif isinstance(content, list):
                    return content
            
            return []
        except Exception as e:
            print(f"Error searching tools: {e}")
            return []

    async def execute_tool(self, tool_name: str, args: Dict[str, Any], session_id: Optional[str] = None) -> Optional[str]:
        """Execute a tool using the gateway's execute_tool function"""
        try:
            call_args = {
                "tool_name": tool_name,
                "args": args
            }
            if session_id:
                call_args["session_id"] = session_id
                
            result = await self.session.call_tool("execute_tool", call_args)
            
            # Return the result content
            if hasattr(result, 'content'):
                return result.content
            
            return str(result)
        except Exception as e:
            print(f"Error executing tool {tool_name}: {e}")
            return f"Error executing tool {tool_name}: {e}"

    def add_message(self, role: str, content: str):
        """Add a message to chat history"""
        self.chat_history.append(ChatMessage(
            role=role,
            content=content,
            timestamp=time.time()
        ))

    def get_chat_history_string(self) -> str:
        """Get linear chat memory as a string"""
        history_parts = []
        for msg in self.chat_history[-20:]:  # Keep last 20 messages
            history_parts.append(f"{msg.role.upper()}: {msg.content}")
        return "\n".join(history_parts)

    def _call_ai_model(self, prompt: str, max_tokens: int = 1000) -> str:
        """Call the configured AI model with a prompt"""
        if self.ai_provider == "anthropic":
            response = self.ai_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()

        elif self.ai_provider == "openai":
            response = self.ai_client.chat.completions.create(
                model="gpt-4o",
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content.strip()

        else:
            raise ValueError(f"Unsupported AI provider: {self.ai_provider}")

    def should_continue_acting(self, response_text: str) -> bool:
        """
        Determine if the agent should continue acting or wait for user input.
        Uses an LLM to decide: returns True if the agent should continue, False otherwise.
        """
        prompt = f"""Given the following assistant response, should the agent continue acting autonomously (e.g., searching, executing tools, or taking the next step) or wait for further user input? 
        Respond with only 'true' (continue acting) or 'false' (wait for user input).
        
        Assistant response:
        \"\"\"{response_text}\"\"\"
        Answer:"""

        result = self._call_ai_model(prompt, max_tokens=5).strip().lower()
        if "true" in result:
            return True
        elif "false" in result:
            return False
        else:
            # Fallback: default to not continue
            return False

    async def think_and_respond(self, user_input: str) -> str:
        """
        Main reasoning loop: decide whether to respond directly or search/execute tools
        """
        # Add user message to history
        self.add_message("user", user_input)

        # Get current context
        chat_context = self.get_chat_history_string()

        # First, ask the AI to analyze what to do
        analysis_prompt = f"""
        
        You are a helpful AI assistant with access to gateway tools for discovering and executing MCP (Model Context Protocol) tools.
        Your primary goal is to provide accurate, relevant, and helpful information to users.
        User requests may require you to chain together multiple tool calls, analyze the output, and make more decisions. Your job
        is to be fairly autonomous and accomplish user requests to the best of your ability, but you should ask the user for 
        clarification whenever there is room for multiple decisions to be made.

        Current conversation context:
        {chat_context}

        Guidelines:
        - If the request is general conversation, facts you know, or simple questions, respond directly
        - If the request involves real-time data, external services, file operations, web searches, API calls, or specialized functionality, you'll need tools
        - For tool searches, be specific about what capabilities you need

        Respond with either:
        - "DIRECT: [your direct response]" 
        - "Reasoning, followed by -- SEARCH: [description of tools needed]"
        - "EXECUTE: [tool_name] [arguments]"

        Example 1:
        User: "Hi, what's up?"
        Assistant: "DIRECT: Hi, how can I help you today?"

        Example 2:
        User: "What's the weather in Tokyo?"
        Assistant: "I don't have access to real-time weather data. I can help you search for weather information using the weather tool. \n\n -- SEARCH: weather tools to find weather in Tokyo"

        Example 3:
        User: "I searched for weather tools, and here is a schema I found of a good weather tool"
        Assistant: "I'll use the weather tool to find weather in Tokyo. \n\n -- EXECUTE: weather_tool | {{"city": "Tokyo"}}

        When using tools:
        1. **search_tools**: Use this to discover available MCP tools
           - Just provide the search query as a string
           - Example: "search the web" or "weather tools" or "current time"
           - The search will return matching tools with their server names and exact tool names

        2. **execute_tool**: Use this to execute a specific MCP tool after discovering it
           - Arguments: {{"tool_name": "<exact_name_from_search>", "args": {{...}}}}
           - Use the EXACT tool_name returned from search_tools
           - Example: {{"tool_name": "exa_search", "args": {{"query": "latest AI news"}}}}

        Important workflow:
        - ALWAYS search for tools first using search_tools before trying to execute them
        - Use the exact tool names returned from search results when executing
        - Include appropriate arguments based on the tool's requirements

        Be professional, friendly, and focused on helping users achieve their goals.
"""

        try:
            analysis_text = self._call_ai_model(
                analysis_prompt, max_tokens=1000)

            if "DIRECT:" in analysis_text:
                # Respond directly
                direct_response = analysis_text.split("DIRECT:", 1)[1].strip()
                self.add_message("assistant", direct_response)
                return direct_response

            elif "SEARCH:" in analysis_text:
                # Search for and use tools
                search_query = analysis_text.split("SEARCH:", 1)[1].strip()
                return await self._handle_tool_workflow(search_query, user_input)

            elif "EXECUTE:" in analysis_text:
                # Execute a tool
                execute_part = analysis_text.split("EXECUTE:", 1)[1].strip()
                # Parse tool_name and args
                if "|" in execute_part:
                    tool_name, args_json = execute_part.split("|", 1)
                    try:
                        args = json.loads(args_json.strip())
                        return await self._execute_and_summarize(tool_name.strip(), args, user_input)
                    except json.JSONDecodeError:
                        return await self._execute_and_summarize(tool_name.strip(), {}, user_input)
                else:
                    return await self._execute_and_summarize(execute_part.strip(), {}, user_input)
            else:
                # Fallback to direct response
                self.add_message("assistant", analysis_text)
                return analysis_text

        except Exception as e:
            error_msg = f"Error in reasoning: {e}"
            self.add_message("assistant", error_msg)
            return error_msg

    async def _handle_tool_workflow(self, search_query: str, original_request: str) -> str:
        """Handle the search -> execute -> summarize workflow"""

        print(f"🔍 Searching for tools: {search_query}")

        # Step 1: Search for tools
        tools = await self.search_tools(search_query)

        if not tools:
            response = f"I searched for tools to help with '{search_query}' but didn't find any suitable tools. Let me try to help you directly instead."
            self.add_message("assistant", response)
            return response

        print(f"📋 Found {len(tools)} relevant tools")
        try:
            tools_display = json.dumps(tools, indent=2)
        except (TypeError, ValueError) as e:
            tools_display = str(tools)

        # Step 2: Let the AI decide which tool to use and how
        tool_selection_prompt = f"""I found these tools that might help with the request "{original_request}":

        Available tools:
        {tools_display}

        Original user request: {original_request}
        Search query used: {search_query}

        Please analyze the situation and respond based on these patterns:

        1. **Tool fits and you have enough context**: If a tool is clearly appropriate and you have all the information needed to call it effectively, proceed with execution.
        
        2. **Tool fits but needs user input**: If a tool could work but you need clarification about parameters, preferences, or specifics from the user, explain what additional information you need.
        
        3. **No suitable tool**: If none of the tools are appropriate for this request, explain why and suggest alternatives.

        Respond with either:
        - "EXECUTE: tool_name|{{json_arguments}}" (e.g., "EXECUTE: web_search|{{"query": "Python tutorials"}}")
        - "CLARIFY: [explanation of what user input is needed before proceeding]"
        - "NONE: [explanation of why no tool is suitable and what alternatives might work]"
        """

        try:
            tool_decision = self._call_ai_model(
                tool_selection_prompt, max_tokens=1000)

            # Extract the action line (EXECUTE:, CLARIFY:, or NONE:) from the response
            action_line = None
            for line in tool_decision.split('\n'):
                line = line.strip()
                if line.startswith(("EXECUTE:", "CLARIFY:", "NONE:")):
                    action_line = line
                    break

            if action_line is None:
                # Fallback - use the whole response if no action line found
                action_line = tool_decision.strip()

            if action_line.startswith("EXECUTE:"):
                # Parse and execute the tool
                execution_part = action_line[8:].strip()
                if "|" in execution_part:
                    tool_name, args_json = execution_part.split("|", 1)
                    try:
                        args = json.loads(args_json)
                        return await self._execute_and_summarize(tool_name.strip(), args, original_request)
                    except json.JSONDecodeError as e:
                        error_msg = f"Error parsing tool arguments: {e}"
                        self.add_message("assistant", error_msg)
                        return error_msg
                else:
                    error_msg = "Invalid tool execution format"
                    self.add_message("assistant", error_msg)
                    return error_msg

            elif action_line.startswith("CLARIFY:"):
                clarification = action_line[8:].strip()
                self.add_message("assistant", clarification)
                return clarification

            elif action_line.startswith("NONE:"):
                explanation = action_line[5:].strip()
                self.add_message("assistant", explanation)
                return explanation

            else:
                # Fallback
                self.add_message("assistant", tool_decision)
                return tool_decision

        except Exception as e:
            error_msg = f"Error selecting tool: {e}"
            self.add_message("assistant", error_msg)
            return error_msg

    async def _execute_and_summarize(self, tool_name: str, args: Dict[str, Any], original_request: str) -> str:
        """Execute a tool and summarize the results"""

        print(f"⚡ Executing tool: {tool_name}")

        # Execute the tool
        result = await self.execute_tool(tool_name, args)

        if not result:
            error_msg = f"Failed to execute tool {tool_name}"
            self.add_message("assistant", error_msg)
            return error_msg

        # Step 3: Summarize and respond based on tool results
        try:
            args_display = json.dumps(args)
        except (TypeError, ValueError):
            args_display = str(args)
            
        summary_prompt = f"""I executed the tool "{tool_name}" with arguments {args_display} in response to the user's request: "{original_request}"

        Tool execution result:
        {result}

        Please provide a helpful response to the user based on these results. Be specific and actionable. If there was an error in the tool execution, explain what went wrong and suggest alternatives.

        Format your response as a natural conversation with the user."""

        try:
            final_response = self._call_ai_model(
                summary_prompt, max_tokens=1500)
            self.add_message("assistant", final_response)

            # Check if agent should continue acting
            if self.should_continue_acting(final_response):
                print("🤖 Agent is continuing to act...")
                return final_response + "\n\n" + await self.think_and_respond("continue with the previous task")

            return final_response

        except Exception as e:
            error_msg = f"Error summarizing results: {e}"
            self.add_message("assistant", error_msg)
            return error_msg

    async def start_chat_loop(self):
        """Start the interactive chat loop"""
        print("\n" + "="*60)
        print("🤖 MCP Gateway Client (Direct MCP Connection)")
        print("Type your requests and I'll help you using available MCP tools!")
        print("Type 'quit' or 'exit' to end the session.")
        print("="*60)

        while True:
            try:
                # Get user input
                user_input = input("\n👤 You: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['quit', 'exit']:
                    print("👋 Goodbye!")
                    break

                # Process the input and respond
                response = await self.think_and_respond(user_input)
                print(f"\n{'='*50}")
                print(f"🤖 Assistant: {response}")
                print('='*50)

                # If response doesn't end with newline, add one
                if not response.endswith('\n'):
                    print()

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()


async def main():
    """Main function to run the client"""
    import argparse

    parser = argparse.ArgumentParser(description="MCP Gateway Client (Direct MCP Connection)")
    parser.add_argument("--gateway-mcp-url", default="http://localhost:3000/mcp",
                        help="Gateway MCP URL (default: http://localhost:3000/mcp)")
    parser.add_argument("--ai-provider", choices=["auto", "anthropic", "openai"], default="auto",
                        help="AI provider to use (default: auto)")
    parser.add_argument("--anthropic-api-key",
                        help="Anthropic API key (can also use ANTHROPIC_API_KEY env var)")
    parser.add_argument("--openai-api-key",
                        help="OpenAI API key (can also use OPENAI_API_KEY env var)")

    args = parser.parse_args()

    client = None
    try:
        client = MCPGatewayClient(
            gateway_mcp_url=args.gateway_mcp_url,
            ai_provider=args.ai_provider,
            anthropic_api_key=args.anthropic_api_key,
            openai_api_key=args.openai_api_key
        )
        await client.connect_to_gateway()
        await client.start_chat_loop()
    except Exception as e:
        print(f"❌ Failed to start client: {e}")
        return 1
    finally:
        if client:
            await client.cleanup()

    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))
#!/usr/bin/env python3
"""
Example HTTP client that connects to the MCP Gateway and provides an interactive chat interface.

This client demonstrates how to:
1. Search for tools using the gateway's HTTP API
2. Execute tools via the gateway
3. Implement an LLM loop with linear chat memory
4. Handle tool search -> tool execute -> summary workflow
"""

import json
import time
import requests
import aiohttp
import asyncio
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

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


class MCPGatewayHTTPClient:
    def __init__(self, gateway_url: str = "http://localhost:8080",
                 ai_provider: str = "auto",
                 anthropic_api_key: Optional[str] = None,
                 openai_api_key: Optional[str] = None):
        self.gateway_url = gateway_url.rstrip('/')
        self.chat_history: List[ChatMessage] = []

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

        # Test gateway connection
        self._test_gateway_connection()

        print(f"✓ Connected to MCP Gateway at {self.gateway_url}")
        print(f"✓ {self.ai_provider.title()} client initialized")

    def _test_gateway_connection(self):
        """Test if the gateway is accessible"""
        try:
            response = requests.get(f"{self.gateway_url}/health", timeout=5)
            response.raise_for_status()
            print(f"Gateway health check passed: {response.json()}")
        except Exception as e:
            print(
                f"Warning: Could not connect to gateway at {self.gateway_url}: {e}")
            print("Make sure the gateway is running and accessible")

    def search_tools(self, query: str) -> List[Dict[str, Any]]:
        """Search for tools using the gateway's search endpoint"""
        try:
            response = requests.post(
                f"{self.gateway_url}/mcp/search",
                json={"query": query},
                timeout=30
            )
            response.raise_for_status()
            result = response.json()

            if result.get("status") == "completed" and "result" in result:
                # Parse the result if it's a JSON string
                tools_data = result["result"]
                if isinstance(tools_data, str):
                    tools_data = json.loads(tools_data)
                return tools_data if isinstance(tools_data, list) else []

            return []
        except Exception as e:
            print(f"Error searching tools: {e}")
            return []

    async def execute_tool(self, tool_name: str, args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute a tool using the gateway's execute endpoint"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.gateway_url}/mcp/execute",
                    json={
                        "tool_name": tool_name,
                        "args": args
                    },
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    response.raise_for_status()
                    result = await response.json()

                    if result.get("status") == "completed" and "result" in result:
                        # Parse the result if it's a JSON string
                        execution_result = result["result"]
                        if isinstance(execution_result, str):
                            execution_result = json.loads(execution_result)
                        return execution_result

                    return None
        except Exception as e:
            print(f"Error executing tool {tool_name}: {e}")
            return None

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

        # First, ask Claude to analyze what to do
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
        1. **gateway_search**: Use this to discover available MCP tools
           - The payload MUST be a JSON string in the format: {{"query": "your search query"}}
           - Example: payload='{{"query": "search the web"}}' or payload='{{"query": "weather tools"}}'
           - The search will return matching tools with their server names and exact tool names

        2. **gateway_execute**: Use this to execute a specific MCP tool after discovering it
           - The payload MUST be a JSON string in the format: {{"tool_name": "<exact_name_from_search>", "args": {...}}}
           - Use the EXACT tool_name returned from gateway_search
           - Example: payload='{{"tool_name": "exa_search", "args": {{"query": "latest AI news"}}}}'

        Important workflow:
        - ALWAYS search for tools first using gateway_search before trying to execute them
        - Use the exact tool names returned from search results when executing
        - Include appropriate arguments based on the tool's requirements

        Be professional, friendly, and focused on helping users achieve their goals.
"""

        try:
            analysis_text = self._call_ai_model(
                analysis_prompt, max_tokens=1000)

            if "DIRECT:" in analysis_text:
                # Respond directly
                direct_response = analysis_text[7:].strip()
                self.add_message("assistant", direct_response)
                return direct_response

            elif "SEARCH:" in analysis_text:
                # Search for and use tools
                search_query = analysis_text[7:].strip()
                return await self._handle_tool_workflow(search_query, user_input)

            elif "EXECUTE:" in analysis_text:
                # Execute a tool
                tool_name = analysis_text[8:].strip()
                return await self._execute_and_summarize(tool_name, {}, user_input)
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
        tools = self.search_tools(search_query)

        if not tools:
            response = f"I searched for tools to help with '{search_query}' but didn't find any suitable tools. Let me try to help you directly instead."
            self.add_message("assistant", response)
            return response

        print(f"📋 Found {len(tools)} relevant tools")

        # Step 2: Let Claude decide which tool to use and how
        tool_selection_prompt = f"""I found these tools that might help with the request "{original_request}":

        Available tools:
        {json.dumps(tools, indent=2)}

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
        summary_prompt = f"""I executed the tool "{tool_name}" with arguments {json.dumps(args)} in response to the user's request: "{original_request}"

        Tool execution result:
        {json.dumps(result, indent=2)}

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

    def start_chat_loop(self):
        """Start the interactive chat loop"""
        print("\n" + "="*60)
        print("🤖 MCP Gateway HTTP Client")
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
                response = asyncio.run(self.think_and_respond(user_input))
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


def main():
    """Main function to run the client"""
    import argparse

    parser = argparse.ArgumentParser(description="MCP Gateway HTTP Client")
    parser.add_argument("--gateway-url", default="http://localhost:8080",
                        help="Gateway URL (default: http://localhost:8080)")
    parser.add_argument("--ai-provider", choices=["auto", "anthropic", "openai"], default="auto",
                        help="AI provider to use (default: auto)")
    parser.add_argument("--anthropic-api-key",
                        help="Anthropic API key (can also use ANTHROPIC_API_KEY env var)")
    parser.add_argument("--openai-api-key",
                        help="OpenAI API key (can also use OPENAI_API_KEY env var)")

    args = parser.parse_args()

    try:
        client = MCPGatewayHTTPClient(
            gateway_url=args.gateway_url,
            ai_provider=args.ai_provider,
            anthropic_api_key=args.anthropic_api_key,
            openai_api_key=args.openai_api_key
        )
        client.start_chat_loop()
    except Exception as e:
        print(f"❌ Failed to start client: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

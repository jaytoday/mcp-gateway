# Example Clients

This directory contains example client implementations showing how to integrate with the MCP Gateway.

## 🔗 Two Integration Approaches

### 1. HTTP API Client (`example_client_http.py`)

**Best for**: Web apps, microservices, custom integrations

- Uses simple HTTP requests to communicate with the gateway
- Works with any HTTP client library
- Session management via headers
- Easy to integrate into existing web services

**Usage:**
```bash
# Terminal 1: Start the gateway
make start

# Terminal 2: Run the HTTP client example
make example-http
```

### 2. MCP Protocol Client (`example_client_mcp.py`)

**Best for**: MCP-native applications, better type safety

- Uses the native MCP protocol for communication
- Full MCP feature support (streaming, resources, etc.)
- Type-safe tool definitions
- Seamless integration with MCP-compatible tools

**Usage:**
```bash
# Terminal 1: Start the gateway
make start

# Terminal 2: Start the MCP server wrapper
make example-mcp-server

# Terminal 3: Run the MCP client example
make example-mcp
```

## 🛠️ Requirements

Both examples require:
```bash
# Install dependencies (automatically handled by setup)
pip install -r examples/requirements.txt
```

And a `.env` file with:
```
ANTHROPIC_API_KEY=your_anthropic_api_key_here
OPENAI_API_KEY=your_openai_api_key_here  # Optional, for OpenAI models
```

## 🎯 What These Examples Show

- **Tool Discovery**: How to fetch available tools from the gateway
- **Tool Execution**: How to call tools with arguments
- **Claude Integration**: How to use tools with Claude's function calling
- **Session Management**: How to maintain context across calls
- **Error Handling**: How to handle failures gracefully
- **Interactive Mode**: How to build a conversational interface

## 🔧 Customization

These examples are templates you can modify for your specific needs:

- **Different LLMs**: Replace Anthropic client with OpenAI, local models, etc.
- **Custom Tool Logic**: Add preprocessing/postprocessing of tool results
- **UI Integration**: Adapt the patterns for web UIs, chat interfaces, etc.
- **Async Patterns**: Both examples use async/await for non-blocking operations

## 📚 Next Steps

1. **Start Simple**: Try the HTTP client first - it's easier to understand
2. **Add Tools**: Configure MCP servers in your gateway for the examples to call
3. **Customize**: Modify the examples for your specific use case
4. **Scale Up**: Use these patterns in your production applications
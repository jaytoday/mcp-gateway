# MCP Gateway

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A **lightweight, modular, and self-hostable** routing proxy for agent traffic to [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) servers. This gateway enables seamless integration between AI agents and MCP servers, providing intelligent request routing, session management, and usage analytics. There are some very interesting downstream applications of a more involved version of this but I wanted to just release a very lightweight and extendable version :)

**Self-Host Ready**: Single Python file deployment with minimal dependencies  
**Highly Modular**: Easy to extend, modify, and customize for your specific needs  
**Zero Lock-in**: Full control over your infrastructure and data

## Features

- **Intelligent Routing**: Routes requests to appropriate MCP servers using OpenAI-powered tool matching
- **Session Management**: Persistent session handling for stateful interactions
- **Redis Integration**: Queue-based request processing with Redis backend
- **HTTP API**: RESTful interface for easy integration with any agent runtime
- **Usage Analytics**: Built-in metrics and monitoring capabilities
- **Extensible**: Easy to add new MCP servers and custom routing logic

## Architecture

The MCP Gateway uses a simple, modular architecture designed for easy self-hosting and customization:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   AI Agents     │    │  MCP Gateway    │    │   Any Tool      │
│                 │    │                 │    │                 │
│ • Claude Code   │◄──►│ • Smart Routing │◄──►│ • MCP           │
│ • Custom Agents │    │ • Session Mgmt  │    │ • Other Agents  │
│ • LLM Apps      │    │ • HTTP/MCP APIs │    │ • (any endpoint)│
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Two Integration Paths

1. **HTTP API Mode**: Direct REST API calls to the gateway
   - Perfect for web apps, microservices, and custom integrations
   - Simple HTTP POST/GET requests
   - Session management via headers

2. **MCP Server Mode**: Gateway acts as an MCP server itself
   - Use `mcp_remote.py` to expose gateway as an MCP server
   - Native MCP protocol support
   - Seamless integration with MCP-compatible agents

### Core Components

- **`gateway.py`**: Main routing engine with intelligent tool matching (currently supports all MCPs, can be extended to generic tools)
- **`session.py`**: Stateful session management for multi-turn conversations  
- **`mcp_remote.py`**: MCP server wrapper for native protocol support
- **`mcp_servers/`**: Example server implementations you can extend
- **`servers.txt`**: List of any MCPs you want to support

## 🚀 Quick Start (30 seconds)

```bash
git clone https://github.com/placeholder-labs/mcp-gateway.git
cd mcp-gateway
./scripts/quick-start.sh  # Sets up everything automatically
```

This creates a complete working environment with:
- Virtual environment and dependencies
- Environment configuration ready for customization
- Makefile with convenient commands

## 📦 Installation Options

### Option 1: Automated Setup (Recommended for first-time users)

```bash
git clone https://github.com/placeholder-labs/mcp-gateway.git
cd mcp-gateway
./scripts/quick-start.sh  # Automated setup with demo
```

**What this creates:**
- `venv/` - Isolated Python environment with all dependencies
- `.env` - Environment file (customize with your API keys)
- `mcp_servers/demo_server.py` - Working MCP server (no API keys needed)
- `demo.py` - Interactive demo script
- `Makefile` - Convenient commands (`make start`, `make demo`, etc.)

## Try the Gateway

After running `./scripts/quick-start.sh`, you have multiple ways to explore:

### 1. Interactive HTTP Client
```bash
make start         # Terminal 1: Start gateway
make example-http  # Terminal 2: Run HTTP client
```

### 2. MCP Protocol Client
```bash
make start              # Terminal 1: Gateway
make example-mcp-server # Terminal 2: MCP wrapper
make example-mcp        # Terminal 3: MCP client
```

### Adding MCP Servers

MCP servers can be configured via HTTP API or by modifying the server configuration. Example locally developed servers are provided in the `mcp_servers/` directory, but the gateway can connect to MCP servers run remotely and via npx.

## Gateway Endpoints

The MCP Gateway provides a comprehensive HTTP API for managing MCP servers, executing tools, and handling sessions. All endpoints support CORS and can be used from web applications or any HTTP client.

### Core MCP Operations

#### Search for Tools
**POST /mcp/search**
```bash
curl -X POST http://localhost:8000/mcp/search \
  -H "Content-Type: application/json" \
  -d '{"query": "search the web for information"}'

# With session support
curl -X POST http://localhost:8000/mcp/search \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: your-session-id" \
  -d '{"query": "search the web for information"}'
```

Uses AI-powered routing to find the most relevant tools for your query. Returns tools ranked by relevance with reasoning.

#### Execute Tool
**POST /mcp/execute**
```bash
curl -X POST http://localhost:8000/mcp/execute \
  -H "Content-Type: application/json" \
  -d '{
    "tool_name": "search_web",
    "args": {
      "query": "MCP protocol documentation"
    }
  }'

# With session support  
curl -X POST http://localhost:8000/mcp/execute \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: your-session-id" \
  -d '{
    "tool_name": "search_web", 
    "args": {"query": "MCP protocol documentation"}
  }'
```

Executes a specific tool with the provided arguments. Automatically routes to the correct MCP server.

#### List Available Tools
**GET /mcp/tools**
```bash
curl http://localhost:8000/mcp/tools

# Session-specific tools
curl -H "X-Session-ID: your-session-id" http://localhost:8000/mcp/tools
```

Returns all available tools from connected MCP servers with their descriptions and input schemas.

### Server Management

#### List Connected Servers
**GET /mcp/servers**
```bash
curl http://localhost:8000/mcp/servers

# Session-specific servers
curl -H "X-Session-ID: your-session-id" http://localhost:8000/mcp/servers
```

Shows all connected MCP servers, their transport types, and available tools.

#### Add MCP Servers
**POST /mcp/servers/add**
```bash
curl -X POST http://localhost:8000/mcp/servers/add \
  -H "Content-Type: application/json" \
  -d '{
    "config_data": "{\"mcpServers\": {\"exa_search\": {\"command\": \"npx\", \"args\": [\"@exa-ai/exa-mcp-server\"], \"transport\": \"stdio\"}}}"
  }'

# Add to specific session
curl -X POST http://localhost:8000/mcp/servers/add \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: your-session-id" \
  -d '{
    "config_data": "{\"mcpServers\": {\"custom_server\": {\"command\": \"python\", \"args\": [\"path/to/server.py\"]}}}"
  }'
```

Dynamically adds new MCP servers using the same format as the servers.txt configuration file.

#### Remove MCP Server
**POST /mcp/servers/remove**
```bash
curl -X POST http://localhost:8000/mcp/servers/remove \
  -H "Content-Type: application/json" \
  -d '{"server_uuid": "server-uuid-here"}'

# Or remove by name
curl -X POST http://localhost:8000/mcp/servers/remove \
  -H "Content-Type: application/json" \
  -d '{"server_name": "exa_search"}'

# Remove from specific session
curl -X POST http://localhost:8000/mcp/servers/remove \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: your-session-id" \
  -d '{"server_uuid": "server-uuid-here"}'
```

Removes a connected MCP server and cleans up its resources.

### Session Management

#### Create Session
**POST /sessions/create**
```bash
curl -X POST http://localhost:8000/sessions/create \
  -H "Content-Type: application/json" \
  -d '{"metadata": {"user": "alice", "project": "research"}}'
```

Creates a new isolated session with its own set of MCP server connections. Automatically loads servers from servers.txt.

#### Get Session Details
**GET /sessions/{session_id}**
```bash
curl http://localhost:8000/sessions/sess_abc123def456
```

Returns detailed information about a session including connected servers and available tools.

#### List Active Sessions
**GET /sessions**
```bash
curl http://localhost:8000/sessions
```

Shows all currently active sessions with their metadata and connection counts.

#### Delete Session
**DELETE /sessions/{session_id}**
```bash
curl -X DELETE http://localhost:8000/sessions/sess_abc123def456
```

Deletes a session and cleans up all its associated server connections and resources.

### Health and Monitoring

#### Health Check
**GET /health**
```bash
curl http://localhost:8000/health
```

Returns gateway health status, connected server count, and total available tools.

### Session-Aware Operations

Most endpoints support session isolation using the `X-Session-ID` header:

- **Global Mode**: Without header, operates on globally connected servers
- **Session Mode**: With header, operates only on servers connected to that specific session

This allows multiple users or applications to maintain separate MCP server environments while sharing the same gateway instance.

### [OPTIONAL] Queue-based Processing

The gateway also optionally supports Redis queue-based processing for asynchronous operations:

- **Request Queue**: `mcp_requests`
- **Response Queue**: `mcp_responses`

This enables integration with job queue systems and asynchronous processing workflows.


## Examples

### 🚀 Ready-to-Run Examples

Complete example clients are in the `examples/` directory:

#### HTTP API Client (`examples/example_client_http.py`)
**Architecture**: Direct REST API integration with the gateway
**Showcases**: 
- Simple HTTP requests for tool discovery and execution
- Session management via headers
- LLM integration (Claude/OpenAI) with function calling
- Interactive chat loop with tool search and execution workflow

```bash
# Terminal 1: Start the gateway
make start

# Terminal 2: Run the interactive HTTP client
make example-http
```

#### MCP Protocol Client (`examples/example_client_mcp.py`)
**Architecture**: Native MCP protocol communication via the gateway's MCP server wrapper
**Showcases**:
- Full MCP protocol support with type safety
- Streaming capabilities and resource management
- Direct MCP tool invocation without HTTP overhead
- Seamless integration with MCP-compatible frameworks

```bash
# Terminal 1: Start the gateway
make start

# Terminal 2: Start the MCP server wrapper
make example-mcp-server

# Terminal 3: Run the interactive MCP client
make example-mcp
```

See [`examples/README.md`](examples/README.md) for detailed documentation.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- Email: oliverye@berkeley.edu

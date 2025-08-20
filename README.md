# MCP Gateway

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A **lightweight, modular, and self-hostable** routing proxy for agent traffic to [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) servers. This gateway enables seamless integration between AI agents and MCP servers, providing intelligent request routing, session management, and usage analytics.

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

## 🚀 Quick Start (30 seconds)

**Zero-config demo that works without API keys:**

```bash
git clone https://github.com/placeholder-labs/mcp-gateway.git
cd mcp-gateway
./scripts/quick-start.sh  # Sets up everything automatically
make start               # Terminal 1: Start gateway
make demo               # Terminal 2: Run interactive demo
```

This creates a complete working environment with:
- Virtual environment and dependencies
- Demo MCP server with basic tools (time, calculator, echo)
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

### Option 2: Manual Setup

```bash
git clone https://github.com/placeholder-labs/mcp-gateway.git
cd mcp-gateway
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

### Option 3: Production Installation

```bash
pip install mcp-gateway  # Coming soon
```

## 🎮 Try the Gateway

After running `./scripts/quick-start.sh`, you have multiple ways to explore:

### 1. Automated Demo (Easiest)
```bash
make start    # Terminal 1: Start gateway
make demo     # Terminal 2: See it in action
```

### 2. Interactive HTTP Client
```bash
make start         # Terminal 1: Start gateway
make example-http  # Terminal 2: Run HTTP client
```

### 3. MCP Protocol Client
```bash
make start              # Terminal 1: Gateway
make example-mcp-server # Terminal 2: MCP wrapper
make example-mcp        # Terminal 3: MCP client
```

**Common commands:**
- `make help` - See all available commands
- `make example-http` - Run HTTP client example
- `make example-mcp-server` - Start MCP server wrapper
- `make example-mcp` - Run MCP client example
- `make dev-shell` - Enter development environment
- `source venv/bin/activate` - Manually activate virtual environment

## Configuration

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `OPENAI_API_KEY` | OpenAI API key for intelligent routing | No | - |
| `REDIS_HOST` | Redis server hostname | No | localhost |
| `REDIS_PORT` | Redis server port | No | 6379 |
| `GATEWAY_HOST` | Gateway server host | No | localhost |
| `GATEWAY_PORT` | Gateway server port | No | 8000 |
| `LOG_LEVEL` | Logging level | No | INFO |

### Running Without Redis

The gateway can run without Redis by using only the HTTP API endpoints. Redis is used for queue-based processing, tool inventory caching, and request/response storage. If you don't need queue-based messaging, you can:

1. **Use HTTP API Only**: Access all functionality through REST endpoints like `/mcp/search`, `/mcp/execute`, `/mcp/tools`, and `/mcp/servers`
2. **Skip Redis Setup**: Simply don't configure Redis environment variables - the gateway will still start and serve HTTP requests
3. **In-Memory Sessions**: Session management uses in-memory storage by default, no external dependencies required

The HTTP API provides complete functionality including tool routing, server management, and session handling without any Redis dependencies.

### Adding MCP Servers

MCP servers can be configured via HTTP API or by modifying the server configuration. Example servers are provided in the `mcp_servers/` directory.

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

### Queue-based Processing

The gateway also supports Redis queue-based processing for asynchronous operations:

- **Request Queue**: `mcp_requests`
- **Response Queue**: `mcp_responses`

This enables integration with job queue systems and asynchronous processing workflows.

## Architecture

The MCP Gateway uses a simple, modular architecture designed for easy self-hosting and customization:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   AI Agents     │    │  MCP Gateway    │    │   MCP Servers   │
│                 │    │                 │    │                 │
│ • Claude Code   │◄──►│ • Smart Routing │◄──►│ • Search Tools  │
│ • Custom Agents │    │ • Session Mgmt  │    │ • File Tools    │
│ • LLM Apps      │    │ • HTTP/MCP APIs │    │ • Custom Tools  │
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

- **`gateway.py`**: Main routing engine with intelligent tool matching
- **`session.py`**: Stateful session management for multi-turn conversations  
- **`mcp_remote.py`**: MCP server wrapper for native protocol support
- **`mcp_servers/`**: Example server implementations you can extend

## 📁 Project Structure

```
├── gateway.py              # Main gateway service
├── mcp_remote.py           # MCP server wrapper for the gateway
├── session.py              # Session management
├── mcp_servers/            # Example MCP servers
│   └── exa_server.py      # Exa search server example
├── examples/               # Example client implementations
│   ├── example_client_http.py  # HTTP API client example
│   ├── example_client_mcp.py   # MCP protocol client example
│   └── README.md          # Examples documentation
├── client/                 # Client libraries
│   ├── client.py          # Python client
│   └── requirements.txt   # Client dependencies
├── requirements.txt        # Core dependencies
└── pyproject.toml         # Package configuration
```

## Examples

### 🚀 Ready-to-Run Examples

We provide complete example clients in the `examples/` directory:

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

### 🔧 Quick Integration Snippets

#### HTTP API Integration

```python
import httpx
import asyncio

async def use_gateway():
    async with httpx.AsyncClient() as client:
        # List available tools
        tools = await client.get("http://localhost:8000/tools")
        print("Available tools:", tools.json())
        
        # Execute a search tool
        result = await client.post(
            "http://localhost:8000/tool",
            json={
                "name": "search_web",
                "arguments": {"query": "MCP protocol documentation"}
            }
        )
        print("Search result:", result.json())

asyncio.run(use_gateway())
```

#### MCP Server Integration

```bash
# Start the gateway
python gateway.py

# In another terminal, start the MCP server wrapper
python mcp_remote.py --port 3000

# Now agents can connect to localhost:3000/mcp as an MCP server
# The wrapper will proxy all requests to your gateway
```

### Example 3: Adding Custom MCP Servers

```python
# Create your custom server in mcp_servers/my_server.py
# Then add it via HTTP API:

import httpx

config = {
    "mcpServers": {
        "my_custom_server": {
            "command": "python",
            "args": ["mcp_servers/my_server.py"],
            "env": {"API_KEY": "your_key_here"}
        }
    }
}

async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8000/add-server",
        json=config
    )
    print("Server added:", response.json())
```

### Example 4: Self-Hosting with Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "gateway.py"]
```

```bash
# Build and run
docker build -t mcp-gateway .
docker run -p 8000:8000 -e OPENAI_API_KEY=your_key mcp-gateway
```

## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔧 Modular Architecture

This gateway is intentionally designed with **lightweight, replaceable components** to make it easy for developers to customize and scale according to their needs. Key modular points include:

- **Tool Routing**: Currently uses OpenAI for semantic matching, but can be swapped for vector embeddings, regex patterns, or custom algorithms
- **Queue System**: Redis-based queue processing can be replaced with RabbitMQ, Kafka, or in-memory alternatives  
- **Session Storage**: In-memory storage is provided for development - easily replace with PostgreSQL, MongoDB, or distributed stores
- **LLM Provider**: OpenAI integration can be swapped for Claude, local models, or any chat completion API
- **Transport Protocols**: Supports stdio/HTTP/SSE out of the box - extend with WebSocket, gRPC, or custom protocols
- **Authentication**: Ready for extension with JWT, API keys, or OAuth integration
- **Configuration**: JSON/env file based - can be enhanced with Consul, etcd, or Kubernetes ConfigMaps

The codebase uses clear interfaces and dependency injection patterns, making component replacement straightforward. Start with the pieces most relevant to your infrastructure and gradually enhance others as needed.

## Support

- Email: oliverye@berkeley.edu

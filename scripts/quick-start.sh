#!/bin/bash

# MCP Gateway Quick Start Script
# This script sets up everything you need to run the MCP Gateway demo

set -e  # Exit on any error

echo "🚀 MCP Gateway Quick Start"
echo "=========================="

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_step() {
    echo -e "${BLUE}▶ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if we're in the right directory
if [ ! -f "gateway.py" ]; then
    print_error "gateway.py not found. Please run this script from the project root directory."
    exit 1
fi

print_step "Checking Python version..."
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is required but not installed."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
print_success "Python $PYTHON_VERSION detected"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    print_step "Creating virtual environment..."
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_success "Virtual environment already exists"
fi

print_step "Activating virtual environment..."
source venv/bin/activate

print_step "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Install additional dependencies for examples
pip install -q anthropic

print_success "Dependencies installed"

# Set up environment file
if [ ! -f ".env" ]; then
    print_step "Setting up environment file..."
    cp .env.example .env
    print_success "Environment file created (.env)"
    print_warning "Edit .env file to add your API keys if needed"
else
    print_success "Environment file already exists"
fi

# Create a simple demo MCP server
print_step "Creating demo MCP server..."
cat > mcp_servers/demo_server.py << 'EOF'
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
EOF

chmod +x mcp_servers/demo_server.py
print_success "Demo MCP server created"

# Create a Makefile for convenience
print_step "Creating Makefile..."
cat > Makefile << 'EOF'
.PHONY: help setup start demo clean test example-http example-mcp example-mcp-server

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup:  ## Set up the development environment
	@./scripts/quick-start.sh

start:  ## Start the gateway server
	@echo "🚀 Starting MCP Gateway..."
	@source venv/bin/activate && python gateway.py

demo:  ## Run the interactive demo
	@echo "🎮 Running demo..."
	@source venv/bin/activate && python demo.py

example-http:  ## Run the HTTP client example (requires gateway to be running)
	@echo "🌐 Running HTTP client example..."
	@echo "Make sure the gateway is running first with 'make start'"
	@source venv/bin/activate && python examples/example_client_http.py

example-mcp-server:  ## Start the MCP server wrapper (run this before example-mcp)
	@echo "🔌 Starting MCP server wrapper..."
	@echo "This exposes the gateway as an MCP server on port 3000"
	@source venv/bin/activate && python mcp_remote.py --port 3000

example-mcp:  ## Run the MCP client example (requires gateway and MCP server wrapper)
	@echo "🤖 Running MCP client example..."
	@echo "Make sure both 'make start' and 'make example-mcp-server' are running first"
	@source venv/bin/activate && python examples/example_client_mcp.py

clean:  ## Clean up virtual environment and cache files
	@echo "🧹 Cleaning up..."
	@rm -rf venv/
	@rm -rf __pycache__/
	@rm -rf *.pyc
	@find . -name "*.pyc" -delete
	@find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

test:  ## Run tests (when available)
	@source venv/bin/activate && python -m pytest tests/ || echo "No tests found"

dev-shell:  ## Activate development shell
	@echo "Activating development environment..."
	@exec bash --init-file <(echo "source venv/bin/activate; echo '🔧 Development environment activated'")
EOF

print_success "Makefile created"

echo ""
echo "🎉 Setup Complete!"
echo "=================="
echo ""
echo "Quick commands to get started:"
echo ""
echo -e "${GREEN}# Start the gateway:${NC}"
echo "  make start"
echo "  # OR: source venv/bin/activate && python gateway.py"
echo ""
echo -e "${GREEN}# Try the example clients:${NC}"
echo "  make example-http      # HTTP client (needs gateway running)"
echo "  make example-mcp-server && make example-mcp  # MCP client (3 terminals)"
echo ""
echo -e "${YELLOW}📝 Next steps:${NC}"
echo "  1. Edit .env to add your API keys (optional for demo)"
echo "  2. Run 'make start' to start the gateway"
echo "  3. In another terminal, run 'make demo' to see it work"
echo ""
echo -e "${BLUE}💡 Use 'make help' to see all available commands${NC}"

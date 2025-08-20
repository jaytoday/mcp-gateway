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

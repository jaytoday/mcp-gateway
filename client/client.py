#!/usr/bin/env python3
"""
MCP Gateway Client
A client library for interacting with the MCP Gateway HTTP API
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import httpx
import click
from rich.console import Console
from rich.table import Table
from rich.json import JSON
from rich import print as rprint

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rich console for pretty output
console = Console()


class MCPGatewayClient:
    """Client for interacting with the MCP Gateway HTTP API"""
    
    def __init__(self, gateway_url: str = "http://localhost:8080", session_id: Optional[str] = None, auto_create_session: bool = True):
        self.gateway_url = gateway_url.rstrip('/')
        self.client = httpx.AsyncClient(timeout=60.0)
        self.session_id = session_id
        self.auto_create_session = auto_create_session and session_id is None
        self._session_created = False
    
    async def __aenter__(self):
        if self.auto_create_session and not self.session_id:
            await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    async def _ensure_session(self):
        """Ensure a session exists, creating one if needed"""
        if not self.session_id and not self._session_created:
            try:
                result = await self.create_session({"auto_created": True, "client": "mcp-gateway-client"})
                # The create_session method already sets self.session_id
                if result.get("success") and self.session_id:
                    self._session_created = True
                    logger.info(f"Auto-created session: {self.session_id}")
            except Exception as e:
                logger.warning(f"Failed to auto-create session: {e}. Continuing without session.")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check the health status of the gateway"""
        try:
            response = await self.client.get(f"{self.gateway_url}/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            raise
    
    async def list_servers(self) -> Dict[str, Any]:
        """List all connected MCP servers"""
        await self._ensure_session()
        try:
            headers = self._add_session_header()
            response = await self.client.get(f"{self.gateway_url}/mcp/servers", headers=headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to list servers: {e}")
            raise
    
    async def list_tools(self) -> Dict[str, Any]:
        """List all available tools from all servers"""
        await self._ensure_session()
        try:
            headers = self._add_session_header()
            response = await self.client.get(f"{self.gateway_url}/mcp/tools", headers=headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to list tools: {e}")
            raise
    
    async def search_tools(self, query: str) -> Dict[str, Any]:
        """Search for tools matching a query"""
        await self._ensure_session()
        try:
            headers = self._add_session_header()
            response = await self.client.post(
                f"{self.gateway_url}/mcp/search",
                json={"query": query},
                headers=headers
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to search tools: {e}")
            raise
    
    async def execute_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific tool with arguments"""
        await self._ensure_session()
        try:
            headers = self._add_session_header()
            response = await self.client.post(
                f"{self.gateway_url}/mcp/execute",
                json={
                    "tool_name": tool_name,
                    "args": args
                },
                headers=headers
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to execute tool: {e}")
            raise
    
    async def add_servers(self, config_file: Path) -> Dict[str, Any]:
        """Add new MCP servers from a configuration file"""
        await self._ensure_session()
        try:
            # Read the configuration file
            config_data = config_file.read_text()
            
            headers = self._add_session_header()
            response = await self.client.post(
                f"{self.gateway_url}/mcp/servers/add",
                json={"config_data": config_data},
                headers=headers
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to add servers: {e}")
            raise
    
    async def remove_server(self, server_identifier: str) -> Dict[str, Any]:
        """Remove an MCP server by name or UUID"""
        await self._ensure_session()
        try:
            # Try to determine if it's a UUID or name
            # UUIDs have a specific format: 8-4-4-4-12 hexadecimal digits
            import re
            uuid_pattern = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.I)
            
            headers = self._add_session_header()
            
            if uuid_pattern.match(server_identifier):
                response = await self.client.post(
                    f"{self.gateway_url}/mcp/servers/remove",
                    json={"server_uuid": server_identifier},
                    headers=headers
                )
            else:
                response = await self.client.post(
                    f"{self.gateway_url}/mcp/servers/remove",
                    json={"server_name": server_identifier},
                    headers=headers
                )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to remove server: {e}")
            raise
    
    # Session management methods
    async def create_session(self, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a new session"""
        try:
            response = await self.client.post(
                f"{self.gateway_url}/sessions/create",
                json={"metadata": metadata} if metadata else {}
            )
            response.raise_for_status()
            result = response.json()
            if result.get("success") and result.get("session_id"):
                self.session_id = result["session_id"]
            return result
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            raise
    
    async def get_session(self, session_id: str) -> Dict[str, Any]:
        """Get session details"""
        try:
            response = await self.client.get(
                f"{self.gateway_url}/sessions/{session_id}"
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get session: {e}")
            raise
    
    async def list_sessions(self) -> Dict[str, Any]:
        """List all active sessions"""
        try:
            response = await self.client.get(
                f"{self.gateway_url}/sessions"
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to list sessions: {e}")
            raise
    
    async def delete_session(self, session_id: str) -> Dict[str, Any]:
        """Delete a session"""
        try:
            response = await self.client.delete(
                f"{self.gateway_url}/sessions/{session_id}"
            )
            response.raise_for_status()
            result = response.json()
            # Clear session ID if we deleted our current session
            if self.session_id == session_id:
                self.session_id = None
            return result
        except Exception as e:
            logger.error(f"Failed to delete session: {e}")
            raise
    
    def _add_session_header(self, headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Add session ID to request headers if available"""
        if headers is None:
            headers = {}
        if self.session_id:
            headers["X-Session-ID"] = self.session_id
        return headers


def display_servers(servers_data: Dict[str, Any]):
    """Display servers in a formatted table"""
    servers = servers_data.get("servers", {})
    
    if not servers:
        console.print("[yellow]No servers connected[/yellow]")
        return
    
    table = Table(title="Connected MCP Servers")
    table.add_column("UUID", style="dim", no_wrap=True)
    table.add_column("Server Name", style="cyan", no_wrap=True)
    table.add_column("Transport", style="magenta")
    table.add_column("Tools", style="green")
    table.add_column("Tool Names", style="yellow")
    
    for server_uuid, info in servers.items():
        tools_list = info.get("tools", [])
        tools_preview = ", ".join(tools_list[:3])
        if len(tools_list) > 3:
            tools_preview += f" ... (+{len(tools_list) - 3} more)"
        
        # Display full UUID
        display_uuid = server_uuid
        
        table.add_row(
            display_uuid,
            info.get("name", "unknown"),
            info.get("transport", "unknown"),
            str(info.get("tool_count", 0)),
            tools_preview
        )
    
    console.print(table)


def display_tools(tools_data: Dict[str, Any]):
    """Display tools in a formatted table"""
    tools = tools_data.get("tools", [])
    
    if not tools:
        console.print("[yellow]No tools available[/yellow]")
        return
    
    table = Table(title=f"Available MCP Tools (Total: {tools_data.get('total', 0)})")
    table.add_column("Server", style="cyan", no_wrap=True)
    table.add_column("Tool Name", style="green", no_wrap=True)
    table.add_column("Description", style="yellow")
    
    for tool in tools:
        description = tool.get("description", "No description")
        # Truncate long descriptions
        if len(description) > 80:
            description = description[:77] + "..."
        
        table.add_row(
            tool.get("mcp_server", "unknown"),
            tool.get("name", "unknown"),
            description
        )
    
    console.print(table)


def display_search_results(search_data: Dict[str, Any]):
    """Display search results in a formatted table"""
    results = search_data.get("result", [])
    
    if not results:
        console.print("[yellow]No matching tools found[/yellow]")
        return
    
    table = Table(title="Search Results")
    table.add_column("Server", style="cyan", no_wrap=True)
    table.add_column("Tool Name", style="green", no_wrap=True)
    table.add_column("Score", style="magenta")
    table.add_column("Description", style="yellow")
    table.add_column("Reasoning", style="blue")
    
    for tool in results:
        description = tool.get("description", "No description")
        if len(description) > 50:
            description = description[:47] + "..."
        
        reasoning = tool.get("reasoning", "")
        if len(reasoning) > 50:
            reasoning = reasoning[:47] + "..."
        
        score = tool.get("match_score", 0.0)
        score_str = f"{score:.2f}"
        
        table.add_row(
            tool.get("mcp_server", "unknown"),
            tool.get("tool_name", "unknown"),
            score_str,
            description,
            reasoning
        )
    
    console.print(table)


@click.group()
@click.option('--gateway-url', default='http://localhost:8080', help='Gateway URL')
@click.option('--session-id', '-s', help='Session ID to use for commands')
@click.option('--no-auto-session', is_flag=True, help='Disable automatic session creation')
@click.pass_context
def cli(ctx, gateway_url, session_id, no_auto_session):
    """MCP Gateway Client CLI"""
    ctx.ensure_object(dict)
    ctx.obj['gateway_url'] = gateway_url
    ctx.obj['session_id'] = session_id
    ctx.obj['auto_create_session'] = not no_auto_session


@cli.command()
@click.pass_context
async def health(ctx):
    """Check gateway health status"""
    async with MCPGatewayClient(ctx.obj['gateway_url']) as client:
        try:
            result = await client.health_check()
            console.print("[green]✓ Gateway is healthy[/green]")
            console.print(f"Total servers connected (all sessions): {result.get('connected_servers', 0)}")
            console.print(f"Total tools available (all sessions): {result.get('total_tools', 0)}")
            console.print(f"Timestamp: {result.get('timestamp', 'unknown')}")
        except Exception as e:
            console.print(f"[red]✗ Gateway health check failed: {e}[/red]")


@cli.command()
@click.pass_context
async def servers(ctx):
    """List all connected MCP servers"""
    async with MCPGatewayClient(
        ctx.obj['gateway_url'], 
        session_id=ctx.obj.get('session_id'),
        auto_create_session=ctx.obj.get('auto_create_session', True)
    ) as client:
        if client.session_id:
            console.print(f"[dim]Using session: {client.session_id}[/dim]")
        try:
            result = await client.list_servers()
            display_servers(result)
        except Exception as e:
            console.print(f"[red]Failed to list servers: {e}[/red]")


@cli.command()
@click.pass_context
async def tools(ctx):
    """List all available tools"""
    async with MCPGatewayClient(
        ctx.obj['gateway_url'], 
        session_id=ctx.obj.get('session_id'),
        auto_create_session=ctx.obj.get('auto_create_session', True)
    ) as client:
        if client.session_id:
            console.print(f"[dim]Using session: {client.session_id}[/dim]")
        try:
            result = await client.list_tools()
            display_tools(result)
        except Exception as e:
            console.print(f"[red]Failed to list tools: {e}[/red]")


@cli.command()
@click.argument('query')
@click.pass_context
async def search(ctx, query):
    """Search for tools matching a query"""
    async with MCPGatewayClient(
        ctx.obj['gateway_url'], 
        session_id=ctx.obj.get('session_id'),
        auto_create_session=ctx.obj.get('auto_create_session', True)
    ) as client:
        if client.session_id:
            console.print(f"[dim]Using session: {client.session_id}[/dim]")
        try:
            console.print(f"[cyan]Searching for: {query}[/cyan]")
            result = await client.search_tools(query)
            display_search_results(result)
        except Exception as e:
            console.print(f"[red]Failed to search tools: {e}[/red]")


@cli.command()
@click.argument('tool_name')
@click.option('--args', '-a', help='Tool arguments as JSON string', default='{}')
@click.pass_context
async def execute(ctx, tool_name, args):
    """Execute a tool with arguments"""
    async with MCPGatewayClient(
        ctx.obj['gateway_url'], 
        session_id=ctx.obj.get('session_id'),
        auto_create_session=ctx.obj.get('auto_create_session', True)
    ) as client:
        if client.session_id:
            console.print(f"[dim]Using session: {client.session_id}[/dim]")
        try:
            # Parse arguments
            try:
                args_dict = json.loads(args)
            except json.JSONDecodeError:
                console.print("[red]Invalid JSON arguments[/red]")
                return
            
            console.print(f"[cyan]Executing tool: {tool_name}[/cyan]")
            result = await client.execute_tool(tool_name, args_dict)
            
            # Display result
            if result.get("status") == "error":
                console.print(f"[red]Error: {result.get('result', 'Unknown error')}[/red]")
            else:
                tool_result = result.get("result", {})
                if isinstance(tool_result, dict):
                    console.print("[green]✓ Tool executed successfully[/green]")
                    if "content" in tool_result:
                        console.print("\n[bold]Result:[/bold]")
                        console.print(tool_result["content"])
                    else:
                        console.print(JSON(json.dumps(tool_result, indent=2)))
                else:
                    console.print(f"[green]✓ Result:[/green] {tool_result}")
        except Exception as e:
            console.print(f"[red]Failed to execute tool: {e}[/red]")


@cli.command()
@click.argument('config_file', type=click.Path(exists=True, path_type=Path))
@click.pass_context
async def add_servers(ctx, config_file):
    """Add new MCP servers from a configuration file"""
    async with MCPGatewayClient(
        ctx.obj['gateway_url'], 
        session_id=ctx.obj.get('session_id'),
        auto_create_session=ctx.obj.get('auto_create_session', True)
    ) as client:
        if client.session_id:
            console.print(f"[dim]Using session: {client.session_id}[/dim]")
        try:
            console.print(f"[cyan]Adding servers from: {config_file}[/cyan]")
            result = await client.add_servers(config_file)
            
            # Display results
            added = result.get("added_servers", [])
            failed = result.get("failed_servers", [])
            
            if added:
                console.print(f"\n[green]✓ Successfully added {len(added)} server(s):[/green]")
                for server in added:
                    uuid_display = server.get('uuid', 'unknown')
                    console.print(f"  - {server['name']} [{uuid_display}]: {server['command']} ({server['tools_count']} tools)")
            
            if failed:
                console.print(f"\n[red]✗ Failed to add {len(failed)} server(s):[/red]")
                for server in failed:
                    console.print(f"  - {server['name']}: {server['error']}")
            
            # Handle both session and non-session responses
            total_servers = result.get('total_servers', result.get('total_servers_in_session', 0))
            console.print(f"\n[bold]Total servers connected:[/bold] {total_servers}")
        except Exception as e:
            console.print(f"[red]Failed to add servers: {e}[/red]")


@cli.command()
@click.argument('server_identifier')
@click.pass_context
async def remove_server(ctx, server_identifier):
    """Remove an MCP server by name or UUID"""
    async with MCPGatewayClient(
        ctx.obj['gateway_url'], 
        session_id=ctx.obj.get('session_id'),
        auto_create_session=ctx.obj.get('auto_create_session', True)
    ) as client:
        if client.session_id:
            console.print(f"[dim]Using session: {client.session_id}[/dim]")
        try:
            console.print(f"[cyan]Removing server: {server_identifier}[/cyan]")
            result = await client.remove_server(server_identifier)
            
            if result.get("success"):
                removed_name = result.get('removed_server', server_identifier)
                console.print(f"[green]✓ Successfully removed server '{removed_name}'[/green]")
                console.print(f"  UUID: {result.get('removed_server_uuid', 'unknown')}")
                console.print(f"  Removed {result.get('removed_tools_count', 0)} tools")
                
                # Display remaining servers with their UUIDs
                remaining = result.get('remaining_servers', {})
                if remaining:
                    remaining_list = [f"{name} ({uuid})" for uuid, name in remaining.items()]
                    console.print(f"  Remaining servers: {', '.join(remaining_list)}")
                else:
                    console.print("  No remaining servers")
                console.print(f"  Total servers: {result.get('total_servers', 0)}")
            else:
                console.print(f"[red]Failed to remove server[/red]")
        except Exception as e:
            console.print(f"[red]Failed to remove server: {e}[/red]")


@cli.command()
@click.option('--metadata', '-m', help='Session metadata as JSON string', default='{}')
@click.pass_context
async def session_create(ctx, metadata):
    """Create a new session"""
    async with MCPGatewayClient(ctx.obj['gateway_url']) as client:
        try:
            # Parse metadata
            try:
                metadata_dict = json.loads(metadata) if metadata else {}
            except json.JSONDecodeError:
                console.print("[red]Invalid JSON metadata[/red]")
                return
            
            result = await client.create_session(metadata_dict)
            
            if result.get("success"):
                console.print("[green]✓ Session created successfully[/green]")
                console.print(f"Session ID: [bold cyan]{result['session_id']}[/bold cyan]")
                console.print(f"Created at: {result.get('created_at', 'unknown')}")
                console.print(f"Expires at: {result.get('expires_at', 'unknown')}")
                if result.get('metadata'):
                    console.print(f"Metadata: {json.dumps(result['metadata'], indent=2)}")
            else:
                console.print("[red]Failed to create session[/red]")
        except Exception as e:
            console.print(f"[red]Failed to create session: {e}[/red]")


@cli.command()
@click.argument('session_id')
@click.pass_context
async def session_info(ctx, session_id):
    """Get information about a specific session"""
    async with MCPGatewayClient(ctx.obj['gateway_url']) as client:
        try:
            result = await client.get_session(session_id)
            
            # Display session info
            console.print(f"\n[bold]Session: {result['session_id']}[/bold]")
            console.print(f"Created: {result.get('created_at', 'unknown')}")
            console.print(f"Last accessed: {result.get('last_accessed', 'unknown')}")
            console.print(f"Expires: {result.get('expires_at', 'unknown')}")
            console.print(f"Active: {'✓' if result.get('active', False) else '✗'}")
            
            if result.get('metadata'):
                console.print("\n[bold]Metadata:[/bold]")
                console.print(JSON(json.dumps(result['metadata'], indent=2)))
            
            connections = result.get('connections', [])
            if connections:
                console.print(f"\n[bold]Connected Servers ({len(connections)}):[/bold]")
                table = Table()
                table.add_column("UUID", style="dim", no_wrap=True)
                table.add_column("Name", style="cyan")
                table.add_column("Transport", style="magenta")
                table.add_column("Tools", style="green")
                
                for conn in connections:
                    tools_preview = ", ".join(conn['tools'][:3])
                    if len(conn['tools']) > 3:
                        tools_preview += f" ... (+{len(conn['tools']) - 3} more)"
                    
                    table.add_row(
                        conn['uuid'],
                        conn['name'],
                        conn['transport'],
                        f"{conn['tools_count']} ({tools_preview})"
                    )
                
                console.print(table)
                console.print(f"\n[bold]Total tools:[/bold] {result.get('total_tools', 0)}")
            else:
                console.print("\n[yellow]No servers connected to this session[/yellow]")
                
        except Exception as e:
            console.print(f"[red]Failed to get session info: {e}[/red]")


@cli.command()
@click.pass_context
async def session_list(ctx):
    """List all active sessions"""
    async with MCPGatewayClient(ctx.obj['gateway_url']) as client:
        try:
            result = await client.list_sessions()
            
            sessions = result.get('sessions', [])
            if not sessions:
                console.print("[yellow]No active sessions[/yellow]")
                return
            
            table = Table(title=f"Active Sessions (Total: {result.get('total', 0)})")
            table.add_column("Session ID", style="cyan", no_wrap=True)
            table.add_column("Created", style="blue")
            table.add_column("Last Accessed", style="blue")
            table.add_column("Expires", style="yellow")
            table.add_column("Servers", style="green")
            table.add_column("Tools", style="magenta")
            
            for session in sessions:
                # Format timestamps for display
                created = session.get('created_at', 'unknown')
                if created != 'unknown':
                    created = created.split('T')[0] + ' ' + created.split('T')[1].split('.')[0]
                
                last_accessed = session.get('last_accessed', 'unknown')
                if last_accessed != 'unknown':
                    last_accessed = last_accessed.split('T')[1].split('.')[0]
                
                expires = session.get('expires_at', 'unknown')
                if expires != 'unknown':
                    expires = expires.split('T')[1].split('.')[0]
                
                table.add_row(
                    session['session_id'],
                    created,
                    last_accessed,
                    expires,
                    str(session.get('connections_count', 0)),
                    str(session.get('tools_count', 0))
                )
            
            console.print(table)
            
        except Exception as e:
            console.print(f"[red]Failed to list sessions: {e}[/red]")


@cli.command()
@click.argument('session_id')
@click.option('--force', '-f', is_flag=True, help='Skip confirmation')
@click.pass_context
async def session_delete(ctx, session_id, force):
    """Delete a specific session"""
    async with MCPGatewayClient(ctx.obj['gateway_url']) as client:
        try:
            # Get session info first
            try:
                session_info = await client.get_session(session_id)
                connections_count = len(session_info.get('connections', []))
                
                if not force and connections_count > 0:
                    if not click.confirm(f"Session has {connections_count} active server connection(s). Delete anyway?"):
                        console.print("[yellow]Deletion cancelled[/yellow]")
                        return
            except:
                # Session might not exist, continue with deletion
                pass
            
            result = await client.delete_session(session_id)
            
            if result.get("success"):
                console.print(f"[green]✓ Successfully deleted session {session_id}[/green]")
            else:
                console.print(f"[red]Failed to delete session[/red]")
                
        except Exception as e:
            console.print(f"[red]Failed to delete session: {e}[/red]")


# Wrapper function to run async commands
def run_async_command(async_func):
    """Wrapper to run async click commands"""
    def wrapper(*args, **kwargs):
        return asyncio.run(async_func(*args, **kwargs))
    wrapper.__name__ = async_func.__name__
    wrapper.__doc__ = async_func.__doc__
    return wrapper


# Apply wrapper to async commands
health.callback = run_async_command(health.callback)
servers.callback = run_async_command(servers.callback)
tools.callback = run_async_command(tools.callback)
search.callback = run_async_command(search.callback)
execute.callback = run_async_command(execute.callback)
add_servers.callback = run_async_command(add_servers.callback)
remove_server.callback = run_async_command(remove_server.callback)
session_create.callback = run_async_command(session_create.callback)
session_info.callback = run_async_command(session_info.callback)
session_list.callback = run_async_command(session_list.callback)
session_delete.callback = run_async_command(session_delete.callback)


if __name__ == "__main__":
    cli()
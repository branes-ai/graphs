"""Transport layer for the graphs MCP server.

Supports two modes:

stdio (default, no extra deps)
    For local Claude Code integration — the same pattern as branes MCP.
    ``python -m graphs.mcp``

SSE over HTTP (requires ``mcp``, ``starlette``, ``uvicorn``)
    For remote/team access with optional Bearer-token auth.
    ``python -m graphs.mcp --sse --port 8100``
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# stdio transport — no extra dependencies
# ---------------------------------------------------------------------------


def _run_stdio() -> None:
    """Run the MCP server over stdio using a simple JSON-RPC loop.

    Protocol: newline-delimited JSON on stdin/stdout.
    Each request is ``{"method": "...", "params": {...}, "id": ...}``.
    Responses are ``{"result": ..., "id": ...}`` or ``{"error": ..., "id": ...}``.
    """
    from graphs.mcp.server import execute_mcp_tool, get_mcp_tool_definitions

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            _write_response({"error": "Invalid JSON"}, request_id=None)
            continue

        request_id = request.get("id")
        method = request.get("method", "")
        params: dict[str, Any] = request.get("params", {})

        if method == "tools/list":
            _write_response({"tools": get_mcp_tool_definitions()}, request_id)
        elif method == "tools/call":
            tool_name = params.get("name", "")
            tool_args = params.get("arguments", {})
            result_json = execute_mcp_tool(tool_name, tool_args)
            _write_response({"content": json.loads(result_json)}, request_id)
        elif method == "initialize":
            _write_response({
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "graphs", "version": "0.8.0"},
                "capabilities": {"tools": {}},
            }, request_id)
        elif method == "ping":
            _write_response({}, request_id)
        else:
            _write_response({"error": f"Unknown method: {method}"}, request_id)


def _write_response(result: Any, request_id: Any) -> None:
    """Write a JSON-RPC response to stdout."""
    response = {"jsonrpc": "2.0", "result": result}
    if request_id is not None:
        response["id"] = request_id
    sys.stdout.write(json.dumps(response) + "\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# SSE transport — requires mcp, starlette, uvicorn
# ---------------------------------------------------------------------------


def _create_sse_app():
    """Create a Starlette app with SSE transport and optional auth."""
    from starlette.applications import Starlette
    from starlette.middleware import Middleware
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request
    from starlette.responses import JSONResponse, Response
    from starlette.routing import Mount, Route

    from mcp.server import Server
    from mcp.server.sse import SseServerTransport
    from mcp.types import TextContent, Tool

    from graphs.mcp.server import (
        execute_mcp_tool,
        get_mcp_tool_definitions,
    )

    # Build an MCP Server that delegates to our tool definitions
    def _make_mcp_server() -> Server:
        server = Server("graphs")

        @server.list_tools()
        async def _list_tools() -> list[Tool]:
            defs = get_mcp_tool_definitions()
            return [
                Tool(
                    name=d["name"],
                    description=d["description"],
                    inputSchema=d["input_schema"],
                )
                for d in defs
            ]

        @server.call_tool()
        async def _call_tool(name: str, arguments: dict) -> list[TextContent]:
            result_json = execute_mcp_tool(name, arguments)
            return [TextContent(type="text", text=result_json)]

        return server

    # Auth middleware
    class BearerAuthMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next) -> Response:
            token = os.environ.get("GRAPHS_MCP_TOKEN")
            if token is None:
                return await call_next(request)
            if request.url.path == "/health":
                return await call_next(request)
            from graphs.mcp.auth import validate_bearer

            auth_header = request.headers.get("Authorization", "")
            if not validate_bearer(auth_header):
                return JSONResponse({"error": "unauthorized"}, status_code=401)
            return await call_next(request)

    sse_transport = SseServerTransport("/messages")

    async def handle_sse(request: Request) -> Response:
        server = _make_mcp_server()
        async with sse_transport.connect_sse(
            request.scope, request.receive, request._send
        ) as streams:
            await server.run(
                streams[0], streams[1], server.create_initialization_options()
            )
        return Response()

    async def health(request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok", "server": "graphs-mcp"})

    return Starlette(
        routes=[
            Route("/health", endpoint=health, methods=["GET"]),
            Route("/sse", endpoint=handle_sse),
            Mount("/messages", app=sse_transport.handle_post_message),
        ],
        middleware=[Middleware(BearerAuthMiddleware)],
    )


def _run_sse(host: str, port: int) -> None:
    """Run the server over SSE/HTTP with uvicorn."""
    import uvicorn

    token = os.environ.get("GRAPHS_MCP_TOKEN")
    if token:
        logger.info("Auth enabled — Bearer token required")
    else:
        logger.warning("GRAPHS_MCP_TOKEN not set — auth disabled (local dev mode)")

    app = _create_sse_app()
    uvicorn.run(app, host=host, port=port)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    """Start the graphs MCP server."""
    parser = argparse.ArgumentParser(description="Graphs MCP server")
    parser.add_argument(
        "--sse", action="store_true",
        help="Run SSE/HTTP transport (requires mcp, starlette, uvicorn)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8100, help="Port (default: 8100)")
    parser.add_argument("--log-level", default="INFO", help="Log level (default: INFO)")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    if args.sse:
        _run_sse(args.host, args.port)
    else:
        _run_stdio()

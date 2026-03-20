"""Bearer-token authentication for the graphs MCP server."""

import hmac
import os


def get_token() -> str:
    """Return the expected token from the environment.

    Raises:
        RuntimeError: If GRAPHS_MCP_TOKEN is not set.
    """
    token = os.environ.get("GRAPHS_MCP_TOKEN")
    if not token:
        raise RuntimeError(
            "GRAPHS_MCP_TOKEN environment variable is not set. "
            "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
        )
    return token


def validate_bearer(authorization: str) -> bool:
    """Validate an ``Authorization: Bearer <token>`` header.

    Args:
        authorization: The full Authorization header value.

    Returns:
        True if the token matches, False otherwise.
    """
    if not authorization.startswith("Bearer "):
        return False
    return hmac.compare_digest(authorization[7:], get_token())

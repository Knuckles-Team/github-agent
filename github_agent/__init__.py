from scion_sdk.connector import Connector
from scion_sdk.connector_identity import connector_version

from github_agent.api_client import Api
from github_agent.auth import get_client
from github_agent.mcp_server import TOOL_REGISTRY

__version__ = connector_version("github_agent")

connector = Connector(
    api=Api,
    get_client=get_client,
    tool_registry=TOOL_REGISTRY,
    instructions="Github MCP Server - Manage your repositories, issues, and pull requests.",
)

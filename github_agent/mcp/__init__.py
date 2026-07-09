"""MCP tool registration modules for github-agent.

Auto-generated during ecosystem standardization.
Each domain has its own module with a register_*_tools function.
"""

from github_agent.mcp.mcp_action import register_action_tools
from github_agent.mcp.mcp_branch import register_branch_tools
from github_agent.mcp.mcp_collaborator import register_collaborator_tools
from github_agent.mcp.mcp_commit import register_commit_tools
from github_agent.mcp.mcp_content import register_content_tools
from github_agent.mcp.mcp_dependabot import register_dependabot_tools
from github_agent.mcp.mcp_issue import register_issue_tools
from github_agent.mcp.mcp_org import register_org_tools
from github_agent.mcp.mcp_pull import register_pull_tools
from github_agent.mcp.mcp_release import register_release_tools
from github_agent.mcp.mcp_repo import register_repo_tools
from github_agent.mcp.mcp_search import register_search_tools

__all__ = [
    "register_action_tools",
    "register_branch_tools",
    "register_collaborator_tools",
    "register_commit_tools",
    "register_content_tools",
    "register_dependabot_tools",
    "register_issue_tools",
    "register_org_tools",
    "register_pull_tools",
    "register_release_tools",
    "register_repo_tools",
    "register_search_tools",
]

#!/usr/bin/python
import sys

# coding: utf-8
import json
import os
import argparse
import logging
import uvicorn
import httpx
from typing import Optional, Any
from contextlib import asynccontextmanager

from pydantic_ai import Agent, ModelSettings, RunContext
from pydantic_ai.mcp import (
    load_mcp_servers,
    MCPServerStreamableHTTP,
    MCPServerSSE,
)
from pydantic_ai_skills import SkillsToolset
from fasta2a import Skill
from github_agent.utils import (
    to_integer,
    to_boolean,
    to_float,
    to_list,
    to_dict,
    get_mcp_config_path,
    get_skills_path,
    load_skills_from_directory,
    create_model,
    tool_in_tag,
    prune_large_messages,
)

from fastapi import FastAPI, Request
from starlette.responses import Response, StreamingResponse
from pydantic import ValidationError
from pydantic_ai.ui import SSE_CONTENT_TYPE
from pydantic_ai.ui.ag_ui import AGUIAdapter

__version__ = "0.2.10"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logging.getLogger("pydantic_ai").setLevel(logging.INFO)
logging.getLogger("fastmcp").setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_HOST = os.getenv("HOST", "0.0.0.0")
DEFAULT_PORT = to_integer(string=os.getenv("PORT", "9000"))
DEFAULT_DEBUG = to_boolean(string=os.getenv("DEBUG", "False"))
DEFAULT_PROVIDER = os.getenv("PROVIDER", "openai")
DEFAULT_MODEL_ID = os.getenv("MODEL_ID", "qwen/qwen3-coder-next")
DEFAULT_LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://host.docker.internal:1234/v1")
DEFAULT_LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")
DEFAULT_MCP_URL = os.getenv("MCP_URL", None)
DEFAULT_MCP_CONFIG = os.getenv("MCP_CONFIG", get_mcp_config_path())
DEFAULT_SKILLS_DIRECTORY = os.getenv("SKILLS_DIRECTORY", get_skills_path())
DEFAULT_ENABLE_WEB_UI = to_boolean(os.getenv("ENABLE_WEB_UI", "False"))
DEFAULT_SSL_VERIFY = to_boolean(os.getenv("SSL_VERIFY", "True"))

DEFAULT_MAX_TOKENS = to_integer(os.getenv("MAX_TOKENS", "16384"))
DEFAULT_TEMPERATURE = to_float(os.getenv("TEMPERATURE", "0.7"))
DEFAULT_TOP_P = to_float(os.getenv("TOP_P", "1.0"))
DEFAULT_TIMEOUT = to_float(os.getenv("TIMEOUT", "32400.0"))
DEFAULT_TOOL_TIMEOUT = to_float(os.getenv("TOOL_TIMEOUT", "32400.0"))
DEFAULT_PARALLEL_TOOL_CALLS = to_boolean(os.getenv("PARALLEL_TOOL_CALLS", "True"))
DEFAULT_SEED = to_integer(os.getenv("SEED", None))
DEFAULT_PRESENCE_PENALTY = to_float(os.getenv("PRESENCE_PENALTY", "0.0"))
DEFAULT_FREQUENCY_PENALTY = to_float(os.getenv("FREQUENCY_PENALTY", "0.0"))
DEFAULT_LOGIT_BIAS = to_dict(os.getenv("LOGIT_BIAS", None))
DEFAULT_STOP_SEQUENCES = to_list(os.getenv("STOP_SEQUENCES", None))
DEFAULT_EXTRA_HEADERS = to_dict(os.getenv("EXTRA_HEADERS", None))
DEFAULT_EXTRA_BODY = to_dict(os.getenv("EXTRA_BODY", None))

AGENT_NAME = "GitHubAgent"
AGENT_DESCRIPTION = (
    "A multi-agent system for interacting with GitHub via delegated specialists."
)


SUPERVISOR_SYSTEM_PROMPT = os.environ.get(
    "SUPERVISOR_SYSTEM_PROMPT",
    default=(
        "You are the GitHub Supervisor Agent.\n"
        "Your goal is to assist the user by assigning tasks to specialized child agents through your available toolset.\n"
        "Analyze the user's request and determine which domain(s) it falls into (e.g., issues, pull requests, repos, etc.).\n"
        "Then, call the appropriate tool(s) to delegate the task.\n"
        "Synthesize the results from the child agents into a final helpful response.\n"
        "Always be warm, professional, and helpful.\n"
        "Note: The final response should contain all the relevant information from the tool executions."
    ),
)

CONTEXT_AGENT_PROMPT = os.environ.get(
    "CONTEXT_AGENT_PROMPT",
    default=(
        "You are the GitHub Context Agent. "
        "Your goal is to provide context about the current user and functionality within GitHub."
    ),
)

ACTIONS_AGENT_PROMPT = os.environ.get(
    "ACTIONS_AGENT_PROMPT",
    default=(
        "You are the GitHub Actions Agent. "
        "Your goal is to manage GitHub Actions workflows and CI/CD operations."
    ),
)

CODE_SECURITY_AGENT_PROMPT = os.environ.get(
    "CODE_SECURITY_AGENT_PROMPT",
    default=(
        "You are the GitHub Code Security Agent. "
        "Your goal is to manage code security tools and scans."
    ),
)

DEPENDABOT_AGENT_PROMPT = os.environ.get(
    "DEPENDABOT_AGENT_PROMPT",
    default=(
        "You are the GitHub Dependabot Agent. "
        "Your goal is to manage Dependabot alerts and configurations."
    ),
)

DISCUSSIONS_AGENT_PROMPT = os.environ.get(
    "DISCUSSIONS_AGENT_PROMPT",
    default=(
        "You are the GitHub Discussions Agent. "
        "Your goal is to manage GitHub Discussions."
    ),
)

GISTS_AGENT_PROMPT = os.environ.get(
    "GISTS_AGENT_PROMPT",
    default=("You are the GitHub Gists Agent. " "Your goal is to manage GitHub Gists."),
)

GIT_AGENT_PROMPT = os.environ.get(
    "GIT_AGENT_PROMPT",
    default=(
        "You are the GitHub Git Agent. "
        "Your goal is to perform low-level Git operations via the GitHub API (e.g., refs, trees, blobs)."
    ),
)

ISSUES_AGENT_PROMPT = os.environ.get(
    "ISSUES_AGENT_PROMPT",
    default=(
        "You are the GitHub Issues Agent. "
        "Your goal is to manage GitHub Issues (create, list, update, comment)."
    ),
)

LABELS_AGENT_PROMPT = os.environ.get(
    "LABELS_AGENT_PROMPT",
    default=(
        "You are the GitHub Labels Agent. " "Your goal is to manage repository labels."
    ),
)

NOTIFICATIONS_AGENT_PROMPT = os.environ.get(
    "NOTIFICATIONS_AGENT_PROMPT",
    default=(
        "You are the GitHub Notifications Agent. "
        "Your goal is to manage and check GitHub notifications."
    ),
)

ORGS_AGENT_PROMPT = os.environ.get(
    "ORGS_AGENT_PROMPT",
    default=(
        "You are the GitHub Organizations Agent. "
        "Your goal is to manage GitHub Organizations and memberships."
    ),
)

PROJECTS_AGENT_PROMPT = os.environ.get(
    "PROJECTS_AGENT_PROMPT",
    default=(
        "You are the GitHub Projects Agent. "
        "Your goal is to manage GitHub Projects (V2/Beta)."
    ),
)

PULL_REQUESTS_AGENT_PROMPT = os.environ.get(
    "PULL_REQUESTS_AGENT_PROMPT",
    default=(
        "You are the GitHub Pull Requests Agent. "
        "Your goal is to manage Pull Requests (list, create, review, merge)."
    ),
)

REPOS_AGENT_PROMPT = os.environ.get(
    "REPOS_AGENT_PROMPT",
    default=(
        "You are the GitHub Repositories Agent. "
        "Your goal is to manage GitHub Repositories (create, list, delete, settings)."
    ),
)

SECRET_PROTECTION_AGENT_PROMPT = os.environ.get(
    "SECRET_PROTECTION_AGENT_PROMPT",
    default=(
        "You are the GitHub Secret Protection Agent. "
        "Your goal is to manage secret scanning and protection features."
    ),
)

SECURITY_ADVISORIES_AGENT_PROMPT = os.environ.get(
    "SECURITY_ADVISORIES_AGENT_PROMPT",
    default=(
        "You are the GitHub Security Advisories Agent. "
        "Your goal is to access and manage security advisories."
    ),
)

STARGAZERS_AGENT_PROMPT = os.environ.get(
    "STARGAZERS_AGENT_PROMPT",
    default=(
        "You are the GitHub Stargazers Agent. "
        "Your goal is to manage and view repository stargazers."
    ),
)

USERS_AGENT_PROMPT = os.environ.get(
    "USERS_AGENT_PROMPT",
    default=(
        "You are the GitHub Users Agent. "
        "Your goal is to access public user information and profile data."
    ),
)

COPILOT_AGENT_PROMPT = os.environ.get(
    "COPILOT_AGENT_PROMPT",
    default=(
        "You are the GitHub Copilot Agent. "
        "Your goal is to assist with coding tasks using GitHub Copilot."
    ),
)

COPILOT_SPACES_AGENT_PROMPT = os.environ.get(
    "COPILOT_SPACES_AGENT_PROMPT",
    default=(
        "You are the GitHub Copilot Spaces Agent. "
        "Your goal is to manage Copilot Spaces."
    ),
)

SUPPORT_DOCS_AGENT_PROMPT = os.environ.get(
    "SUPPORT_DOCS_AGENT_PROMPT",
    default=(
        "You are the GitHub Support Docs Agent. "
        "Your goal is to search GitHub documentation to answer support questions."
    ),
)


def create_agent(
    provider: str = DEFAULT_PROVIDER,
    model_id: str = DEFAULT_MODEL_ID,
    base_url: Optional[str] = DEFAULT_LLM_BASE_URL,
    api_key: Optional[str] = DEFAULT_LLM_API_KEY,
    mcp_url: str = DEFAULT_MCP_URL,
    mcp_config: str = DEFAULT_MCP_CONFIG,
    skills_directory: Optional[str] = DEFAULT_SKILLS_DIRECTORY,
    ssl_verify: bool = DEFAULT_SSL_VERIFY,
) -> Agent:
    """
    Creates the Supervisor Agent with sub-agents registered as tools.
    """
    logger.info("Initializing Multi-Agent System for GitHub...")

    model = create_model(
        provider=provider,
        model_id=model_id,
        base_url=base_url,
        api_key=api_key,
        ssl_verify=ssl_verify,
        timeout=DEFAULT_TIMEOUT,
    )
    settings = ModelSettings(
        max_tokens=DEFAULT_MAX_TOKENS,
        temperature=DEFAULT_TEMPERATURE,
        top_p=DEFAULT_TOP_P,
        timeout=DEFAULT_TIMEOUT,
        parallel_tool_calls=DEFAULT_PARALLEL_TOOL_CALLS,
        seed=DEFAULT_SEED,
        presence_penalty=DEFAULT_PRESENCE_PENALTY,
        frequency_penalty=DEFAULT_FREQUENCY_PENALTY,
        logit_bias=DEFAULT_LOGIT_BIAS,
        stop_sequences=DEFAULT_STOP_SEQUENCES,
        extra_headers=DEFAULT_EXTRA_HEADERS,
        extra_body=DEFAULT_EXTRA_BODY,
    )

    agent_toolsets = []
    if mcp_url:
        if "sse" in mcp_url.lower():
            server = MCPServerSSE(
                mcp_url,
                http_client=httpx.AsyncClient(
                    verify=ssl_verify, timeout=DEFAULT_TIMEOUT
                ),
            )
        else:
            server = MCPServerStreamableHTTP(
                mcp_url,
                http_client=httpx.AsyncClient(
                    verify=ssl_verify, timeout=DEFAULT_TIMEOUT
                ),
            )
        agent_toolsets.append(server)
        logger.info(f"Connected to MCP Server: {mcp_url}")
    elif mcp_config:
        mcp_toolset = load_mcp_servers(mcp_config)
        for server in mcp_toolset:
            if hasattr(server, "http_client"):
                server.http_client = httpx.AsyncClient(
                    verify=ssl_verify, timeout=DEFAULT_TIMEOUT
                )
        agent_toolsets.extend(mcp_toolset)
        logger.info(f"Connected to MCP Config JSON: {mcp_toolset}")

    if skills_directory and os.path.exists(skills_directory):
        agent_toolsets.append(SkillsToolset(directories=[str(skills_directory)]))

    agent_defs = {
        "person": (CONTEXT_AGENT_PROMPT, "GitHub_Context_Agent"),
        "workflow": (ACTIONS_AGENT_PROMPT, "GitHub_Actions_Agent"),
        "codescan": (CODE_SECURITY_AGENT_PROMPT, "GitHub_Code_Security_Agent"),
        "dependabot": (DEPENDABOT_AGENT_PROMPT, "GitHub_Dependabot_Agent"),
        "comment-discussion": (DISCUSSIONS_AGENT_PROMPT, "GitHub_Discussions_Agent"),
        "logo-gist": (GISTS_AGENT_PROMPT, "GitHub_Gists_Agent"),
        "git-branch": (GIT_AGENT_PROMPT, "GitHub_Git_Agent"),
        "issue-opened": (ISSUES_AGENT_PROMPT, "GitHub_Issues_Agent"),
        "tag": (LABELS_AGENT_PROMPT, "GitHub_Labels_Agent"),
        "bell": (NOTIFICATIONS_AGENT_PROMPT, "GitHub_Notifications_Agent"),
        "organization": (ORGS_AGENT_PROMPT, "GitHub_Organizations_Agent"),
        "project": (PROJECTS_AGENT_PROMPT, "GitHub_Projects_Agent"),
        "git-pull-request": (PULL_REQUESTS_AGENT_PROMPT, "GitHub_Pull_Requests_Agent"),
        "repo": (REPOS_AGENT_PROMPT, "GitHub_Repos_Agent"),
        "shield-lock": (
            SECRET_PROTECTION_AGENT_PROMPT,
            "GitHub_Secret_Protection_Agent",
        ),
        "shield": (
            SECURITY_ADVISORIES_AGENT_PROMPT,
            "GitHub_Security_Advisories_Agent",
        ),
        "star": (STARGAZERS_AGENT_PROMPT, "GitHub_Stargazers_Agent"),
        "people": (USERS_AGENT_PROMPT, "GitHub_Users_Agent"),
        "copilot": (COPILOT_AGENT_PROMPT, "GitHub_Copilot_Agent"),
        "copilot_spaces": (COPILOT_SPACES_AGENT_PROMPT, "GitHub_Copilot_Spaces_Agent"),
        "github_support_docs_search": (
            SUPPORT_DOCS_AGENT_PROMPT,
            "GitHub_Support_Docs_Agent",
        ),
    }

    child_agents = {}

    for tag, (system_prompt, agent_name) in agent_defs.items():
        tag_toolsets = []
        for ts in agent_toolsets:

            def filter_func(ctx, tool_def, t=tag):
                return tool_in_tag(tool_def, t)

            if hasattr(ts, "filtered"):
                filtered_ts = ts.filtered(filter_func)
                tag_toolsets.append(filtered_ts)
            else:
                pass

        # Collect tool names for logging
        all_tool_names = []
        for ts in tag_toolsets:
            try:
                # Unwrap FilteredToolset
                current_ts = ts
                while hasattr(current_ts, "wrapped"):
                    current_ts = current_ts.wrapped

                # Check for .tools (e.g. SkillsToolset)
                if hasattr(current_ts, "tools") and isinstance(current_ts.tools, dict):
                    all_tool_names.extend(current_ts.tools.keys())
                # Check for ._tools (some implementations might use private attr)
                elif hasattr(current_ts, "_tools") and isinstance(
                    current_ts._tools, dict
                ):
                    all_tool_names.extend(current_ts._tools.keys())
                else:
                    # Fallback for MCP or others where tools are not available sync
                    all_tool_names.append(f"<{type(current_ts).__name__}>")
            except Exception as e:
                logger.info(f"Unable to retrieve toolset: {e}")
                pass

        tool_list_str = ", ".join(all_tool_names)
        logger.info(f"Available tools for {agent_name} ({tag}): {tool_list_str}")
        agent = Agent(
            model=model,
            system_prompt=system_prompt,
            name=agent_name,
            toolsets=tag_toolsets,
            tool_timeout=DEFAULT_TOOL_TIMEOUT,
            model_settings=settings,
        )
        child_agents[tag] = agent

    supervisor = Agent(
        name=AGENT_NAME,
        system_prompt=SUPERVISOR_SYSTEM_PROMPT,
        model=model,
        model_settings=settings,
        deps_type=Any,
    )

    @supervisor.tool
    async def assign_task_to_context_agent(ctx: RunContext[Any], task: str) -> str:
        """Assign a task related to user context and general GitHub status to the Context Agent."""
        try:
            return (
                await child_agents["person"].run(task, usage=ctx.usage, deps=ctx.deps)
            ).output
        except Exception as e:
            logger.exception(f"Error in Context Agent: {e}")
            return f"Error executing task for Context Agent: {e}"

    @supervisor.tool
    async def assign_task_to_actions_agent(ctx: RunContext[Any], task: str) -> str:
        """Assign a task related to GitHub Actions and Workflows to the Actions Agent."""
        try:
            return (
                await child_agents["workflow"].run(task, usage=ctx.usage, deps=ctx.deps)
            ).output
        except Exception as e:
            logger.exception(f"Error in Actions Agent: {e}")
            return f"Error executing task for Actions Agent: {e}"

    @supervisor.tool
    async def assign_task_to_code_security_agent(
        ctx: RunContext[Any], task: str
    ) -> str:
        """Assign a task related to code security and scanning to the Code Security Agent."""
        try:
            return (
                await child_agents["codescan"].run(task, usage=ctx.usage, deps=ctx.deps)
            ).output
        except Exception as e:
            logger.exception(f"Error in Code Security Agent: {e}")
            return f"Error executing task for Code Security Agent: {e}"

    @supervisor.tool
    async def assign_task_to_dependabot_agent(ctx: RunContext[Any], task: str) -> str:
        """Assign a task related to Dependabot to the Dependabot Agent."""
        try:
            return (
                await child_agents["dependabot"].run(
                    task, usage=ctx.usage, deps=ctx.deps
                )
            ).output
        except Exception as e:
            logger.exception(f"Error in Dependabot Agent: {e}")
            return f"Error executing task for Dependabot Agent: {e}"

    @supervisor.tool
    async def assign_task_to_discussions_agent(ctx: RunContext[Any], task: str) -> str:
        """Assign a task related to GitHub Discussions to the Discussions Agent."""
        try:
            return (
                await child_agents["comment-discussion"].run(
                    task, usage=ctx.usage, deps=ctx.deps
                )
            ).output
        except Exception as e:
            logger.exception(f"Error in Discussions Agent: {e}")
            return f"Error executing task for Discussions Agent: {e}"

    @supervisor.tool
    async def assign_task_to_gists_agent(ctx: RunContext[Any], task: str) -> str:
        """Assign a task related to Gists to the Gists Agent."""
        try:
            return (
                await child_agents["logo-gist"].run(
                    task, usage=ctx.usage, deps=ctx.deps
                )
            ).output
        except Exception as e:
            logger.exception(f"Error in Gists Agent: {e}")
            return f"Error executing task for Gists Agent: {e}"

    @supervisor.tool
    async def assign_task_to_git_agent(ctx: RunContext[Any], task: str) -> str:
        """Assign a task related to low-level Git operations (refs, blobs) to the Git Agent."""
        try:
            return (
                await child_agents["git-branch"].run(
                    task, usage=ctx.usage, deps=ctx.deps
                )
            ).output
        except Exception as e:
            logger.exception(f"Error in Git Agent: {e}")
            return f"Error executing task for Git Agent: {e}"

    @supervisor.tool
    async def assign_task_to_issues_agent(ctx: RunContext[Any], task: str) -> str:
        """Assign a task related to Issues (create, list, comment) to the Issues Agent."""
        try:
            return (
                await child_agents["issue-opened"].run(
                    task, usage=ctx.usage, deps=ctx.deps
                )
            ).output
        except Exception as e:
            logger.exception(f"Error in Issues Agent: {e}")
            return f"Error executing task for Issues Agent: {e}"

    @supervisor.tool
    async def assign_task_to_labels_agent(ctx: RunContext[Any], task: str) -> str:
        """Assign a task related to Labels to the Labels Agent."""
        try:
            return (
                await child_agents["tag"].run(task, usage=ctx.usage, deps=ctx.deps)
            ).output
        except Exception as e:
            logger.exception(f"Error in Labels Agent: {e}")
            return f"Error executing task for Labels Agent: {e}"

    @supervisor.tool
    async def assign_task_to_notifications_agent(
        ctx: RunContext[Any], task: str
    ) -> str:
        """Assign a task related to Notifications to the Notifications Agent."""
        try:
            return (
                await child_agents["bell"].run(task, usage=ctx.usage, deps=ctx.deps)
            ).output
        except Exception as e:
            logger.exception(f"Error in Notifications Agent: {e}")
            return f"Error executing task for Notifications Agent: {e}"

    @supervisor.tool
    async def assign_task_to_organizations_agent(
        ctx: RunContext[Any], task: str
    ) -> str:
        """Assign a task related to Organizations to the Organizations Agent."""
        try:
            return (
                await child_agents["organization"].run(
                    task, usage=ctx.usage, deps=ctx.deps
                )
            ).output
        except Exception as e:
            logger.exception(f"Error in Organizations Agent: {e}")
            return f"Error executing task for Organizations Agent: {e}"

    @supervisor.tool
    async def assign_task_to_projects_agent(ctx: RunContext[Any], task: str) -> str:
        """Assign a task related to GitHub Projects to the Projects Agent."""
        try:
            return (
                await child_agents["project"].run(task, usage=ctx.usage, deps=ctx.deps)
            ).output
        except Exception as e:
            logger.exception(f"Error in Projects Agent: {e}")
            return f"Error executing task for Projects Agent: {e}"

    @supervisor.tool
    async def assign_task_to_pull_requests_agent(
        ctx: RunContext[Any], task: str
    ) -> str:
        """Assign a task related to Pull Requests to the Pull Requests Agent."""
        try:
            return (
                await child_agents["git-pull-request"].run(
                    task, usage=ctx.usage, deps=ctx.deps
                )
            ).output
        except Exception as e:
            logger.exception(f"Error in Pull Requests Agent: {e}")
            return f"Error executing task for Pull Requests Agent: {e}"

    @supervisor.tool
    async def assign_task_to_repos_agent(ctx: RunContext[Any], task: str) -> str:
        """Assign a task related to Repositories (list, settings, delete) to the Repositories Agent."""
        try:
            return (
                await child_agents["repo"].run(task, usage=ctx.usage, deps=ctx.deps)
            ).output
        except Exception as e:
            logger.exception(f"Error in Repositories Agent: {e}")
            return f"Error executing task for Repositories Agent: {e}"

    @supervisor.tool
    async def assign_task_to_secret_protection_agent(
        ctx: RunContext[Any], task: str
    ) -> str:
        """Assign a task related to Secret Protection to the Secret Protection Agent."""
        try:
            return (
                await child_agents["shield-lock"].run(
                    task, usage=ctx.usage, deps=ctx.deps
                )
            ).output
        except Exception as e:
            logger.exception(f"Error in Secret Protection Agent: {e}")
            return f"Error executing task for Secret Protection Agent: {e}"

    @supervisor.tool
    async def assign_task_to_security_advisories_agent(
        ctx: RunContext[Any], task: str
    ) -> str:
        """Assign a task related to Security Advisories to the Security Advisories Agent."""
        try:
            return (
                await child_agents["shield"].run(task, usage=ctx.usage, deps=ctx.deps)
            ).output
        except Exception as e:
            logger.exception(f"Error in Security Advisories Agent: {e}")
            return f"Error executing task for Security Advisories Agent: {e}"

    @supervisor.tool
    async def assign_task_to_stargazers_agent(ctx: RunContext[Any], task: str) -> str:
        """Assign a task related to Stargazers to the Stargazers Agent."""
        try:
            return (
                await child_agents["star"].run(task, usage=ctx.usage, deps=ctx.deps)
            ).output
        except Exception as e:
            logger.exception(f"Error in Stargazers Agent: {e}")
            return f"Error executing task for Stargazers Agent: {e}"

    @supervisor.tool
    async def assign_task_to_users_agent(ctx: RunContext[Any], task: str) -> str:
        """Assign a task related to Users to the Users Agent."""
        try:
            return (
                await child_agents["people"].run(task, usage=ctx.usage, deps=ctx.deps)
            ).output
        except Exception as e:
            logger.exception(f"Error in Users Agent: {e}")
            return f"Error executing task for Users Agent: {e}"

    @supervisor.tool
    async def assign_task_to_copilot_agent(ctx: RunContext[Any], task: str) -> str:
        """Assign a task related to GitHub Copilot coding tasks to the Copilot Agent."""
        try:
            return (
                await child_agents["copilot"].run(task, usage=ctx.usage, deps=ctx.deps)
            ).output
        except Exception as e:
            logger.exception(f"Error in Copilot Agent: {e}")
            return f"Error executing task for Copilot Agent: {e}"

    @supervisor.tool
    async def assign_task_to_copilot_spaces_agent(
        ctx: RunContext[Any], task: str
    ) -> str:
        """Assign a task related to Copilot Spaces to the Copilot Spaces Agent."""
        try:
            return (
                await child_agents["copilot_spaces"].run(
                    task, usage=ctx.usage, deps=ctx.deps
                )
            ).output
        except Exception as e:
            logger.exception(f"Error in Copilot Spaces Agent: {e}")
            return f"Error executing task for Copilot Spaces Agent: {e}"

    @supervisor.tool
    async def assign_task_to_support_docs_agent(ctx: RunContext[Any], task: str) -> str:
        """Assign a task to search GitHub Support Docs to the Support Docs Agent."""
        try:
            return (
                await child_agents["github_support_docs_search"].run(
                    task, usage=ctx.usage, deps=ctx.deps
                )
            ).output
        except Exception as e:
            logger.exception(f"Error in Support Docs Agent: {e}")
            return f"Error executing task for Support Docs Agent: {e}"

    return supervisor


def create_agent_server(
    provider: str = DEFAULT_PROVIDER,
    model_id: str = DEFAULT_MODEL_ID,
    base_url: Optional[str] = DEFAULT_LLM_BASE_URL,
    api_key: Optional[str] = DEFAULT_LLM_API_KEY,
    mcp_url: str = DEFAULT_MCP_URL,
    mcp_config: str = DEFAULT_MCP_CONFIG,
    skills_directory: Optional[str] = DEFAULT_SKILLS_DIRECTORY,
    debug: Optional[bool] = DEFAULT_DEBUG,
    host: Optional[str] = DEFAULT_HOST,
    port: Optional[int] = DEFAULT_PORT,
    enable_web_ui: bool = DEFAULT_ENABLE_WEB_UI,
    ssl_verify: bool = DEFAULT_SSL_VERIFY,
):
    print(
        f"Starting {AGENT_NAME}:"
        f"\tprovider={provider}"
        f"\tmodel={model_id}"
        f"\tbase_url={base_url}"
        f"\tmcp={mcp_url} | {mcp_config}"
        f"\tssl_verify={ssl_verify}"
    )
    agent = create_agent(
        provider=provider,
        model_id=model_id,
        base_url=base_url,
        api_key=api_key,
        mcp_url=mcp_url,
        mcp_config=mcp_config,
        skills_directory=skills_directory,
        ssl_verify=ssl_verify,
    )

    if skills_directory and os.path.exists(skills_directory):
        skills = load_skills_from_directory(skills_directory)
        logger.info(f"Loaded {len(skills)} skills from {skills_directory}")
    else:
        skills = [
            Skill(
                id="github_agent",
                name="GitHub Agent",
                description="General access to GitHub tools",
                tags=["github"],
                input_modes=["text"],
                output_modes=["text"],
            )
        ]

    a2a_app = agent.to_a2a(
        name=AGENT_NAME,
        description=AGENT_DESCRIPTION,
        version=__version__,
        skills=skills,
        debug=debug,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if hasattr(a2a_app, "router") and hasattr(a2a_app.router, "lifespan_context"):
            async with a2a_app.router.lifespan_context(a2a_app):
                yield
        else:
            yield

    app = FastAPI(
        title=f"{AGENT_NAME} - A2A + AG-UI Server",
        description=AGENT_DESCRIPTION,
        debug=debug,
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health_check():
        return {"status": "OK"}

    app.mount("/a2a", a2a_app)

    @app.post("/ag-ui")
    async def ag_ui_endpoint(request: Request) -> Response:
        accept = request.headers.get("accept", SSE_CONTENT_TYPE)
        try:
            run_input = AGUIAdapter.build_run_input(await request.body())
        except ValidationError as e:
            return Response(
                content=json.dumps(e.json()),
                media_type="application/json",
                status_code=422,
            )

        if hasattr(run_input, "messages"):
            run_input.messages = prune_large_messages(run_input.messages)

        adapter = AGUIAdapter(agent=agent, run_input=run_input, accept=accept)
        event_stream = adapter.run_stream()
        sse_stream = adapter.encode_stream(event_stream)

        return StreamingResponse(
            sse_stream,
            media_type=accept,
        )

    if enable_web_ui:
        web_ui = agent.to_web(instructions=SUPERVISOR_SYSTEM_PROMPT)
        app.mount("/", web_ui)
        logger.info(
            "Starting server on %s:%s (A2A at /a2a, AG-UI at /ag-ui, Web UI: %s)",
            host,
            port,
            "Enabled at /" if enable_web_ui else "Disabled",
        )

    uvicorn.run(
        app,
        host=host,
        port=port,
        timeout_keep_alive=1800,
        timeout_graceful_shutdown=60,
        log_level="debug" if debug else "info",
    )


def agent_server():
    print(f"github_agent v{__version__}")
    parser = argparse.ArgumentParser(
        add_help=False, description=f"Run the {AGENT_NAME} A2A + AG-UI Server"
    )
    parser.add_argument(
        "--host", default=DEFAULT_HOST, help="Host to bind the server to"
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="Port to bind the server to"
    )
    parser.add_argument("--debug", type=bool, default=DEFAULT_DEBUG, help="Debug mode")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    parser.add_argument(
        "--provider",
        default=DEFAULT_PROVIDER,
        choices=["openai", "anthropic", "google", "huggingface"],
        help="LLM Provider",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="LLM Model ID")
    parser.add_argument(
        "--base-url",
        default=DEFAULT_LLM_BASE_URL,
        help="LLM Base URL (for OpenAI compatible providers)",
    )
    parser.add_argument("--api-key", default=DEFAULT_LLM_API_KEY, help="LLM API Key")
    parser.add_argument("--mcp-url", default=DEFAULT_MCP_URL, help="MCP Server URL")
    parser.add_argument(
        "--mcp-config", default=DEFAULT_MCP_CONFIG, help="MCP Server Config"
    )
    parser.add_argument(
        "--skills-directory",
        default=DEFAULT_SKILLS_DIRECTORY,
        help="Directory containing agent skills",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        default=DEFAULT_ENABLE_WEB_UI,
        help="Enable Pydantic AI Web UI",
    )

    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable SSL verification for LLM requests (Use with caution)",
    )
    parser.add_argument("--help", action="store_true", help="Show usage")

    args = parser.parse_args()

    if hasattr(args, "help") and args.help:

        usage()

        sys.exit(0)

    if args.debug:
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler()],
            force=True,
        )
        logging.getLogger("pydantic_ai").setLevel(logging.DEBUG)
        logging.getLogger("fastmcp").setLevel(logging.DEBUG)
        logging.getLogger("httpcore").setLevel(logging.DEBUG)
        logging.getLogger("httpx").setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")

    create_agent_server(
        provider=args.provider,
        model_id=args.model_id,
        base_url=args.base_url,
        api_key=args.api_key,
        mcp_url=args.mcp_url,
        mcp_config=args.mcp_config,
        skills_directory=args.skills_directory,
        debug=args.debug,
        host=args.host,
        port=args.port,
        enable_web_ui=args.web,
        ssl_verify=not args.insecure,
    )


def usage():
    print(
        f"Github Agent ({__version__}): CLI Tool\n\n"
        "Usage:\n"
        "--host                [ Host to bind the server to ]\n"
        "--port                [ Port to bind the server to ]\n"
        "--debug               [ Debug mode ]\n"
        "--reload              [ Enable auto-reload ]\n"
        "--provider            [ LLM Provider ]\n"
        "--model-id            [ LLM Model ID ]\n"
        "--base-url            [ LLM Base URL (for OpenAI compatible providers) ]\n"
        "--api-key             [ LLM API Key ]\n"
        "--mcp-url             [ MCP Server URL ]\n"
        "--mcp-config          [ MCP Server Config ]\n"
        "--skills-directory    [ Directory containing agent skills ]\n"
        "--web                 [ Enable Pydantic AI Web UI ]\n"
        "\n"
        "Examples:\n"
        "  [Simple]  github-agent \n"
        '  [Complex] github-agent --host "value" --port "value" --debug "value" --reload --provider "value" --model-id "value" --base-url "value" --api-key "value" --mcp-url "value" --mcp-config "value" --skills-directory "value" --web\n'
    )


if __name__ == "__main__":
    agent_server()

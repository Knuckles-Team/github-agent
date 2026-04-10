# MCP_AGENTS.md - Dynamic Agent Registry

This file tracks the generated agents from MCP servers. You can manually modify the 'Tools' list to customize agent expertise.

## Agent Mapping Table

| Name | Description | System Prompt | Tools | Tag | Source MCP |
|------|-------------|---------------|-------|-----|------------|
| Github Contents Specialist | Expert specialist for contents domain tasks. | You are a Github Contents specialist. Help users manage and interact with Contents functionality using the available tools. | github-mcp_contents_toolset | contents | github-mcp |
| Github Issue Specialist | Expert specialist for issue domain tasks. | You are a Github Issue specialist. Help users manage and interact with Issue functionality using the available tools. | github-mcp_issue_toolset | issue | github-mcp |
| Github Pulls Specialist | Expert specialist for pulls domain tasks. | You are a Github Pulls specialist. Help users manage and interact with Pulls functionality using the available tools. | github-mcp_pulls_toolset | pulls | github-mcp |
| Github Repos Specialist | Expert specialist for repos domain tasks. | You are a Github Repos specialist. Help users manage and interact with Repos functionality using the available tools. | github-mcp_repos_toolset | repos | github-mcp |

## Tool Inventory Table

| Tool Name | Description | Tag | Source |
|-----------|-------------|-----|--------|
| github-mcp_contents_toolset | Static hint toolset for contents based on config env. | contents | github-mcp |
| github-mcp_issue_toolset | Static hint toolset for issue based on config env. | issue | github-mcp |
| github-mcp_pulls_toolset | Static hint toolset for pulls based on config env. | pulls | github-mcp |
| github-mcp_repos_toolset | Static hint toolset for repos based on config env. | repos | github-mcp |

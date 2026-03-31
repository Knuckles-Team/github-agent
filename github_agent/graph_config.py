"""GitHub graph configuration — tag prompts and env var mappings."""

                                                                       
TAG_PROMPTS: dict[str, str] = {
    "repos": (
        "You are a GitHub Repositories specialist. Help users manage and interact with Repository functionality using the available tools."
    ),
    "issues": (
        "You are a GitHub Issues specialist. Help users manage and interact with Issues functionality using the available tools."
    ),
    "pulls": (
        "You are a GitHub Pull Requests specialist. Help users manage and interact with Pull Request functionality using the available tools."
    ),
    "contents": (
        "You are a GitHub Contents specialist. Help users manage and interact with Repository Content (files/directories) using the available tools."
    ),
}

                                                                        
TAG_ENV_VARS: dict[str, str] = {
    "repos": "REPOSTOOL",
    "issues": "ISSUETOOL",
    "pulls": "PULLSTOOL",
    "contents": "CONTENTSTOOL",
}

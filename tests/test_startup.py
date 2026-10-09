import runpy
import sys
from unittest.mock import patch


def test_server_startup():
    """Validates that __main__ delegates to the MCP server."""
    with patch("github_agent.mcp_server.mcp_server") as mock_mcp_server:
        test_args = ["__main__.py"]
        with patch.object(sys, "argv", test_args):
            runpy.run_module("github_agent.__main__", run_name="__main__")

        mock_mcp_server.assert_called_once()


def test_mcp_server_main_startup():
    """Validates that the mcp_server module can run under __main__."""
    # runpy.run_module(..., run_name="__main__") re-executes the module's
    # own argparse against the LIVE sys.argv, so any pytest flag (e.g.
    # `-p no:randomly`) leaks in and is rejected by the module's CLI
    # parser. Pin argv to a clean single-element list for the duration of
    # the run so the module under test sees the same argv regardless of how
    # pytest was invoked (mirrors the pattern already used above for
    # `__main__`).
    with (
        patch("fastmcp.FastMCP.run") as mock_run,
        patch.object(sys, "argv", ["mcp_server.py"]),
    ):
        runpy.run_module("github_agent.mcp_server", run_name="__main__")
        mock_run.assert_called_once_with(transport="stdio")

import sys
import runpy
from unittest.mock import patch, MagicMock


def test_server_startup():
    """Validates that the server module and __main__ can run successfully."""
    with (
        patch("agent_utilities.create_agent_server") as mock_create,
        patch("agent_utilities.initialize_workspace") as mock_init,
        patch("agent_utilities.load_identity") as mock_load,
    ):
        mock_load.return_value = {
            "name": "Test Github Agent",
            "description": "Test description",
        }

        # Test agent_server with --debug
        test_args = ["agent_server.py", "--debug"]
        with patch.object(sys, "argv", test_args):
            runpy.run_module("github_agent.agent_server", run_name="__main__")

        assert mock_create.called

        # Test __main__ execution
        mock_create.reset_mock()
        test_args = ["__main__.py"]
        with patch.object(sys, "argv", test_args):
            runpy.run_module("github_agent.__main__", run_name="__main__")

        assert mock_create.called


def test_mcp_server_main_startup():
    """Validates that the mcp_server module can run under __main__."""
    with patch("fastmcp.FastMCP.run") as mock_run:
        runpy.run_module("github_agent.mcp_server", run_name="__main__")
        mock_run.assert_called_once_with(transport="stdio")

# CONN-GITHUB-AGENT-CONNECTOR-CONTENT quickstart

Run the served proof from a fresh checkout after `uv sync`:

- `PYTHONNOUSERSITE=1 uv run pytest tests/test_change_proposal_served.py::test_served_tools_and_normalizer_round_trip -q`
- `PYTHONNOUSERSITE=1 uv run pytest tests/test_change_proposal_tools.py -q`
- `PYTHONNOUSERSITE=1 uv run pytest tests/test_repository_effect_served.py::test_tool_applies_one_conditional_file_effect -q`

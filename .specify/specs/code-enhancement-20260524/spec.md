# Code Enhancement: github-agent

> Automated code enhancement review for github-agent. Covers 17 analysis domains.

## User Stories

- As a **developer**, I want to **address Project Analysis findings (grade: C, score: 74)**, so that **improve project project analysis from C to at least B (80+)**.
- As a **developer**, I want to **address Codebase Optimization findings (grade: F, score: 56)**, so that **improve project codebase optimization from F to at least B (80+)**.
- As a **developer**, I want to **address Test Coverage findings (grade: C, score: 75)**, so that **improve project test coverage from C to at least B (80+)**.
- As a **developer**, I want to **address Architecture & Design Patterns findings (grade: C, score: 70)**, so that **improve project architecture & design patterns from C to at least B (80+)**.
- As a **developer**, I want to **address Concept Traceability findings (grade: F, score: 25)**, so that **improve project concept traceability from F to at least B (80+)**.
- As a **developer**, I want to **address Test Execution findings (grade: F, score: 25)**, so that **improve project test execution from F to at least B (80+)**.
- As a **developer**, I want to **address Changelog Audit findings (grade: C, score: 75)**, so that **improve project changelog audit from C to at least B (80+)**.
- As a **developer**, I want to **address Pytest Quality findings (grade: C, score: 71)**, so that **improve project pytest quality from C to at least B (80+)**.
- As a **developer**, I want to **address Environment Variables findings (grade: D, score: 60)**, so that **improve project environment variables from D to at least B (80+)**.
- As a **developer**, I want to **address analyze_xdg_kg findings (grade: F, score: 0)**, so that **improve project analyze_xdg_kg from F to at least B (80+)**.

## Functional Requirements

- **FR-001**: Minor update: pytest-xdist 3.6.0 (constraint — not installed) -> 3.8.0
- **FR-002**: Minor update: agent-utilities 0.2.40 (installed) -> 0.16.0
- **FR-003**: Moderate avg cyclomatic complexity: 7.4
- **FR-004**: 61 functions exceed 50 lines
- **FR-005**: Monolithic: mcp_server.py (1251L) — 7 functions with high complexity (worst: register_action_tools at 153L, CC=32); Low cohesion: 14 distinct concepts in one file
- **FR-006**: 46 functions with nesting depth >4
- **FR-007**: Test suite lacks intent diversity (only one type)
- **FR-008**: 13 potential doc-test drift items
- **FR-009**: README.md missing sections: usage|quick start
- **FR-010**: 2 broken internal links in README.md
- **FR-011**: README missing: Has a Table of Contents
- **FR-012**: README missing: Has usage examples with code blocks
- **FR-013**: SRP: 2 modules exceed 500 lines (god modules)
- **FR-014**: No discernible layer architecture (no domain/service/adapter separation)
- **FR-015**: Low dependency injection ratio: 2%
- **FR-016**: Low traceability ratio: 0% concepts fully traced
- **FR-017**: 19 orphaned concepts (only in one source)
- **FR-018**: 56 test functions missing concept markers
- **FR-019**: 104 significant functions (>10 lines) missing concept markers in docstrings
- **FR-020**: Total lint findings: 0 (high/error: 0, medium/warning: 0, low: 0)
- **FR-021**: 2 hook(s) may be outdated: ruff-pre-commit, uv-pre-commit
- **FR-022**: 1 rogue/throwaway scripts detected (fix_*, validate_*, patch_*, etc.): scripts/validate_a2a_agent.py
- **FR-023**: CHANGELOG.md exists but could not be parsed — check format compliance
- **FR-024**: No changelog entries within the last 30 days
- **FR-025**: keepachangelog not installed — pip install 'universal-skills[code-enhancer]'
- **FR-026**: 1 test files exceed 500 lines — split into focused modules
- **FR-027**: Test directory lacks subdirectory organization (consider unit/, integration/, e2e/)
- **FR-028**: Low fixture usage: only 20% of tests use fixtures
- **FR-029**: No @pytest.mark.parametrize usage — consider data-driven tests
- **FR-030**: 5 tests have no assertions
- **FR-031**: Only 22% of env vars documented in README.md
- **FR-032**: Undocumented env vars: ACTIONSTOOL, ALLOWED_CLIENT_REDIRECT_URIS, AUTH_TYPE, BRANCHESTOOL, COLLABORATORSTOOL, COMMITSTOOL, CONTENTSTOOL, EUNOMIA_POLICY_FILE, EUNOMIA_REMOTE_URL, EUNOMIA_TYPE
- **FR-033**: 6 Python env vars not in .env.example: ACTIONSTOOL, COLLABORATORSTOOL, TLS_PROFILE_REF, ORGSTOOL, RELEASESTOOL
- **FR-034**: Analysis error: No module named 'agent_utilities.knowledge_graph'

## Success Criteria

- Overall GPA: 2.24 → 3.0
- Domains at B or above: 7 → 17
- Actionable findings: 34 → 0

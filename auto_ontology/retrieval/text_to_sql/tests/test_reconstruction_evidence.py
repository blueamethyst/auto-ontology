# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Error analysis sees the caller's evidence so it stops blaming required filters."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from auto_ontology.retrieval.text_to_sql.agents import sql_reconstruction as mod
from auto_ontology.retrieval.text_to_sql.agents.sql_reconstruction import (
    ErrorAnalysis,
    ErrorType,
    SQLReconstructionAgent,
)


def _analyze(evidence: str) -> str:
    seen: list[str] = []

    def fake_invoke(llm: Any, messages: list, model: Any) -> ErrorAnalysis:
        seen.append(messages[0].content)
        return ErrorAnalysis(error_type=ErrorType.FIXABLE, explanation="x")

    state: Any = {"llm": object(), "path_state": {}, "evidence": evidence}
    with patch.object(mod, "invoke_with_structured_output", fake_invoke):
        SQLReconstructionAgent()._analyze_error(state, "q", "error", [])
    return seen[0]


def test_error_analysis_includes_evidence_and_protects_it() -> None:
    prompt = _analyze("Restrict to userno = 122802. Filter rctcd = 0.")
    assert "userno = 122802" in prompt
    assert "never suggest removing or relaxing them" in prompt


def test_error_analysis_without_evidence_is_unchanged() -> None:
    assert "Authoritative evidence" not in _analyze("")

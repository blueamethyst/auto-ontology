# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tables the caller's evidence names are put in scope."""

from __future__ import annotations

from unittest.mock import patch

from auto_ontology.retrieval.text_to_sql.agents import candidates_preparation as mod

EVIDENCE = (
    "Read payment/order facts from wow2rpt.dbo.bf_rctmst_02 (or v_rctmst_02). "
    "Exclude test members: <fact>.userno NOT IN (SELECT ud.userno FROM "
    "wow2rpt.dbo.tb_userdev AS ud). Filter rctcd = 0."
)


def test_table_names_in_evidence_are_resolved_in_the_target_db() -> None:
    catalog = {"bf_rctmst_02": "t1", "v_rctmst_02": "t2", "tb_userdev": "t3"}
    looked_up: list[tuple[str, str | None]] = []

    def find(name: str, db: str | None = None) -> str | None:
        looked_up.append((name, db))
        return catalog.get(name)

    with patch.object(mod, "find_table_id_by_name", find):
        ids = mod.tables_named_in_evidence(EVIDENCE, "wow2rpt")
    assert ids == ["t1", "t2", "t3"]
    # columns and words without "_" are not looked up; every lookup is scoped
    assert all(db == "wow2rpt" for _, db in looked_up)
    assert "rctcd" not in [n for n, _ in looked_up]


def test_no_evidence_or_database_means_no_lookup() -> None:
    with patch.object(mod, "find_table_id_by_name", side_effect=AssertionError):
        assert mod.tables_named_in_evidence("", "wow2rpt") == []
        assert mod.tables_named_in_evidence(EVIDENCE, None) == []

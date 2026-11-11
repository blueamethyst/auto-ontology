# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from auto_ontology.utils.env import positive_int_env


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, 7),
        ("", 7),
        ("  ", 7),
        ("4", 4),
        (" 3 ", 3),
        ("0", 7),
        ("-2", 7),
        ("x", 7),
    ],
)
def test_positive_int_env(
    monkeypatch: pytest.MonkeyPatch, raw: str | None, expected: int
) -> None:
    if raw is None:
        monkeypatch.delenv("AO_TEST_KNOB", raising=False)
    else:
        monkeypatch.setenv("AO_TEST_KNOB", raw)
    assert positive_int_env("AO_TEST_KNOB", 7) == expected

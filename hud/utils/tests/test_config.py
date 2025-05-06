from __future__ import annotations

import pytest

from hud.utils.common import FunctionConfig
from hud.utils.config import (
    _is_list_of_configs,
    _is_valid_python_name,
    _split_and_validate_path,
    _validate_hud_config,
    expand_config,
)


@pytest.mark.parametrize(
    "config, expected",
    [
        ("test", [{"function": "test", "args": [], "id": None}]),
        (("test",), [{"function": "test", "args": [], "id": None}]),
        (
            [FunctionConfig(function="test", args=[])],
            [{"function": "test", "args": [], "id": None}],
        ),
        ({"function": "test", "args": []}, [{"function": "test", "args": [], "id": None}]),
        (
            {"function": "test", "args": ["arg1"]},
            [{"function": "test", "args": ["arg1"], "id": None}],
        ),
        (
            {"function": "test", "args": ["arg1"], "id": "test_id"},
            [{"function": "test", "args": ["arg1"], "id": "test_id"}],
        ),
        (("test", "arg1", "arg2"), [{"function": "test", "args": ["arg1", "arg2"], "id": None}]),
    ],
)
def test_expand_config(config, expected):
    result = expand_config(config)
    assert len(result) == len(expected)
    for i, item in enumerate(result):
        assert item.function == expected[i]["function"]
        assert item.args == expected[i]["args"]
        assert item.id == expected[i]["id"]


@pytest.mark.parametrize(
    "name, expected",
    [
        ("valid_name", True),
        ("ValidName", True),
        ("valid_name_123", True),
        ("_valid_name", True),
        ("123_invalid", False),
        ("invalid-name", False),
        ("", False),
    ],
)
def test_is_valid_python_name(name, expected):
    assert _is_valid_python_name(name) == expected


def test_validate_hud_config_valid():
    config = {"function": "test.func", "args": ["arg1", "arg2"]}
    result = _validate_hud_config(config)
    assert result.function == "test.func"
    assert result.args == ["arg1", "arg2"]
    assert result.id is None

    # Test with single arg (not in a list)
    config = {"function": "test.func", "args": "arg1"}
    result = _validate_hud_config(config)
    assert result.function == "test.func"
    assert result.args == ["arg1"]

    # Test with ID
    config = {"function": "test.func", "args": [], "id": "test_id"}
    result = _validate_hud_config(config)
    assert result.id == "test_id"


def test_validate_hud_config_invalid():
    with pytest.raises(ValueError, match="function must be a string"):
        _validate_hud_config({"args": []})

    with pytest.raises(ValueError, match="function must be a string"):
        _validate_hud_config({"function": 123, "args": []})


def test_split_and_validate_path_valid():
    # none should raise
    _split_and_validate_path("module.submodule.function")
    _split_and_validate_path("function")
    _split_and_validate_path("Module_123.function_456")


def test_split_and_validate_path_invalid():
    with pytest.raises(ValueError, match="Invalid Python identifier in path"):
        _split_and_validate_path("invalid-module.function")


def test_is_list_of_configs():
    valid_list = [
        FunctionConfig(function="test1", args=[]),
        FunctionConfig(function="test2", args=["arg1"]),
    ]
    assert _is_list_of_configs(valid_list) is True

    # Empty list
    assert _is_list_of_configs([]) is True

    # Invalid: not a list
    assert _is_list_of_configs("not_a_list") is False

    # Invalid: list with non-FunctionConfig items
    invalid_list = [FunctionConfig(function="test", args=[]), "not_a_function_config"]
    assert _is_list_of_configs(invalid_list) is False


def test_expand_config_errors():
    with pytest.raises(ValueError):
        empty_tuple = ()
        expand_config(empty_tuple)  # type: ignore

    with pytest.raises(ValueError):
        invalid_tuple = (123, "arg1")
        expand_config(invalid_tuple)  # type: ignore

    with pytest.raises(ValueError, match="Unknown configuration type"):
        invalid_value = 123
        expand_config(invalid_value)  # type: ignore

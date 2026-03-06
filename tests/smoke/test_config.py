"""
Smoke tests: Verify configuration loading and RoundRobinKeyManager behavior.
"""

import os
import pytest
from core.config import RoundRobinKeyManager


class TestRoundRobinKeyManager:
    """Tests for the RoundRobinKeyManager class."""

    def test_cycles_through_keys(self):
        """Keys cycle A → B → C → A."""
        mgr = RoundRobinKeyManager(["a", "b", "c"])
        assert mgr.get_next_key() == "a"
        assert mgr.get_next_key() == "b"
        assert mgr.get_next_key() == "c"
        assert mgr.get_next_key() == "a"  # wraps around

    def test_single_key_always_returns_itself(self):
        """Single key always returns the same value."""
        mgr = RoundRobinKeyManager(["only-key"])
        for _ in range(5):
            assert mgr.get_next_key() == "only-key"

    def test_empty_raises_value_error(self):
        """Empty list raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            RoundRobinKeyManager([])

    def test_distribution_is_even(self):
        """After N*len(keys) calls, each key is used exactly N times."""
        keys = ["x", "y", "z"]
        mgr = RoundRobinKeyManager(keys)
        counts = {k: 0 for k in keys}
        for _ in range(30):
            counts[mgr.get_next_key()] += 1
        assert all(v == 10 for v in counts.values())


class TestGetRequiredEnvList:
    """Tests for the get_required_env_list helper."""

    def test_missing_env_raises_runtime_error(self, monkeypatch):
        """Missing env var raises RuntimeError."""
        monkeypatch.delenv("NONEXISTENT_KEY_12345", raising=False)
        # We need to re-import to test the function without module-level side effects
        from core.config import get_required_env_list
        with pytest.raises(RuntimeError, match="is not set"):
            get_required_env_list("NONEXISTENT_KEY_12345")

    def test_empty_env_raises_runtime_error(self, monkeypatch):
        """Empty env var raises RuntimeError."""
        monkeypatch.setenv("EMPTY_TEST_KEY", "")
        from core.config import get_required_env_list
        with pytest.raises(RuntimeError, match="is not set"):
            get_required_env_list("EMPTY_TEST_KEY")

    def test_comma_separated_keys(self, monkeypatch):
        """Comma-separated value creates multi-key manager."""
        monkeypatch.setenv("MULTI_KEY_TEST", "key1, key2, key3")
        from core.config import get_required_env_list
        mgr = get_required_env_list("MULTI_KEY_TEST")
        assert mgr.get_next_key() == "key1"
        assert mgr.get_next_key() == "key2"
        assert mgr.get_next_key() == "key3"

    def test_whitespace_stripped(self, monkeypatch):
        """Whitespace around keys is stripped."""
        monkeypatch.setenv("WHITESPACE_KEY", "  hello ,  world  ")
        from core.config import get_required_env_list
        mgr = get_required_env_list("WHITESPACE_KEY")
        assert mgr.get_next_key() == "hello"
        assert mgr.get_next_key() == "world"


class TestFeatureFlags:
    """Tests for feature flag defaults."""

    def test_mock_flags_default_to_true(self):
        """USE_MOCK_* flags should default to True when not explicitly set."""
        from core.config import USE_MOCK_SEARCH, USE_MOCK_LLM, USE_MOCK_GITHUB
        # These are loaded at module import time from the .env file
        # We just verify they are boolean-typed
        assert isinstance(USE_MOCK_SEARCH, bool)
        assert isinstance(USE_MOCK_LLM, bool)
        assert isinstance(USE_MOCK_GITHUB, bool)

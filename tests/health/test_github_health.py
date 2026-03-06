"""
Health check: Verify GitHub API authentication and rate limits.
"""

import pytest

pytestmark = pytest.mark.health


class TestGithubHealth:
    """Verify GitHub API connectivity."""

    def test_github_auth(self):
        """Authenticated user login returns a string."""
        from github import Github
        from core.config import GITHUB_ACCESS_TOKENS

        g = Github(GITHUB_ACCESS_TOKENS.get_next_key())
        user = g.get_user()
        assert user.login is not None
        assert isinstance(user.login, str)
        assert len(user.login) > 0

    def test_github_rate_limit_sufficient(self):
        """Remaining rate limit is above 100 (safe to run tools)."""
        from github import Github
        from core.config import GITHUB_ACCESS_TOKENS

        g = Github(GITHUB_ACCESS_TOKENS.get_next_key())
        rate = g.get_rate_limit()
        remaining = rate.core.remaining
        assert remaining > 100, f"GitHub rate limit too low: {remaining} remaining"

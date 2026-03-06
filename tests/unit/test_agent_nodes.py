"""
Unit tests: Agent node contracts.
Verifies each agent node returns the correct state keys using mock mode.
"""

import pytest


class TestResearcherNode:
    """Tests for company_researcher_node."""

    def test_returns_company_summary(self, mock_trecl_state):
        """Output dict has 'company_summary' key."""
        from agents.researcher import company_researcher_node
        result = company_researcher_node(mock_trecl_state)
        assert "company_summary" in result
        assert isinstance(result["company_summary"], str)
        assert len(result["company_summary"]) > 0


class TestJobDecoderNode:
    """Tests for job_decoder_node."""

    def test_returns_company_jobs(self, mock_trecl_state):
        """Output dict has 'company_jobs' key."""
        from agents.job_decoder import job_decoder_node
        result = job_decoder_node(mock_trecl_state)
        assert "company_jobs" in result
        assert isinstance(result["company_jobs"], str)


class TestPainSynthesizerNode:
    """Tests for pain_synthesizer_node."""

    def test_returns_pain_points_and_projects(self, mock_trecl_state):
        """Output has both 'pain_points_ranked' and 'project_ideas' keys."""
        from agents.pain_synthesizer import pain_synthesizer_node
        result = pain_synthesizer_node(mock_trecl_state)
        assert "pain_points_ranked" in result
        assert "project_ideas" in result
        assert isinstance(result["pain_points_ranked"], str)


class TestWriterNode:
    """Tests for cold_email_writer_node."""

    def test_returns_cold_email(self, mock_trecl_state):
        """Output dict has 'cold_email' key."""
        from agents.writer import cold_email_writer_node
        result = cold_email_writer_node(mock_trecl_state)
        assert "cold_email" in result
        assert isinstance(result["cold_email"], str)


class TestOpportunityCuratorNode:
    """Tests for opportunity_curator_node."""

    def test_returns_curated_opportunities(self, mock_trecl_state):
        """Output has 'curated_opportunities' list."""
        from agents.opportunity_curator import opportunity_curator_node
        result = opportunity_curator_node(mock_trecl_state)
        assert "curated_opportunities" in result
        assert isinstance(result["curated_opportunities"], list)

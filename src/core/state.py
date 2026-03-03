"""
State definitions for the Trecl graph.
This module defines the central 'memory' of the multi-agent system.
"""

from typing import TypedDict

class TreclState(TypedDict):
    """
    The shared state dictionary updated by agents throughout the graph execution.
    
    Attributes:
        company_name (str): The name of the company being researched (Initial Input).
        user_domain (str): The user's technical field (e.g. "backend", "frontend").
        user_stack (list[str]): The user's specific technologies (e.g. ["Python", "Go"]).
        company_summary (str): The synthesized research containing 5 core facts.
            Populated by the `company_researcher_node`.
        company_jobs (str): The synthesized open roles and hiring needs.
            Populated by the `job_decoder_node`.
        pain_points_ranked (str): Detailed ranking of company's technical pain points.
            Populated by the `pain_synthesizer_node`.
        project_ideas (str): Specific custom project ideas based on the user's stack.
            Populated by the `pain_synthesizer_node`.
        cold_email (str): The generated targeted cold outreach email.
            Populated by the `cold_email_writer_node`.
    """
    company_name: str
    user_domain: str
    user_stack: list[str]
    company_summary: str
    company_jobs: str
    pain_points_ranked: str
    project_ideas: str
    cold_email: str

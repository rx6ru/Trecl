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
        company_summary (str): The synthesized research containing 5 core facts.
            Populated by the `company_researcher_node`.
        cold_email (str): The generated targeted cold outreach email.
            Populated by the `cold_email_writer_node`.
    """
    company_name: str
    company_summary: str
    cold_email: str

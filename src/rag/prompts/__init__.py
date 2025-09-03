"""Initializes the prompt management system.

This module exports the `PromptRenderer` class and helper functions for easy access
to prompt templating and rendering functionalities throughout the application.
"""

from .prompt_utils import (
    PromptRenderer,
    prompt_renderer,
    render_deal_summary,
    render_contact_engagement,
    render_lead_scoring,
)

__all__ = [
    "PromptRenderer",
    "prompt_renderer",
    "render_deal_summary",
    "render_contact_engagement",
    "render_lead_scoring",
]


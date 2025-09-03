"""
Utility functions for working with prompt templates.
"""
from typing import Dict, Any, Optional, List
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, select_autoescape
import os

class PromptRenderer:
    """Helper class for rendering prompt templates."""
    
    def __init__(self, templates_dir: Optional[str] = None):
        """
        Initialize the prompt renderer.
        
        Args:
            templates_dir: Directory containing prompt templates
        """
        if templates_dir is None:
            templates_dir = str(Path(__file__).parent / "templates")
        
        self.env = Environment(
            loader=FileSystemLoader(templates_dir),
            autoescape=select_autoescape(),
            trim_blocks=True,
            lstrip_blocks=True
        )
    
    def render(
        self, 
        template_name: str, 
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """
        Render a template with the given context.
        
        Args:
            template_name: Name of the template file
            context: Dictionary of variables to pass to the template
            **kwargs: Additional template variables
            
        Returns:
            Rendered template as a string
        """
        if context is None:
            context = {}
        context.update(kwargs)
        
        template = self.env.get_template(template_name)
        return template.render(**context)
    
    def list_templates(self) -> List[str]:
        """List all available templates."""
        return self.env.list_templates()


# Create a default instance
prompt_renderer = PromptRenderer()

# Helper functions for common tasks
def render_deal_summary(deals: List[Dict[str, Any]], **kwargs) -> str:
    """Render a deal summary using the summarize_deals template."""
    return prompt_renderer.render(
        "summarize_deals.jinja2",
        deals=deals,
        **kwargs
    )

def render_contact_engagement(contacts: List[Dict[str, Any]], **kwargs) -> str:
    """Render a contact engagement analysis using the contact_engagement template."""
    return prompt_renderer.render(
        "contact_engagement.jinja2",
        contacts=contacts,
        **kwargs
    )

def render_lead_scoring(leads: List[Dict[str, Any]], **kwargs) -> str:
    """Render a lead scoring analysis using the lead_scoring template."""
    return prompt_renderer.render(
        "lead_scoring.jinja2",
        leads=leads,
        **kwargs
    )

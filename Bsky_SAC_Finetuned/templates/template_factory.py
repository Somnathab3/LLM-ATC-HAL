"""Template factory for creating environment-specific templates"""

from typing import Dict, Type, List
from .base_template import BaseEnvironmentTemplate
from .horizontal_template import HorizontalTemplate
from .vertical_template import VerticalTemplate
from .sector_template import SectorTemplate
from .merge_template import MergeTemplate


class TemplateFactory:
    """Factory for creating environment-specific templates"""
    
    _template_registry: Dict[str, Type[BaseEnvironmentTemplate]] = {
        "HorizontalCREnv-v0": HorizontalTemplate,
        "VerticalCREnv-v0": VerticalTemplate,
        "SectorCREnv-v0": SectorTemplate,
        "MergeEnv-v0": MergeTemplate
    }
    
    @classmethod
    def create_template(cls, environment_name: str) -> BaseEnvironmentTemplate:
        """Create appropriate template for the given environment"""
        template_class = cls._template_registry.get(environment_name, BaseEnvironmentTemplate)
        return template_class(environment_name)
    
    @classmethod
    def register_template(cls, environment_name: str, template_class: Type[BaseEnvironmentTemplate]):
        """Register a new template for an environment"""
        cls._template_registry[environment_name] = template_class
    
    @classmethod
    def get_supported_environments(cls) -> List[str]:
        """Get list of supported environment names"""
        return list(cls._template_registry.keys())

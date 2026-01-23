"""
API configuration utilities for YTL ASR API.
"""

import os
from typing import Tuple


def resolve_api_config(
    env_name: str = "production",
    base_url_override: str | None = None,
) -> Tuple[str, str]:
    """
    Resolve API key and base URL from environment variables.
    
    Args:
        env_name: Environment name ("staging" or "production")
        base_url_override: Optional base URL override
        
    Returns:
        Tuple of (api_key, base_url)
        
    Raises:
        ValueError: If required environment variables are not set
    """
    if base_url_override:
        api_key = os.getenv("ILMU_API_KEY")
        if not api_key:
            raise ValueError("ILMU_API_KEY environment variable not set")
        return api_key, base_url_override

    if env_name == "staging":
        api_key = os.getenv("ILMU_STAGING_API_KEY")
        base_url = os.getenv("ILMU_STAGING_URL")
    elif env_name == "production":
        api_key = os.getenv("ILMU_PRODUCTION_API_KEY")
        base_url = os.getenv("ILMU_PRODUCTION_URL")
    else:
        api_key = os.getenv("ILMU_API_KEY")
        base_url = os.getenv("ILMU_API_BASE_URL", "https://api.ytlailabs.tech/v1")

    if not api_key or not base_url:
        env_label = env_name or "default"
        raise ValueError(
            f"Missing API config for {env_label}. "
            "Set ILMU_STAGING_URL/ILMU_STAGING_API_KEY or "
            "ILMU_PRODUCTION_URL/ILMU_PRODUCTION_API_KEY in .env."
        )

    return api_key, base_url

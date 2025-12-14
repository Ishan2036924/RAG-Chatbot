"""
=============================================================================
Utility Functions
=============================================================================
Helper functions used across the application.
=============================================================================
"""

from config import PREVIEW_TEXT_LENGTH


def truncate_text(text, max_length=PREVIEW_TEXT_LENGTH):
    """
    Shorten text with ellipsis if too long.
    
    Args:
        text: Input string
        max_length: Maximum characters (default from config)
    
    Returns:
        Truncated string with '...' if shortened
    """
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text


def format_percentage(value):
    """
    Convert decimal to percentage string.
    
    Args:
        value: Decimal number (e.g., 0.85)
    
    Returns:
        Formatted string (e.g., "85.0%")
    """
    return f"{value * 100:.1f}%"


def parse_key_value_string(content):
    """
    Parse 'KEY: value' format into dictionary.
    
    Args:
        content: String with lines like "TITLE: My Title"
    
    Returns:
        Dictionary with parsed key-value pairs
    """
    data = {}
    lines = content.strip().split('\n')
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            data[key.strip()] = value.strip()
    return data


def safe_get(dictionary, key, default="N/A"):
    """
    Safely get value from dictionary with default.
    
    Args:
        dictionary: Dict to search
        key: Key to find
        default: Value if key not found
    
    Returns:
        Value or default
    """
    return dictionary.get(key, default)


def validate_api_key(api_key):
    """
    Check if API key exists and has valid format.
    
    Args:
        api_key: OpenAI API key string
    
    Returns:
        Boolean indicating validity
    """
    if not api_key:
        return False
    if not api_key.startswith("sk-"):
        return False
    if len(api_key) < 20:
        return False
    return True
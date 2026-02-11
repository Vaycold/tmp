"""
Text processing utilities.
"""


def tokenize(text: str) -> list[str]:
    """
    Simple tokenization for BM25.
    
    Args:
        text: Input text
        
    Returns:
        List of tokens (lowercased)
    """
    return text.lower().split()
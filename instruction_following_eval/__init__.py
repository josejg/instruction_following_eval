__all__ = ["get_examples", "evaluate_instruction_following"]

from .evaluation import get_examples, evaluate_instruction_following


def ensure_nltk_resource():
    import nltk
    from nltk.tokenize import word_tokenize

    try:
        # Try to tokenize a simple text
        # If 'punkt' is not downloaded, this will raise an exception
        word_tokenize("This is a test sentence.")
    except LookupError:
        # If exception is raised, it means 'punkt' is not available
        # Download 'punkt'
        nltk.download("punkt")


# Ensure NLTK resource is available before proceeding
ensure_nltk_resource()
nltk.download('punkt_tab')

import re

def clean_telegram_text(text: str) -> str:
    """
    Sanitize Telegram message text for LLM/RAG ingestion.
    - Removes emojis and non-standard characters.
    - Preserves Korean, English, numbers, and basic punctuation.
    - Collapses multiple exclamation/question marks and newlines.
    """
    if not text:
        return ""

    # Remove emojis and unusual symbols, keeping standard alphanumeric, ko/en, and basic punctuation
    # \w matches alphanumeric and underscore (includes Korean characters).
    # \s matches whitespace.
    # The literal characters allow standard punctuation and financial symbols.
    cleaned = re.sub(r'[^\w\s.,;:!?\'"$\-%&/()+=]', '', text)

    # Collapse multiple exclamation or question marks into a single one (or two)
    cleaned = re.sub(r'([!?])\1+', r'\1', cleaned)

    # Collapse multiple newlines into a single standard newline
    cleaned = re.sub(r'\n+', '\n', cleaned)

    # Collapse multiple spaces into a single space
    cleaned = re.sub(r' +', ' ', cleaned)

    return cleaned.strip()

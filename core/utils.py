def safe_len(value):
    """Return length if possible."""
    try:
        return len(value)
    except Exception:
        return None

import os
import time
from typing import Iterable, List, Optional, Union
import requests
import config

DEEPL_PRO = "https://api.deepl.com"
DEEPL_FREE = "https://api-free.deepl.com"

class DeepLTranslateError(RuntimeError):
    pass

def translate_deepl(
    text: Union[str, Iterable[str]],
    target_lang: str,
    *,
    source_lang: Optional[str] = None,
    api_key: Optional[str] = None,
    use_free_api: bool = False,
    formality: Optional[str] = None,            # "default" | "more" | "less" (depends on target language)
    glossary_id: Optional[str] = None,          # existing glossary ID in your DeepL account
    split_sentences: Optional[str] = None,      # "0" | "1" | "nonewlines"
    preserve_formatting: Optional[bool] = None, # True/False
    timeout: float = 20.0,
    max_retries: int = 4,
) -> Union[str, List[str]]:
    """
    Translate text using the DeepL API.

    Args:
        text: A string or an iterable of strings to translate.
        target_lang: Target language code (e.g., "EN", "EN-GB", "DE", "FR", "JA").
        source_lang: Optional source language code (e.g., "DE"). If omitted, DeepL will auto-detect.
        api_key: Your DeepL API key. If None, reads from env var DEEPL_API_KEY.
        use_free_api: Set True if your key is for the free plan (api-free.deepl.com).
        formality: Optional formality setting for supported target languages: "default", "more", or "less".
        glossary_id: Optional DeepL glossary ID to apply.
        split_sentences: How to split sentences ("0", "1", "nonewlines"); see DeepL docs.
        preserve_formatting: If True, keep line breaks etc. where possible.
        timeout: Request timeout in seconds.
        max_retries: Retry on 429/5xx with exponential backoff.

    Returns:
        If input was a single string → a single translated string.
        If input was an iterable → list of translated strings in the same order.

    Raises:
        DeepLTranslateError on non-recoverable API errors.
    """
    api_key = api_key or os.getenv("DEEPL_API_KEY")
    if not api_key:
        raise DeepLTranslateError("Missing DeepL API key. Pass api_key=... or set DEEPL_API_KEY env var.")

    base = DEEPL_FREE if use_free_api else DEEPL_PRO
    url = f"{base}/v2/translate"

    # Normalize input to a list of strings (DeepL accepts multiple 'text' fields)
    is_single = isinstance(text, str)
    texts: List[str] = [text] if is_single else list(text)
    if not texts:
        return "" if is_single else []

    # Build request params
    params = {
        "auth_key": api_key,
        "target_lang": target_lang,
    }
    if source_lang:
        params["source_lang"] = source_lang
    if formality:
        params["formality"] = formality
    if glossary_id:
        params["glossary_id"] = glossary_id
    if split_sentences is not None:
        params["split_sentences"] = split_sentences
    if preserve_formatting is not None:
        params["preserve_formatting"] = "1" if preserve_formatting else "0"

    # DeepL accepts multiple text inputs by repeating the "text" param
    data = [("text", t) for t in texts]

    # Simple retry loop for rate limit / transient errors
    backoff = 1.0
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(url, data=data, params=params, timeout=timeout)
        except requests.RequestException as e:
            # Only retry for transient network errors; otherwise raise
            if attempt < max_retries:
                time.sleep(backoff)
                backoff *= 2
                continue
            raise DeepLTranslateError(f"Network error contacting DeepL: {e}") from e

        # Handle rate limiting / transient server errors
        if resp.status_code in (429, 502, 503, 504):
            if attempt < max_retries:
                # Respect Retry-After if provided
                retry_after = resp.headers.get("Retry-After")
                sleep_for = float(retry_after) if retry_after else backoff
                time.sleep(sleep_for)
                backoff = min(backoff * 2, 16)
                continue

        if resp.ok:
            try:
                payload = resp.json()
                translations = [item["text"] for item in payload["translations"]]
            except Exception as e:
                raise DeepLTranslateError(f"Unexpected response format from DeepL: {resp.text[:500]}") from e

            return translations[0] if is_single else translations

        # Non-OK and not retriable
        msg = _format_deepl_error(resp)
        raise DeepLTranslateError(msg)

    # Should not reach here
    raise DeepLTranslateError("Failed to translate after retries.")

def _format_deepl_error(resp: requests.Response) -> str:
    try:
        j = resp.json()
        detail = j.get("message") or j
    except Exception:
        detail = resp.text
    return f"DeepL API error {resp.status_code}: {detail}"



if __name__ == "__main__":  # pragma: no cover
    text = translate_deepl(text="Привет, мир!!", target_lang="EN", use_free_api=True, preserve_formatting=True)

    print(text)
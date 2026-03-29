"""
Converts a page excerpt to plain text file and caches it.
"""
import os
import threading
from pathlib import Path
from scan2latex_entropy import encode_image, chat
from utils import init_openai_client, get_page_id_from_path, get_cache_key, load_cache_json, get_token_logprobs, write_cache_json, convert_all_tif_to_jpg
from loader import load_image

from utils import MODEL, TOP_K

client = init_openai_client()
_cache_lock = threading.Lock()


def flush_cache_updates(cache_updates: dict) -> bool:
    """Merge an in-memory cache_updates dict into cache/cache.json once.

    Intended for use when worker threads accumulate local cache entries and the
    main thread flushes them in a single locked write.
    """
    if not cache_updates:
        return True

    with _cache_lock:
        cache = load_cache_json()
        cache.update(cache_updates)
        return write_cache_json(cache)


def transcribe_with_logprobs(
    image_path,
    top_k: int = 5,
    model: str = MODEL,
    prompt_version: int = 1,
    *,
    cache_snapshot: dict | None = None,
    local_cache: dict | None = None,
    persist_cache: bool = True,
):
    page_id = get_page_id_from_path(image_path)
    cache_key = get_cache_key(page_id, model, top_k, prompt_version)

    # 1) Check worker-local cache first, if provided.
    if local_cache is not None:
        value = local_cache.get(cache_key)
        if value:
            return value["transcript"], value["token_logprobs"]

    # 2) Check read-only snapshot (loaded once by the coordinator).
    if cache_snapshot is not None:
        value = cache_snapshot.get(cache_key)
        if value:
            return value["transcript"], value["token_logprobs"]

    # 3) Fallback: direct on-disk cache check for the simple single-threaded case.
    if cache_snapshot is None:
        with _cache_lock:
            cache = load_cache_json()
            value = cache.get(cache_key)
            if value:
                return value["transcript"], value["token_logprobs"]
    
    encoded_image = encode_image(image_path)

    system_prompt = """ 
    You are an OCR engine for historical English newspapers.
    Your task is to transcribe the provided image exactly as it appears.
    Do not:
    - Correct spelling
    - Expand abbreviations
    - Normalize ligatures
    - Correct punctuation
    - Modernize language
    - Interpret or infer missing content
    Preserve the original text exactly, except where explicitly instructed otherwise.
    Output only the transcribed plain text and nothing else.
    """

    user_text = """
    Transcribe this image of a historical English newspaper excerpt.
    Preserve exactly as shown:
    - Original capitalization
    - Original spelling (including archaic spellings)
    - Original punctuation
    - Special characters (currency symbols, fractions, quotation marks, dashes, apostrophes, etc.)
    - Section dividers exactly as they appear
    Line-break handling rules:
    - If a word is split across lines using a hyphen at the end of a line, remove the hyphen and reconstruct the full word on a single line.
    - Do NOT remove hyphens that are part of compound words within a line.
    If text is unclear, output your best guess based only on visible characters. **Do not** mark uncertainty and **do not** insert placeholders.
    Return only the plain text transcription. Do not add commentary or formatting.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}",
                        "detail": "high",
                    },
                },
            ],
        },
    ]
    resp = chat(messages, client, model, top_k)
    choice = resp.choices[0]

    transcript_text = choice.message.content.strip()
    token_logprobs = get_token_logprobs(choice, top_k)
    cache_value = {"transcript": transcript_text, "token_logprobs": token_logprobs}

    # Worker-local mode: just stash in local_cache and let the caller flush.
    if local_cache is not None:
        local_cache[cache_key] = cache_value
        return transcript_text, token_logprobs

    # Default/simple mode: write through to disk immediately.
    if persist_cache:
        with _cache_lock:
            cache = load_cache_json()
            value = cache.get(cache_key)
            if value:
                # Another worker wrote this entry while we were waiting on API.
                return value["transcript"], value["token_logprobs"]

            cache[cache_key] = cache_value
            successful = write_cache_json(cache)
        if not successful:
            print(f"Transcribed file {page_id} was not written to cache.")
    
    return transcript_text, token_logprobs


if __name__ == "__main__":
    image_paths = load_image()
    for path in image_paths:    
        transcript_text, token_logprobs = transcribe_with_logprobs(path)
    print(transcript_text)
    print(token_logprobs)
    
"""
Converts a page excerpt to plain text file and caches it.
"""

from scan2latex_entropy import encode_image, chat
from utils import (
    init_openai_client,
    get_page_id_from_path,
    get_cache_key,
    load_cache_json,
    get_token_logprobs,
    write_cache_json,
)
from loader import load_image

from utils import MODEL

client = init_openai_client()


def transcribe_with_logprobs(image_path, top_k=5, model=MODEL, prompt_version=1):
    cache = load_cache_json()

    page_id = get_page_id_from_path(image_path)
    cache_key = get_cache_key(page_id, model, top_k, prompt_version)

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

    # Cache it
    cache[cache_key] = {"transcript": transcript_text, "token_logprobs": token_logprobs}
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

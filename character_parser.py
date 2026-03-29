"""
character_parser.py — LLM-based paragraph speaker attribution.

Reads LLM_PROVIDER env var: 'ollama_7b' | 'ollama_14b' | 'claude'
Reads OLLAMA_HOST env var: default 'http://ollama:11434'
Reads CLAUDE_API_KEY env var for Claude mode.
"""
import json
import logging
import os
import re
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Character name deduplication
# ---------------------------------------------------------------------------

# Tokens that are titles/honorifics/connectives — stripped before comparing
_TITLE_TOKENS = {
    'mr', 'mrs', 'miss', 'ms', 'dr', 'prof', 'professor', 'sir', 'lord', 'lady',
    'sergeant', 'sgt', 'captain', 'cap', 'lieutenant', 'lt', 'colonel', 'col',
    'general', 'gen', 'private', 'pvt', 'corporal', 'cpl', 'major', 'admiral',
    'reverend', 'rev', 'father', 'mother', 'brother', 'sister', 'saint', 'st',
    'the', 'of', 'a', 'an', 'and', 'or', 'von', 'van', 'de', 'del', 'der',
    'le', 'la', 'du', 'al',
}


def _name_core_tokens(name: str) -> frozenset:
    """Extract significant tokens from a character name, stripping titles and punctuation."""
    tokens = re.split(r'[\s\-_]+', name.lower())
    tokens = [re.sub(r'[^a-z]', '', t) for t in tokens]
    return frozenset(t for t in tokens if t and t not in _TITLE_TOKENS and len(t) > 1)


def _deduplicate_characters(characters: dict, segments: list) -> tuple[dict, list]:
    """
    Merge character name variants that share core name tokens.

    E.g. "John", "Sergeant John Smith", "Mr. Smith" → single canonical name.
    The canonical name is chosen as whichever variant appears most in segments.

    Returns updated (characters, segments) with canonical names.
    """
    names = [n for n in characters if n != 'NARRATOR']
    if not names:
        return characters, segments

    # Core tokens for each name
    core: dict[str, frozenset] = {n: _name_core_tokens(n) for n in names}

    # Union-Find
    parent = {n: n for n in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Merge names that share at least one core token
    for i, n1 in enumerate(names):
        for n2 in names[i + 1:]:
            if core[n1] and core[n2] and (core[n1] & core[n2]):
                union(n1, n2)

    # Count how many segments each original name appears in
    seg_counts: dict[str, int] = {}
    for seg in segments:
        spk = seg.get('speaker', 'NARRATOR')
        seg_counts[spk] = seg_counts.get(spk, 0) + 1

    # Build groups and pick canonical (most-frequent; tie-break: shortest name)
    groups: dict[str, list] = {}
    for n in names:
        groups.setdefault(find(n), []).append(n)

    alias_to_canonical: dict[str, str] = {'NARRATOR': 'NARRATOR'}
    for group in groups.values():
        canonical = max(group, key=lambda n: (seg_counts.get(n, 0), -len(n)))
        for n in group:
            alias_to_canonical[n] = canonical

    # Rebuild characters dict merging data under canonical name
    new_characters: dict = {'NARRATOR': characters.get('NARRATOR', {})}
    for group in groups.values():
        canonical = alias_to_canonical[group[0]]
        merged: dict = {}
        for n in group:
            merged.update(characters.get(n, {}))
        new_characters[canonical] = merged

    # Remap segments
    new_segments = [
        {**seg, 'speaker': alias_to_canonical.get(seg.get('speaker', 'NARRATOR'), seg.get('speaker', 'NARRATOR'))}
        for seg in segments
    ]

    merged_count = sum(1 for g in groups.values() if len(g) > 1)
    if merged_count:
        logger.info("Deduplicated %d character groups from %d raw names into %d canonical names",
                    merged_count, len(names), len(new_characters) - 1)

    return new_characters, new_segments


def _add_appearance_counts(characters: dict, segments: list) -> dict:
    """Add a 'count' field to each character dict with the number of segments they speak."""
    counts: dict[str, int] = {}
    for seg in segments:
        spk = seg.get('speaker', 'NARRATOR')
        counts[spk] = counts.get(spk, 0) + 1
    return {
        name: {**data, 'count': counts.get(name, 0)}
        for name, data in characters.items()
    }

OLLAMA_HOST = os.environ.get('OLLAMA_HOST', 'http://ollama:11434')

OLLAMA_MODELS = {
    'ollama_7b':  'qwen2.5:7b',
    'ollama_14b': 'qwen2.5:14b',
}

_BATCH_SIZE = 20


def _ensure_ollama_model(provider):
    """Pull the Ollama model if not already cached. Swallows exceptions."""
    model = OLLAMA_MODELS.get(provider)
    if not model:
        return
    url = f"{OLLAMA_HOST}/api/pull"
    payload = json.dumps({"name": model, "stream": False}).encode()
    req = urllib.request.Request(url, data=payload,
                                 headers={'Content-Type': 'application/json'},
                                 method='POST')
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read())
            logger.info("Ollama pull %s: %s", model, data.get('status', 'ok'))
    except Exception as e:
        logger.warning("Could not pull Ollama model %s (may already be cached): %s", model, e)


def _chat_ollama(model, messages, timeout=180):
    """Send a chat request to the local Ollama API. Returns content string."""
    url = f"{OLLAMA_HOST}/api/chat"
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "stream": False,
        "format": "json",
    }).encode()
    req = urllib.request.Request(url, data=payload,
                                 headers={'Content-Type': 'application/json'},
                                 method='POST')
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    return data["message"]["content"]


def _chat_claude(messages, timeout=120):
    """Send a chat request to the Claude API. Returns content string."""
    api_key = os.environ.get('CLAUDE_API_KEY', '')
    url = "https://api.anthropic.com/v1/messages"

    # Claude API expects system as a top-level field, not in messages
    system_prompt = None
    chat_messages = []
    for msg in messages:
        if msg.get('role') == 'system':
            system_prompt = msg['content']
        else:
            chat_messages.append(msg)

    body = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 2048,
        "messages": chat_messages,
    }
    if system_prompt:
        body["system"] = system_prompt

    payload = json.dumps(body).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            'Content-Type': 'application/json',
            'x-api-key': api_key,
            'anthropic-version': '2023-06-01',
        },
        method='POST',
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    return data["content"][0]["text"]


def _parse_json_response(raw):
    """Parse an LLM response that should be JSON. Strips markdown fences if present."""
    # Strip markdown code fences
    stripped = raw.strip()
    if stripped.startswith('```'):
        stripped = re.sub(r'^```[a-z]*\n?', '', stripped)
        stripped = re.sub(r'\n?```$', '', stripped)
        stripped = stripped.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    # Fallback: find first JSON object in the response
    match = re.search(r'\{.*\}', stripped, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not parse JSON from LLM response: {raw[:200]!r}")


def _call_llm(provider, messages):
    """Route to the correct LLM backend and return a parsed JSON dict."""
    if provider in OLLAMA_MODELS:
        model = OLLAMA_MODELS[provider]
        raw = _chat_ollama(model, messages)
    elif provider == 'claude':
        raw = _chat_claude(messages)
    else:
        raise ValueError(f"Unknown LLM provider: {provider!r}")
    return _parse_json_response(raw)


_SYSTEM_PROMPT = """\
You are a literary analyst for audiobook production. Your task is to identify who is speaking in each paragraph of a novel.

Rules:
- NARRATOR: narration, description, action without dialogue
- Character name: when a paragraph IS or CONTAINS the character's primary dialogue
- If a paragraph mixes narration and dialogue, use the speaker of the dialogue
- Use exact character names (proper nouns, consistent spelling)
- Return ONLY valid JSON, nothing else"""


def _build_user_message(roster, indexed_paragraphs):
    """Build the user message string for a batch of (local_idx, text) pairs."""
    roster_str = ', '.join(sorted(roster)) if roster else 'None yet'
    lines = [
        f"Known characters: {roster_str}",
        "",
        "Identify the primary speaker for each numbered paragraph. Return JSON:",
        '{',
        '  "new_characters": ["Name1", "Name2"],',
        '  "speakers": {"0": "NARRATOR", "1": "Alice", "2": "NARRATOR"}',
        '}',
        "",
        "Paragraphs:",
    ]
    for local_idx, text in indexed_paragraphs:
        lines.append(f"[{local_idx}] {text}")
    return '\n'.join(lines)


def analyse_book(paragraphs, provider, progress_callback=None):
    """
    Analyse a list of ParsedParagraph objects and return speaker attribution.

    Returns:
        {
            "characters": {"NARRATOR": {}, "Alice": {}, ...},
            "segments": [{"paragraph_index": int, "speaker": str}, ...]
        }
    """
    # Pull Ollama model up front for ollama providers
    if provider in OLLAMA_MODELS:
        _ensure_ollama_model(provider)

    # Group paragraphs by chapter_index
    chapters: dict[int, list] = {}
    for para in paragraphs:
        chapters.setdefault(para.chapter_index, []).append(para)

    character_roster: set = set()
    # paragraph_index → speaker
    speaker_map: dict[int, str] = {}

    chapter_indices = sorted(chapters.keys())
    total_chapters = len(chapter_indices)

    for ch_num, chapter_idx in enumerate(chapter_indices):
        ch_paras = chapters[chapter_idx]

        # Filter to paragraphs with meaningful text
        valid_paras = [p for p in ch_paras if len(p.text.strip()) > 10]

        if not valid_paras:
            # Nothing to process; move on
            if progress_callback and total_chapters > 0:
                pct = round((ch_num + 1) / total_chapters * 100, 1)
                chapter_title = ch_paras[0].chapter_title if ch_paras else f"Chapter {chapter_idx}"
                progress_callback(pct, f"Analysing: {chapter_title}")
            continue

        # Process in batches of _BATCH_SIZE
        for batch_start in range(0, len(valid_paras), _BATCH_SIZE):
            batch = valid_paras[batch_start:batch_start + _BATCH_SIZE]
            # Use 0-based local index within this batch for the LLM
            indexed = list(enumerate(p.text for p in batch))

            messages = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_message(character_roster, indexed)},
            ]

            try:
                result = _call_llm(provider, messages)
                new_chars = result.get('new_characters', [])
                if isinstance(new_chars, list):
                    for name in new_chars:
                        if isinstance(name, str) and name.strip():
                            character_roster.add(name.strip())

                speakers_raw = result.get('speakers', {})
                for local_idx, para in enumerate(batch):
                    speaker = speakers_raw.get(str(local_idx)) or 'NARRATOR'
                    if not isinstance(speaker, str) or not speaker.strip():
                        speaker = 'NARRATOR'
                    speaker_map[para.paragraph_index] = speaker.strip()

            except Exception as e:
                logger.error(
                    "LLM failed for chapter %d batch starting at %d: %s",
                    chapter_idx, batch_start, e,
                )
                # Fall back to NARRATOR for all paragraphs in this batch
                for para in batch:
                    speaker_map[para.paragraph_index] = 'NARRATOR'

        if progress_callback and total_chapters > 0:
            pct = round((ch_num + 1) / total_chapters * 100, 1)
            chapter_title = valid_paras[0].chapter_title if valid_paras else f"Chapter {chapter_idx}"
            progress_callback(pct, f"Analysed: {chapter_title}")

    # Ensure NARRATOR is always present
    character_roster.add('NARRATOR')

    # Build raw output
    characters = {name: {} for name in sorted(character_roster)}
    characters.setdefault('NARRATOR', {})

    segments = []
    for para in paragraphs:
        speaker = speaker_map.get(para.paragraph_index, 'NARRATOR')
        segments.append({
            'paragraph_index': para.paragraph_index,
            'speaker': speaker,
        })

    # Deduplicate name variants (e.g. "John" / "Sgt. John" / "Mr. Smith" → one entry)
    characters, segments = _deduplicate_characters(characters, segments)

    # Add appearance counts to each character
    characters = _add_appearance_counts(characters, segments)

    return {
        'characters': characters,
        'segments': segments,
    }

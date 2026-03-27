import os
import re
import logging
from dataclasses import dataclass, field
from typing import Optional

import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup, NavigableString, Tag

logger = logging.getLogger(__name__)

# Abbreviations to expand before TTS to avoid mispronunciation
ABBREVIATIONS = [
    (r'\bMr\.', 'Mister'),
    (r'\bMrs\.', 'Misses'),
    (r'\bMs\.', 'Miss'),
    (r'\bDr\.', 'Doctor'),
    (r'\bProf\.', 'Professor'),
    (r'\bSt\.', 'Saint'),
    (r'\bAve\.', 'Avenue'),
    (r'\bJr\.', 'Junior'),
    (r'\bSr\.', 'Senior'),
    (r'\bvs\.', 'versus'),
    (r'\betc\.', 'et cetera'),
    (r'\be\.g\.', 'for example'),
    (r'\bi\.e\.', 'that is'),
]
_ABBREV_PATTERNS = [(re.compile(p, re.IGNORECASE), r) for p, r in ABBREVIATIONS]

MIN_PARAGRAPH_CHARS = 20
MIN_DOCUMENT_CHARS = 200

# epub:type values on a document's body/section that mean we should skip it
SKIP_DOC_EPUB_TYPES = {
    'cover', 'frontmatter', 'titlepage', 'copyright-page',
    'toc', 'backmatter', 'endnotes', 'rearnotes', 'footnotes',
    'loi', 'lot', 'index',
}

# Tags whose entire subtree we skip unconditionally
SKIP_SUBTREE_TAGS = {'nav', 'script', 'style', 'noscript'}

# Tags that create chapter/section boundaries
HEADING_TAGS = {'h1', 'h2', 'h3'}


@dataclass
class ParsedParagraph:
    chapter_index: int
    chapter_title: str
    paragraph_index: int
    text: str


@dataclass
class ParsedBook:
    title: str
    author: str
    cover_data: Optional[bytes]
    cover_media_type: Optional[str]
    paragraphs: list = field(default_factory=list)  # list[ParsedParagraph]
    series: Optional[str] = None
    series_index: Optional[float] = None
    publisher: Optional[str] = None
    published_date: Optional[str] = None


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Strip HTML entities, normalise whitespace, expand abbreviations."""
    text = re.sub(r'\s+', ' ', text).strip()
    for pattern, replacement in _ABBREV_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def _get_epub_type(tag: Tag) -> str:
    """Return the epub:type attribute value, or '' if absent."""
    return tag.get('epub:type', '') or tag.get('epub_type', '') or ''


def _has_class(tag: Tag, *classes: str) -> bool:
    tag_classes = tag.get('class', [])
    return any(c in tag_classes for c in classes)


def _text_excluding(el: Tag, exclude_tags=('sup',), exclude_epub_types=('pagebreak',)) -> str:
    """
    Extract text from el, skipping descendants that are in exclude_tags
    or whose epub:type contains one of exclude_epub_types.
    Used to strip footnote reference numbers and page-break markers.
    """
    parts = []
    for node in el.descendants:
        if not isinstance(node, NavigableString):
            continue
        skip = False
        for ancestor in node.parents:
            if ancestor is el:
                break
            if not isinstance(ancestor, Tag):
                continue
            if ancestor.name in exclude_tags:
                skip = True
                break
            et = _get_epub_type(ancestor)
            if any(x in et for x in exclude_epub_types):
                skip = True
                break
        if not skip:
            parts.append(str(node))
    return ''.join(parts)


# ---------------------------------------------------------------------------
# Document-level skip logic
# ---------------------------------------------------------------------------

def _is_skip_document(soup: BeautifulSoup, raw_text: str) -> bool:
    """Return True if this spine document should be excluded from TTS."""
    # Too short to be real content
    if len(raw_text.strip()) < MIN_DOCUMENT_CHARS:
        return True

    # EPUB3 nav document
    if soup.find('nav'):
        return True

    # Check epub:type on body and immediate top-level section
    body = soup.find('body')
    if body:
        et = _get_epub_type(body)
        if any(t in et for t in SKIP_DOC_EPUB_TYPES):
            return True
        first_section = body.find('section', recursive=False)
        if first_section:
            et = _get_epub_type(first_section)
            if any(t in et for t in SKIP_DOC_EPUB_TYPES):
                return True

    return False


# ---------------------------------------------------------------------------
# Recursive element walker
# ---------------------------------------------------------------------------

class _WalkState:
    __slots__ = ('chapter_index', 'chapter_title', 'first_heading_seen',
                 'paragraph_index', 'paragraphs')

    def __init__(self):
        self.chapter_index = 0
        self.chapter_title = 'Chapter 1'
        self.first_heading_seen = False
        self.paragraph_index = 0
        self.paragraphs = []

    def add_paragraph(self, text: str):
        self.paragraphs.append(ParsedParagraph(
            chapter_index=self.chapter_index,
            chapter_title=self.chapter_title,
            paragraph_index=self.paragraph_index,
            text=text,
        ))
        self.paragraph_index += 1

    def new_chapter(self, title: str):
        self.chapter_index += 1
        self.chapter_title = title


def _process_element(el: Tag, state: _WalkState):
    """Recursively walk el, extracting paragraphs and tracking chapters."""
    tag = el.name
    if tag is None:
        return

    epub_type = _get_epub_type(el)
    css_classes = el.get('class', [])

    # --- Skip entire subtrees ---
    if tag in SKIP_SUBTREE_TAGS:
        return
    if 'pagebreak' in epub_type:
        return
    if 'chapter-end' in css_classes:
        return
    # Skip footnote/endnote sections
    if any(t in epub_type for t in ('endnotes', 'footnotes', 'rearnotes')):
        return

    # --- Headings → chapter boundary + spoken aloud ---
    if tag in HEADING_TAGS:
        heading_text = clean_text(_text_excluding(el))
        if heading_text:
            if not state.first_heading_seen:
                # First heading in this document: update title of the already-started chapter
                state.chapter_title = heading_text
                state.first_heading_seen = True
            else:
                # Subsequent heading within document: new section
                state.new_chapter(heading_text)
            # Always synthesise the heading text so it is read aloud
            state.add_paragraph(heading_text)
        return  # don't recurse into headings

    # --- Regular paragraphs ---
    if tag == 'p':
        # Skip decorative/signature-only paragraphs
        if 'chapter-end' in css_classes:
            return
        text = clean_text(_text_excluding(el))
        if len(text) >= MIN_PARAGRAPH_CHARS:
            state.add_paragraph(text)
        return  # don't recurse further into <p>

    # --- Lists: each <li> becomes a paragraph ---
    if tag in ('ol', 'ul'):
        for li in el.find_all('li', recursive=False):
            text = clean_text(_text_excluding(li))
            if len(text) >= MIN_PARAGRAPH_CHARS:
                state.add_paragraph(text)
        return

    # --- Blockquotes: extract <p> children, skip <footer> attribution ---
    if tag == 'blockquote':
        for child in el.children:
            if not isinstance(child, Tag):
                continue
            if child.name == 'footer':
                continue  # skip attribution lines
            if child.name == 'p':
                text = clean_text(_text_excluding(child))
                if len(text) >= MIN_PARAGRAPH_CHARS:
                    state.add_paragraph(text)
            else:
                # e.g. nested blockquote, div
                _process_element(child, state)
        return

    # --- Tables: caption + rows ---
    if tag == 'table':
        caption = el.find('caption')
        if caption:
            text = clean_text(_text_excluding(caption))
            if len(text) >= MIN_PARAGRAPH_CHARS:
                state.add_paragraph(text)
        for row in el.find_all('tr'):
            cells = []
            for cell in row.find_all(['td', 'th']):
                t = clean_text(_text_excluding(cell))
                if t:
                    cells.append(t)
            if cells:
                text = ', '.join(cells)
                if len(text) >= MIN_PARAGRAPH_CHARS:
                    state.add_paragraph(text)
        return

    # --- Recurse into everything else (section, div, aside, article, etc.) ---
    for child in el.children:
        if isinstance(child, Tag):
            _process_element(child, state)


# ---------------------------------------------------------------------------
# Cover extraction
# ---------------------------------------------------------------------------

def _extract_cover(book: epub.EpubBook) -> tuple[Optional[bytes], Optional[str]]:
    """Try to extract cover image bytes and media type."""
    cover_id = None
    for meta in book.get_metadata('OPF', 'meta'):
        attrs = meta[1] if len(meta) > 1 else {}
        if attrs.get('name') == 'cover':
            cover_id = attrs.get('content')
            break

    if cover_id:
        try:
            item = book.get_item_with_id(cover_id)
            if item:
                return item.get_content(), item.media_type
        except Exception:
            pass

    for item in book.get_items_of_type(ebooklib.ITEM_IMAGE):
        name = (item.file_name or '').lower()
        if 'cover' in name or 'front' in name:
            return item.get_content(), item.media_type

    for item in book.get_items_of_type(ebooklib.ITEM_IMAGE):
        return item.get_content(), item.media_type

    return None, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_epub(epub_path: str) -> ParsedBook:
    """
    Parse an EPUB and return a ParsedBook with ordered paragraphs.

    Chapter boundaries:
      - Each new (non-skipped) spine document starts a new chapter.
      - The first heading in a document sets that chapter's title.
      - Subsequent h1/h2/h3 within the same document create new sub-chapters.

    Content handled:
      - <p> paragraphs (text extracted without footnote ref superscripts)
      - <ol>/<ul> list items (each item → paragraph)
      - <blockquote> paragraphs (attribution <footer> skipped)
      - <table> caption + rows
      - <aside> and sidebar content (included — informational text)

    Skipped:
      - nav, script, style
      - epub:type pagebreak spans
      - Decorative ornaments (class="chapter-end")
      - Documents flagged as cover/titlepage/copyright/toc/endnotes via epub:type
      - Documents shorter than MIN_DOCUMENT_CHARS
    """
    book = epub.read_epub(epub_path)

    title = 'Unknown Title'
    author = 'Unknown Author'

    titles = book.get_metadata('DC', 'title')
    if titles:
        title = titles[0][0]

    creators = book.get_metadata('DC', 'creator')
    if creators:
        author = creators[0][0]

    publisher = None
    published_date = None
    series = None
    series_index = None

    publishers = book.get_metadata('DC', 'publisher')
    if publishers:
        publisher = publishers[0][0]

    dates = book.get_metadata('DC', 'date')
    if dates:
        published_date = dates[0][0]

    for meta in book.get_metadata('OPF', 'meta'):
        value = meta[0] if meta else None
        attrs = meta[1] if len(meta) > 1 else {}
        name = attrs.get('name', '')
        prop = attrs.get('property', '')
        if name == 'calibre:series' and attrs.get('content'):
            series = attrs['content']
        elif name == 'calibre:series_index' and attrs.get('content'):
            try:
                series_index = float(attrs['content'])
            except (ValueError, TypeError):
                pass
        elif prop == 'belongs-to-collection' and value:
            if not series:
                series = value
        elif prop == 'group-position' and value:
            if series_index is None:
                try:
                    series_index = float(value)
                except (ValueError, TypeError):
                    pass

    cover_data, cover_media_type = _extract_cover(book)

    state = _WalkState()
    first_doc = True

    spine_ids = [item_id for item_id, _ in book.spine]

    for spine_id in spine_ids:
        item = book.get_item_with_id(spine_id)
        if item is None:
            continue
        if item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue

        try:
            content = item.get_content().decode('utf-8', errors='replace')
        except Exception as e:
            logger.warning("Could not decode spine item %s: %s", spine_id, e)
            continue

        soup = BeautifulSoup(content, 'html.parser')
        raw_text = soup.get_text()

        if _is_skip_document(soup, raw_text):
            logger.debug("Skipping document: %s", spine_id)
            continue

        # New spine document = new chapter
        if first_doc:
            # chapter_index starts at 0 from _WalkState init
            first_doc = False
        else:
            state.new_chapter(f"Chapter {state.chapter_index + 2}")

        # Reset per-document heading tracker
        state.first_heading_seen = False

        body = soup.find('body') or soup
        for child in body.children:
            if isinstance(child, Tag):
                _process_element(child, state)

    logger.info(
        "Parsed '%s': %d paragraphs, %d chapters",
        title, len(state.paragraphs),
        state.chapter_index + 1 if state.paragraphs else 0,
    )

    return ParsedBook(
        title=title,
        author=author,
        cover_data=cover_data,
        cover_media_type=cover_media_type,
        paragraphs=state.paragraphs,
        series=series,
        series_index=series_index,
        publisher=publisher,
        published_date=published_date,
    )


def save_cover(cover_data: bytes, media_type: str, dest_dir: str, book_id: int) -> Optional[str]:
    """Write cover image bytes to dest_dir and return the filename, or None."""
    if not cover_data:
        return None

    ext_map = {
        'image/jpeg': '.jpg',
        'image/jpg': '.jpg',
        'image/png': '.png',
        'image/gif': '.gif',
        'image/webp': '.webp',
        'image/svg+xml': '.svg',
    }
    ext = ext_map.get(media_type, '.jpg')
    filename = f"cover_{book_id}{ext}"
    path = os.path.join(dest_dir, filename)

    with open(path, 'wb') as f:
        f.write(cover_data)

    return filename

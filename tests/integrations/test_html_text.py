from jarvis.integrations.html_text import extract_readable

LONG = "This sentence is long enough to count as a real paragraph for the extractor."


def page(body: str, head: str = "") -> str:
    return f"<html><head>{head}</head><body>{body}</body></html>"


def test_title_prefers_og_title_then_h1_then_title_tag() -> None:
    og = page("<h1>H1</h1>", '<title>T</title><meta property="og:title" content="OG">')
    assert extract_readable(og).title == "OG"
    assert extract_readable(page("<h1>H1</h1>", "<title>T</title>")).title == "H1"
    assert extract_readable(page("<p>x</p>", "<title>T - Site</title>")).title == "T - Site"
    assert extract_readable(page("<p>x</p>")).title == ""


def test_byline_from_meta_rel_author_or_class() -> None:
    assert extract_readable(page("", '<meta name="author" content="Ann">')).byline == "Ann"
    assert extract_readable(page('<a rel="author">Bob</a>')).byline == "Bob"
    assert extract_readable(page('<span class="post-byline">By Cy</span>')).byline == "By Cy"
    assert extract_readable(page('<div class="author">' + "x" * 200 + "</div>")).byline is None


def test_published_from_meta_or_time_element() -> None:
    meta = '<meta property="article:published_time" content="2026-01-02">'
    assert extract_readable(page("", meta)).published == "2026-01-02"
    assert extract_readable(page('<time datetime="2026-03-04">x</time>')).published == "2026-03-04"
    assert extract_readable(page("<time>no attr</time>")).published is None


def test_body_falls_back_to_whole_page_when_nothing_votes() -> None:
    readable = extract_readable(page("<div>A text-only div that is long enough to keep.</div>"))

    assert readable.paragraphs == ("A text-only div that is long enough to keep.",)


def test_content_block_beats_link_heavy_blocks() -> None:
    menu = (
        "<div id='menu'>"
        + "".join(f"<p><a href='/{i}'>{LONG}</a></p>" for i in range(5))
        + "</div>"
    )
    content = "<div id='post'>" + "".join(f"<p>{LONG} {i}</p>" for i in range(3)) + "</div>"
    readable = extract_readable(page(menu + content))

    assert len(readable.paragraphs) == 3
    assert all(paragraph.startswith(LONG) for paragraph in readable.paragraphs)


def test_scripts_styles_and_boilerplate_classes_are_dropped() -> None:
    html = page(
        f"<script>var x = '{LONG}';</script><style>p {{}}</style>"
        f"<div class='post-comments'><p>{LONG} comment</p></div>"
        f"<article><p>{LONG} body</p><p>{LONG} body two</p></article>"
    )
    readable = extract_readable(html)

    assert readable.paragraphs == (f"{LONG} body", f"{LONG} body two")


def test_unclosed_paragraphs_and_list_items_are_separated() -> None:
    html = page(f"<article><p>{LONG} one<p>{LONG} two<ul><li>{LONG} a<li>{LONG} b</ul></article>")

    readable = extract_readable(html)

    assert readable.paragraphs == (f"{LONG} one", f"{LONG} two", f"{LONG} a", f"{LONG} b")


def test_headings_are_kept_and_short_lines_are_not() -> None:
    html = page(f"<article><h2>Heading</h2><p>{LONG}</p><p>tiny</p><p>{LONG}!</p></article>")

    readable = extract_readable(html)

    assert readable.paragraphs == ("Heading", LONG, f"{LONG}!")
    assert readable.text == f"Heading\n\n{LONG}\n\n{LONG}!"


def test_nested_blocks_do_not_duplicate_text() -> None:
    html = page(f"<article><blockquote><p>{LONG}</p></blockquote><p>{LONG} after</p></article>")

    assert extract_readable(html).paragraphs == (LONG, f"{LONG} after")


def test_malformed_markup_never_raises() -> None:
    for html in ("", "<p", "</div></div><b>", "<html><body><p>" + LONG, "&nbsp;&#x41;&bogus;"):
        extract_readable(html)

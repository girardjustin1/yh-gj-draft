"""Every CSS class the page's JavaScript emits must have a rule behind it.

This exists because of a specific, repeated failure. The page is one file with a <style>
block and a <script> block, and several section comments appear verbatim in both. Editing
by anchoring on those comments has four times now sliced across the boundary — once
deleting 78 of 143 CSS rules, which left the player list rendering as an unstyled bulleted
list while every check I had ("the JS parses", "the element is gone") still passed.

Parsing was never the property that mattered. This is: extract the classes the JS actually
puts into the DOM, and assert the stylesheet defines them.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PAGE = Path(__file__).parent.parent / "src/fantasy_draft/api/static/index.html"

#: Emitted by JS but deliberately unstyled, or styled only via a parent rule.
UNSTYLED_BY_DESIGN = {
    "grow", "dot", "n", "who", "lbl", "v", "k", "a", "b", "c", "d", "p", "u", "g",
    "med", "on", "best", "top", "mine", "armed", "taken", "done", "open", "empty",
    "collapsed", "muted", "note", "err", "show", "hidden",
}


@pytest.fixture(scope="module")
def page() -> str:
    return PAGE.read_text()


@pytest.fixture(scope="module")
def css(page: str) -> str:
    return re.findall(r"<style>(.*?)</style>", page, re.S)[0]


@pytest.fixture(scope="module")
def js(page: str) -> str:
    return re.findall(r"<script>(.*?)</script>", page, re.S)[0]


def defined_selectors(css: str) -> set[str]:
    """Every class name appearing in any selector.

    Split on braces rather than matching line-anchored rules: several rules share a line
    (``.pos-QB{...}.pos-RB{...}``) and an anchored pattern silently misses all but the
    first, which would make this guard quietly under-report.
    """
    body = re.sub(r"/\*.*?\*/", " ", css, flags=re.S)
    names: set[str] = set()
    stack: list[bool] = []          # True where the open block is an at-rule wrapper
    selector = ""
    for chunk in re.split(r"([{}])", body):
        if chunk == "{":
            is_at_rule = selector.strip().startswith("@")
            # @media and friends wrap rules rather than being one, so their contents are
            # still real selectors. Treating them as a rule body hides every rule inside.
            if not is_at_rule and all(stack) if stack else not is_at_rule:
                names.update(re.findall(r"\.([A-Za-z][\w-]*)", selector))
            stack.append(is_at_rule)
            selector = ""
        elif chunk == "}":
            if stack:
                stack.pop()
            selector = ""
        elif all(stack):            # top level, or inside only at-rule wrappers
            selector = chunk
    return names


def _strip_interpolations(text: str) -> str:
    """Remove ${...}, honouring the quotes and nested braces inside them."""
    out, i = [], 0
    while i < len(text):
        if text.startswith("${", i):
            depth, i = 1, i + 2
            while i < len(text) and depth:
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                i += 1
            out.append(" ")
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def class_attributes(source: str) -> set[str]:
    """Class names in every ``class="..."``, tolerating quotes inside interpolations.

    A naive ``class="([^"]*)"`` stops at the first quote inside a ``${a ? "x" : "y"}``
    and yields fragments like ``${flex?`` as if they were class names.
    """
    names: set[str] = set()
    for match in re.finditer(r'class="', source):
        i = match.end()
        depth, end = 0, None
        while i < len(source):
            char = source[i]
            if source.startswith("${", i):
                depth += 1
                i += 2
                continue
            if char == "{" and depth:
                depth += 1
            elif char == "}" and depth:
                depth -= 1
            elif char == '"' and depth == 0:
                end = i
                break
            i += 1
        if end is None:
            continue
        cleaned = _strip_interpolations(source[match.end():end])
        names.update(token for token in cleaned.split() if token)
    return names


def emitted_classes(js: str) -> set[str]:
    """Every class name the JS writes into the DOM."""
    return class_attributes(js)


class TestStylesheetCoversTheMarkup:
    def test_every_emitted_class_has_a_rule(self, css: str, js: str):
        defined = defined_selectors(css)
        missing = sorted(emitted_classes(js) - defined - UNSTYLED_BY_DESIGN)
        assert not missing, (
            "JavaScript emits classes with no CSS rule, so those elements render "
            f"unstyled: {missing}"
        )

    def test_static_markup_classes_are_styled_too(self, page: str, css: str):
        body = page[page.index('<div class="wrap">'):page.index("<script")]
        defined = defined_selectors(css)
        missing = sorted(class_attributes(body) - defined - UNSTYLED_BY_DESIGN)
        assert not missing, f"markup uses unstyled classes: {missing}"

    def test_position_chips_are_all_coloured(self, css: str):
        """posClass() yields pos-QB/RB/WR/TE/K; each needs a colour or chips go invisible."""
        defined = defined_selectors(css)
        for position in ("QB", "RB", "WR", "TE", "K"):
            assert f"pos-{position}" in defined, f"pos-{position} has no colour"

    def test_the_core_layout_rules_exist(self, css: str):
        """The rules whose loss produced an unstyled bulleted list."""
        defined = defined_selectors(css)
        for name in ("prow", "pinner", "plist", "pname", "pmeta", "pnums",
                     "draftbtn", "swipehint", "tabs", "tab2", "filters", "pillbtn",
                     "pathline", "thenpos", "strow", "sttrack", "teamslot"):
            assert name in defined, f".{name} is missing — that block renders unstyled"

    def test_the_blocks_have_not_bled_into_each_other(self, css: str, js: str):
        """The failure mode itself: JS in <style>, or CSS rules in <script>."""
        assert "function " not in css
        assert "=>" not in css
        assert ".prow{" not in js
        assert "display:grid;grid-template-columns" not in js

    def test_the_stylesheet_has_not_collapsed(self, css: str):
        """A blunt tripwire: a big accidental deletion should fail loudly."""
        assert len(defined_selectors(css)) > 100, (
            "the stylesheet has far fewer rules than expected — something was deleted"
        )

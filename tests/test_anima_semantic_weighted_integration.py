from dataclasses import dataclass

from sd_embed.embedding_funcs import (
    _anima_semantic_compile_prompt,
    _anima_semantic_compile_item,
)


@dataclass
class _Result:
    compiled: str


class _DummyFrontend:
    def __init__(self, mapping=None):
        self.mapping = dict(mapping or {})
        self.calls = []

    def process_one(self, text, negative=False):
        self.calls.append((text, negative))
        compiled = self.mapping.get(text, text)
        return _Result(compiled=compiled)


def test_semantic_and_preserves_top_level_segments_and_weights():
    frontend = _DummyFrontend(
        {
            "a girl sitting near a lake": "1girl, sitting near lake",
            "a boy standing behind her": "1boy, standing behind girl",
        }
    )
    compiled = _anima_semantic_compile_prompt(
        frontend,
        "a girl sitting near a lake AND a boy standing behind her:0.8",
        negative=False,
    )
    assert compiled == "1girl, sitting near lake AND 1boy, standing behind girl:0.8"


def test_semantic_item_preserves_simple_weighted_group():
    frontend = _DummyFrontend({"red hair": "long red hair"})
    compiled = _anima_semantic_compile_item(frontend, "(red hair:1.4)", negative=False)
    assert compiled == "(long red hair:1.4)"


def test_semantic_item_preserves_complex_inline_attention_without_rewrite():
    frontend = _DummyFrontend({"girl with red hair and blue eyes": "should not be used"})
    original = "girl with (red hair:1.4) and blue eyes"
    compiled = _anima_semantic_compile_item(frontend, original, negative=False)
    assert compiled == original
    assert frontend.calls == []


def test_semantic_break_boundary_is_preserved():
    frontend = _DummyFrontend(
        {
            "a girl on the left": "1girl, left side",
            "a city skyline at night": "city skyline, night",
        }
    )
    compiled = _anima_semantic_compile_prompt(
        frontend,
        "a girl on the left, BREAK, a city skyline at night",
        negative=False,
    )
    assert compiled == "1girl, left side, BREAK, city skyline, night"

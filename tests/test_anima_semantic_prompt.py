from __future__ import annotations

from types import SimpleNamespace

from sd_embed.anima_semantic_prompt import (
    AnimaSemanticPromptFrontend,
    PROMPT_MODE_COMPILE,
    PROMPT_MODE_DIRECT,
    TagLexiconResolver,
)


class TinyTokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def __call__(self, text, **kwargs):
        words = str(text).replace(",", " , ").replace(";", " ; ").split()
        ids = list(range(3, 3 + len(words)))
        max_length = kwargs.get("max_length")
        if kwargs.get("truncation") and max_length is not None:
            ids = ids[: int(max_length)]
        if kwargs.get("return_tensors") == "pt":
            import torch

            return {"input_ids": torch.tensor([ids or [2]]), "attention_mask": torch.ones(1, len(ids or [2]), dtype=torch.long)}
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"tok{int(i)}" for i in ids)


class DummyPipe:
    def __init__(self):
        tok = TinyTokenizer()
        self.prompt_tokenizer = SimpleNamespace(qwen_tokenizer=tok, t5_tokenizer=tok)
        self.text_encoder = object()
        self.processor = None

    def set_prompt_processor(self, processor):
        self.processor = processor

    def clear_prompt_processor(self):
        self.processor = None


def test_tag_alias_and_description_expansion():
    resolver = TagLexiconResolver(
        aliases={"old_tag": "canonical_tag"},
        implications={"canonical_tag": ["parent_tag"]},
        descriptions={"canonical_tag": "visual gloss"},
    )
    assert resolver.expand_tag_prompt("old_tag, blue_eyes") == (
        "canonical_tag, parent_tag, visual gloss, blue_eyes"
    )


def test_direct_mode_budget_stays_within_target():
    pipe = DummyPipe()
    frontend = AnimaSemanticPromptFrontend(pipe, mode=PROMPT_MODE_DIRECT, target_t5_tokens=32)
    result = frontend.process_one(", ".join(f"tag_{i}" for i in range(100)))
    assert result.anima_t5_tokens <= 32
    assert result.used_generation is False


def test_compile_mode_falls_back_without_generate():
    pipe = DummyPipe()
    frontend = AnimaSemanticPromptFrontend(pipe, mode=PROMPT_MODE_COMPILE, target_t5_tokens=32)
    result = frontend.process_one("A person is standing behind a chair while looking left.")
    assert result.mode == PROMPT_MODE_COMPILE
    assert result.used_generation is False
    assert result.anima_t5_tokens <= 32


def test_install_and_uninstall():
    pipe = DummyPipe()
    frontend = AnimaSemanticPromptFrontend(pipe)
    frontend.install()
    assert pipe.processor is frontend
    frontend.uninstall()
    assert pipe.processor is None


def test_base_semantic_generation_is_opt_in_even_when_over_budget():
    class GenerateMustNotRun:
        def generate(self, *args, **kwargs):
            raise AssertionError("Base semantic generation must stay opt-in")

    pipe = DummyPipe()
    pipe.text_encoder = GenerateMustNotRun()
    frontend = AnimaSemanticPromptFrontend(
        pipe,
        mode=PROMPT_MODE_DIRECT,
        target_t5_tokens=32,
    )
    result = frontend.process_one(", ".join(f"tag_{i}" for i in range(100)))
    assert result.used_generation is False
    assert result.anima_t5_tokens <= 32

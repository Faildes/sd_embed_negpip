"""SD_Embed must honor Diffusers-Anima's persistent transformer mode."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

import sd_embed.embedding_funcs as embedding_funcs


@pytest.mark.parametrize("resident, expected_offload", [(True, False), (False, True)])
def test_condition_builder_keeps_resident_transformer_loaded(
    monkeypatch, resident, expected_offload
):
    seen = []

    @contextmanager
    def record_context(module, *, execution_device, execution_dtype, enable_offload):
        seen.append(enable_offload)
        yield

    monkeypatch.setattr(
        embedding_funcs, "_anima_v3_module_execution_context", record_context
    )

    class FakeTransformer:
        def preprocess_text_embeds(self, hidden, token_ids, t5xxl_weights):
            return hidden

    pipe = SimpleNamespace(
        transformer=FakeTransformer(),
        execution_device="cpu",
        model_dtype=torch.float32,
        text_encoder_dtype=torch.float32,
        use_module_cpu_offload=True,
        keep_transformer_on_device=resident,
    )
    embedding_funcs._anima_v3_build_condition(
        pipe,
        qwen_hidden=torch.ones(1, 2, 4),
        t5_ids=torch.zeros(1, 2, dtype=torch.int32),
        t5_weights=torch.ones(1, 2, 1),
        target_length=None,
    )
    assert seen == [expected_offload]

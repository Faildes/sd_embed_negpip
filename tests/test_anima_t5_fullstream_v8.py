from pathlib import Path


def test_v9_prompt_plan_declares_vanilla_contract_full_t5_stream_without_selection():
    source = Path("src/sd_embed/embedding_funcs.py").read_text(encoding="utf-8")
    for marker in (
        '"prompt_plan_version": 9',
        '"conditioning_mode": "single_qwen_memory_vanilla_t5_single_pass"',
        '"preserve_full_t5_stream": True',
        '"long_source_policy": "full_qwen_memory_full_t5_single_pass"',
        '"t5_query_policy": "full_exact_token_stream_v1"',
        '"t5_query_paging": False',
        '"t5_query_compression": False',
        '"t5_query_selection": False',
        '"conditioning_stability_policy": "vanilla_t5_single_pass_variable_length_v2"',
    ):
        assert marker in source


def test_v9_preserves_independent_cfg_condition_lengths():
    source = Path("src/sd_embed/embedding_funcs.py").read_text(encoding="utf-8")
    assert "def _anima_v8_preserve_independent_cfg_lengths" in source
    assert "t5_single_pass_full_stream" in source

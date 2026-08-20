from pathlib import Path


def test_v8_prompt_plan_declares_full_t5_stream_without_selection():
    source = Path("src/sd_embed/embedding_funcs.py").read_text(encoding="utf-8")
    for marker in (
        '"prompt_plan_version": 8',
        '"conditioning_mode": "single_qwen_memory_single_t5_stream"',
        '"preserve_full_t5_stream": True',
        '"long_source_policy": "full_qwen_memory_full_t5_single_pass"',
        '"t5_query_policy": "full_exact_token_stream_v1"',
        '"t5_query_paging": False',
        '"t5_query_compression": False',
        '"t5_query_selection": False',
        '"conditioning_stability_policy": "null_occupancy_single_pass_v1"',
    ):
        assert marker in source


def test_v8_preserves_independent_cfg_condition_lengths():
    source = Path("src/sd_embed/embedding_funcs.py").read_text(encoding="utf-8")
    assert "def _anima_v8_preserve_independent_cfg_lengths" in source
    assert "t5_single_pass_full_stream" in source

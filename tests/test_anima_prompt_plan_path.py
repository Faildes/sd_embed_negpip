import ast
from pathlib import Path


def _final_anima_embedding_def():
    tree = ast.parse(Path("src/sd_embed/embedding_funcs.py").read_text(encoding="utf-8"))
    defs = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_weighted_text_embeddings_anima"
    ]
    return defs[-1]


def test_final_anima_path_exposes_prompt_plan_switch():
    node = _final_anima_embedding_def()
    args = {arg.arg for arg in node.args.kwonlyargs}
    assert "use_prompt_plan" in args
    assert "prompt_plan_auto_subject_groups" in args
    assert "prompt_plan_exact_subject_count" in args


def test_prompt_plan_helpers_exist():
    source = Path("src/sd_embed/embedding_funcs.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    defs = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert "_anima_build_prompt_plan" in defs
    assert "_anima_encode_prompt_plans_if_supported" in defs


def test_v4_prompt_plan_exposes_binding_and_stability_controls():
    node = _final_anima_embedding_def()
    args = {arg.arg for arg in node.args.kwonlyargs}
    expected = {
        "conditioning_delta_clip_ratio",
        "conditioning_token_rms_strength",
        "conditioning_token_rms_min_ratio",
        "conditioning_token_rms_max_ratio",
        "semantic_expansion_group_aware",
        "semantic_expansion_coherence_power",
        "semantic_expansion_min_coherence",
        "prompt_plan_semicolon_groups",
    }
    assert expected <= args


def test_v4_semicolon_grouping_is_threaded_into_prompt_plan_builder():
    source = Path("src/sd_embed/embedding_funcs.py").read_text(encoding="utf-8")
    assert 'semicolon_groups: bool = True' in source
    assert '"semicolon_groups": bool(semicolon_groups)' in source
    assert 'semicolon_groups=bool(prompt_plan_semicolon_groups)' in source


def test_v5_subject_binding_metadata_is_threaded_without_rewriting_prompt():
    source = Path("src/sd_embed/embedding_funcs.py").read_text(encoding="utf-8")
    for marker in (
        '"subject_binding_version": 2',
        '"subject_group_ids": subject_group_ids',
        'metadata["subject_count"]',
        '_ANIMA_COMPACT_GENDER_COUNT_RE',
        'auto_subject_groups=bool(prompt_plan_auto_subject_groups)',
        'exact_subject_count=bool(prompt_plan_exact_subject_count)',
    ):
        assert marker in source
    ast.parse(source)


def test_v6_prompt_plan_exposes_color_intent_and_calibration_bucket_metadata():
    source = Path("src/sd_embed/embedding_funcs.py").read_text(encoding="utf-8")
    for marker in (
        '"saturation_intent_version": 1',
        '"color_intent": color_intent',
        '"explicit_color_intent": color_intent != "neutral"',
        '"calibration_bucket": calibration_bucket',
        'def _anima_color_intent',
        'def _anima_calibration_bucket',
    ):
        assert marker in source
    ast.parse(source)


def test_v7_full_source_preservation_and_prompt_adherence_metadata():
    source = Path("src/sd_embed/embedding_funcs.py").read_text(encoding="utf-8")
    for marker in (
        '"prompt_plan_version": 5',
        '"long_source_policy": "full_qwen_memory_fixed_512_queries"',
        '"prompt_adherence_version": 1',
        'def _anima_prompt_modality',
        'def _anima_directive_density',
        'there (?:is|are)',
        'female|male',
    ):
        assert marker in source
    ast.parse(source)

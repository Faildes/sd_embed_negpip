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

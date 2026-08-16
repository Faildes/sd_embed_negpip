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

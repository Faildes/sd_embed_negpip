import ast
from pathlib import Path


def _final_anima_embedding_def():
    tree = ast.parse(Path("src/sd_embed/embedding_funcs.py").read_text(encoding="utf-8"))
    defs = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_weighted_text_embeddings_anima"
    ]
    return defs[-1]


def test_final_anima_path_can_require_aligned_encoder():
    node = _final_anima_embedding_def()
    args = {arg.arg for arg in node.args.kwonlyargs}
    assert "require_aligned_text_encoder" in args


def test_prompt_plan_v2_metadata_is_present():
    source = Path("src/sd_embed/embedding_funcs.py").read_text(encoding="utf-8")
    assert '"prompt_plan_version": 2' in source
    assert '"conditioning_mode": "single_qwen_memory"' in source

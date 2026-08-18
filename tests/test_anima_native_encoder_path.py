import ast
from pathlib import Path


def _embedding_def():
    tree = ast.parse(Path("src/sd_embed/embedding_funcs.py").read_text(encoding="utf-8"))
    defs = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "get_weighted_text_embeddings_anima"
    ]
    return defs[-1]


def test_native_encoder_can_be_required_for_final_validation():
    node = _embedding_def()
    args = {arg.arg for arg in node.args.kwonlyargs}
    assert "require_native_text_encoder" in args


def test_native_encoder_guard_rejects_legacy_path_when_requested():
    source = Path("src/sd_embed/embedding_funcs.py").read_text(encoding="utf-8")
    assert "native_required" in source
    assert "anima_native_text_encoder_v1" in source
    assert 'info.get("native_encoder", False)' in source


def test_bridge_runtime_knobs_are_not_applied_to_native_encoder():
    source = Path("src/sd_embed/embedding_funcs.py").read_text(encoding="utf-8")
    assert "native_encoder_active" in source
    assert "not native_encoder_active" in source

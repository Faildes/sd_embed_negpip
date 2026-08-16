import ast
from pathlib import Path


def _final_anima_embedding_def():
    source = Path("src/sd_embed/embedding_funcs.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    defs = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "get_weighted_text_embeddings_anima"
    ]
    assert defs, "get_weighted_text_embeddings_anima is not defined"
    return defs[-1]


def test_final_anima_embedding_signature_exposes_semantic_and_artist_mixer_options():
    node = _final_anima_embedding_def()
    names = {arg.arg for arg in node.args.args + node.args.kwonlyargs}

    semantic = {
        "enable_semantic",
        "semantic_frontend",
        "semantic_mode",
        "semantic_target_t5_tokens",
        "semantic_qwen_input_max_tokens",
        "semantic_compiler_max_new_tokens",
        "semantic_system_prompt",
        "semantic_tag_resolver",
        "semantic_tag_resolver_path",
        "semantic_process_negative",
        "semantic_generation_kwargs",
        "semantic_compression_retries",
    }
    artist_mixer = {
        "enable_artist_mixer",
        "artist_mixer",
        "return_artist_mixer",
    }

    assert semantic <= names
    assert artist_mixer <= names


def test_final_anima_embedding_does_not_reference_missing_semantic_parameters():
    node = _final_anima_embedding_def()
    parameters = {arg.arg for arg in node.args.args + node.args.kwonlyargs}
    loaded = {
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
    }
    semantic_loaded = {
        name for name in loaded
        if name == "enable_semantic" or name.startswith("semantic_")
    }
    # semantic_compiler is a local variable, so exclude it from the parameter check.
    semantic_loaded.discard("semantic_compiler")
    assert semantic_loaded <= parameters

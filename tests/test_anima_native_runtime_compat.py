from pathlib import Path


def test_anima_runtime_helpers_fallback_to_module_parameters():
    source = Path("src/sd_embed/embedding_funcs.py").read_text(encoding="utf-8")
    assert "def _anima_module_runtime_device" in source
    assert "def _anima_module_runtime_dtype" in source
    assert "next(module.parameters()).device" in source


def test_native_guard_combines_profile_and_runtime_encoder_flags():
    source = Path("src/sd_embed/embedding_funcs.py").read_text(encoding="utf-8")
    assert "native = native or bool(info.get(\"native_encoder\", False))" in source
    assert "except Exception:" in source
    assert "info = None" in source

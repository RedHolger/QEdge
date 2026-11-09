"""Configuration and pinned metadata for QEdge Real-Model Reference Runtime."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Pinned primary model metadata
SMOLLM2_135M_ID: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
SMOLLM2_135M_REVISION: str = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
SMOLLM2_135M_LICENSE: str = "Apache-2.0"

# Canonical expected SHA256 hashes for all execution-affecting checkpoint files
SMOLLM2_135M_EXPECTED_HASHES: Dict[str, str] = {
    "config.json": "8eb740e8bbe4cff95ea7b4588d17a2432deb16e8075bc5828ff7ba9be94d982a",
    "generation_config.json": "87b916edaaab66b3899b9d0dd0752727dff6666686da0504d89ae0a6e055a013",
    "model.safetensors": "5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c",
    "special_tokens_map.json": "2b7379f3ae813529281a5c602bc5a11c1d4e0a99107aaa597fe936c1e813ca52",
    "tokenizer.json": "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c",
    "tokenizer_config.json": "4ec77d44f62efeb38d7e044a1db318f6a939438425312dfa333b8382dbad98df",
    "vocab.json": "82b84012e3add4d01d12ba14442026e49b8cbbaead1f79ecf3d919784f82dc79",
    "merges.txt": "0b54e8aa4e53d5383e2e4bc635a56b43f9647f7b13832d5d9ecd8f82dac4f510",
}

# Expected SHA256 digest of canonical SmolLM2 chat template extracted from verified tokenizer_config.json
SMOLLM2_135M_EXPECTED_TEMPLATE_DIGEST: str = "872be49dbb638044ad01b60388f48d469ff2980e5f0dccdc22ec907db54d0788"

# Harmless metadata and documentation files allowed alongside execution assets
HARMLESS_METADATA_FILENAMES: frozenset[str] = frozenset({
    "acquisition_manifest.json",
    "README",
    "README.MD",
    "README.TXT",
    "LICENSE",
    "LICENSE.MD",
    "LICENSE.TXT",
    "NOTICE",
    "NOTICE.TXT",
    "CHANGELOG",
    "CHANGELOG.MD",
    "CITATION.CFF",
    "COPYING",
})

# Extensions that indicate potential execution-affecting loader overrides or model code
KNOWN_LOADER_OVERRIDE_EXTENSIONS: frozenset[str] = frozenset({
    ".jinja", ".model", ".bin", ".safetensors", ".pt", ".pth", ".ckpt", ".py", ".msgpack", ".h5"
})

# Required files that must exist for execution
REQUIRED_ASSET_FILES: Tuple[str, ...] = tuple(sorted(SMOLLM2_135M_EXPECTED_HASHES.keys()))

# Estimated asset sizes in bytes
SMOLLM2_135M_ESTIMATED_SIZES: Dict[str, int] = {
    "model.safetensors": 269_060_552,   # ~269.1 MB
    "tokenizer.json": 2_104_556,        # ~2.1 MB
    "vocab.json": 800_662,              # ~800 KB
    "merges.txt": 466_391,              # ~466 KB
    "tokenizer_config.json": 3_764,
    "config.json": 861,
    "special_tokens_map.json": 655,
    "generation_config.json": 132,
}
SMOLLM2_135M_TOTAL_ESTIMATED_BYTES: int = sum(SMOLLM2_135M_ESTIMATED_SIZES.values())


def get_default_model_dir() -> Path:
    """Return the unified canonical model directory path.
    
    Checks QEDGE_MODEL_DIR environment variable first, then locates
    llm-inference/models/smollm2-135m-instruct relative to cwd or repo root.
    """
    env_dir = os.environ.get("QEDGE_MODEL_DIR")
    if env_dir:
        return Path(env_dir).resolve()

    cwd = Path.cwd().resolve()
    # Check directly from repo root
    cand1 = cwd / "llm-inference" / "models" / "smollm2-135m-instruct"
    if cand1.is_dir():
        return cand1
    # Check if inside llm-inference/runtime
    cand2 = cwd.parent / "models" / "smollm2-135m-instruct"
    if cand2.is_dir():
        return cand2
    # Check relative to this file
    cand3 = Path(__file__).resolve().parents[3] / "models" / "smollm2-135m-instruct"
    return cand3


@dataclass(frozen=True)
class ModelArchitectureConfig:
    """Canonical model architecture settings loaded from checkpoint."""
    model_id: str = SMOLLM2_135M_ID
    revision: str = SMOLLM2_135M_REVISION
    license: str = SMOLLM2_135M_LICENSE
    model_type: str = "llama"
    vocab_size: int = 49152
    hidden_size: int = 576
    intermediate_size: int = 1536
    num_hidden_layers: int = 30
    num_attention_heads: int = 9
    num_key_value_heads: int = 3
    head_dim: int = 64
    max_position_embeddings: int = 8192
    rope_theta: float = 100000.0
    rope_interleaved: bool = False
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 2
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True
    torch_dtype: str = "bfloat16"

    @classmethod
    def from_checkpoint_dict(
        cls,
        data: Dict[str, Any],
        model_id: str = SMOLLM2_135M_ID,
        revision: str = SMOLLM2_135M_REVISION,
    ) -> ModelArchitectureConfig:
        hidden_size = int(data.get("hidden_size", 576))
        num_heads = int(data.get("num_attention_heads", 9))
        head_dim = hidden_size // num_heads
        return cls(
            model_id=model_id,
            revision=revision,
            license=SMOLLM2_135M_LICENSE,
            model_type=str(data.get("model_type", "llama")),
            vocab_size=int(data.get("vocab_size", 49152)),
            hidden_size=hidden_size,
            intermediate_size=int(data.get("intermediate_size", 1536)),
            num_hidden_layers=int(data.get("num_hidden_layers", 30)),
            num_attention_heads=num_heads,
            num_key_value_heads=int(data.get("num_key_value_heads", 3)),
            head_dim=head_dim,
            max_position_embeddings=int(data.get("max_position_embeddings", 8192)),
            rope_theta=float(data.get("rope_theta", 100000.0)),
            rope_interleaved=bool(data.get("rope_interleaved", False)),
            bos_token_id=int(data.get("bos_token_id", 1)),
            eos_token_id=int(data.get("eos_token_id", 2)),
            pad_token_id=int(data.get("pad_token_id", 2)),
            rms_norm_eps=float(data.get("rms_norm_eps", 1e-5)),
            tie_word_embeddings=bool(data.get("tie_word_embeddings", True)),
            torch_dtype=str(data.get("torch_dtype", "bfloat16")),
        )


def format_smollm_chat_prompt(prompt: str, system_prompt: Optional[str] = None) -> str:
    """Format prompt according to SmolLM2 chat template:
    <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n
    """
    sys_content = system_prompt or "You are a helpful AI assistant named SmolLM, trained by Hugging Face"
    return f"<|im_start|>system\n{sys_content}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


def compute_file_sha256(path: Path) -> str:
    """Compute SHA256 hex digest of a local file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def compute_directory_hashes(model_dir: Path) -> Dict[str, str]:
    """Compute SHA256 digests for all execution-affecting files present in model_dir."""
    hashes: Dict[str, str] = {}
    for filename in REQUIRED_ASSET_FILES:
        target = model_dir / filename
        if target.is_file():
            hashes[filename] = compute_file_sha256(target)
    return hashes


def verify_directory_hashes(
    model_dir: Path,
    expected_hashes: Optional[Dict[str, str]] = None,
) -> Tuple[bool, List[str]]:
    """Verify hashes of execution-affecting files against trusted expectations.
    
    Distinguishes calculating hashes from verifying against trusted expectations.
    """
    targets = expected_hashes or SMOLLM2_135M_EXPECTED_HASHES
    errors: List[str] = []
    for filename, exp_hash in targets.items():
        target = model_dir / filename
        if not target.is_file():
            errors.append(f"Missing execution file: {filename}")
            continue
        actual_hash = compute_file_sha256(target)
        if actual_hash != exp_hash:
            errors.append(
                f"Hash mismatch on {filename}: expected {exp_hash[:16]}..., got {actual_hash[:16]}..."
            )
    return len(errors) == 0, errors

from dataclasses import dataclass, field
from typing import Literal

@dataclass
class ModelConfig:
    vocab_size: int = 10_000
    context_length: int = 256
    d_model: int = 512
    num_layers: int = 4
    num_heads: int = 16
    d_ff: int = 1344
    rope_theta: float = 10_000


@dataclass
class OptimConfig:
    lr: float = 0.001
    weight_decay: float = 0.1
    beta_1: float = 0.9
    beta_2: float = 0.999
    eps: float = 1e-5


@dataclass(kw_only=True)
class TrainingConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)

    train_data: str
    val_data: str

    batch_size: int = 4
    total_steps: int | None = None
    total_tokens: int | None = None
    val_interval: int = 10

    wandb: bool = False
    wandb_project: str = "cs336-1"
    log_interval: int = 100
    checkpoint_interval: int = 10_000

    device: Literal["mps", "cuda", "cpu"] = "mps"
    seed: int = 42

@dataclass(kw_only=True)
class DecodingConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)

    prompt: str

    checkpoint: str = "runs/train_20260418_221619/ckpt_step_9900.pt"
    device: Literal["mps", "cuda", "cpu"] = "mps"

    vocab_filepath: str = "runs/TinyStoriesV2-GPT4-train_vocab10000_20260413_180136/vocab.pkl"
    merges_filepath: str = "runs/TinyStoriesV2-GPT4-train_vocab10000_20260413_180136/merges.pkl"

    special_tokens: list[str] = field(default_factory=lambda: ["<|endoftext|>"])

    max_tokens: int = 2048
    temperature: float = 1.0
    top_p: float | None = None

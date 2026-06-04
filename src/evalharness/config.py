from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, field_validator


class ModelConfig(BaseModel):
    id: str
    label: str


class JudgeModelConfig(BaseModel):
    id: str
    label: str


class RunConfig(BaseModel):
    temperature: float = 0.0
    max_examples: int = 80
    concurrency: int = 4
    repeats: int = 1
    cache: bool = True
    smoke_test: bool = False
    judge_sample_size: int = 20


class Config(BaseModel):
    task: str
    dataset_path: str
    human_quality_path: str
    models: list[ModelConfig]
    judge_model: JudgeModelConfig
    prompts: list[str]
    run: RunConfig

    @field_validator("models")
    @classmethod
    def at_least_one_model(cls, v: list[ModelConfig]) -> list[ModelConfig]:
        if not v:
            raise ValueError("config.yaml must define at least one model")
        return v

    @field_validator("prompts")
    @classmethod
    def at_least_one_prompt(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("config.yaml must define at least one prompt")
        return v


def load_config(path: str | Path = "config.yaml") -> Config:
    """Load and validate config.yaml, resolving paths relative to the config file."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Config.model_validate(data)

"""Shared structured configs for simulation, training, evaluation, and GUI launching.

This module keeps the launcher, CLI entry points, and backend code aligned on a
single config representation instead of long ad-hoc argument strings.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import MISSING, asdict, dataclass, fields
from pathlib import Path
from typing import Any, Iterable, get_args, get_origin, get_type_hints


SCENARIOS = ("forward", "reverse", "forward_obs", "reverse_obs")
OBS_CHOICES = ("state", "lidar", "bev")
ENCODER_MODES = ("scratch", "state_only", "scaled_cnn", "ae_frozen", "ae_unfrozen", "unet_frozen", "unet_unfrozen")
FORWARD_REWARD_MODES = ("dense", "tractor_focus", "multiplicative", "guided")
REVERSE_REWARD_MODES = ("dense", "no_hitch", "multiplicative", "guided")
ALL_REWARD_MODES = ("dense", "tractor_focus", "no_hitch", "multiplicative", "guided")
CONTROLLER_CHOICES = ("fpp", "rpp", "pid", "mpc")


@dataclass(frozen=True)
class FieldSpec:
    name: str
    label: str
    field_type: str
    choices: tuple[str, ...] | None = None
    min_value: int | float | None = None
    max_value: int | float | None = None
    help_text: str = ""


class JsonConfigMixin:
    """Dataclass mixin with JSON helpers."""

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        type_hints = get_type_hints(cls)
        values: dict[str, Any] = {}
        for dc_field in fields(cls):
            if dc_field.name in data:
                raw = data[dc_field.name]
            elif dc_field.default is not MISSING:
                raw = dc_field.default
            elif dc_field.default_factory is not MISSING:  # type: ignore[attr-defined]
                raw = dc_field.default_factory()  # type: ignore[misc]
            else:
                continue

            values[dc_field.name] = _coerce_value(raw, type_hints.get(dc_field.name))
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key, value in list(data.items()):
            if isinstance(value, Path):
                data[key] = str(value)
        return data

    @classmethod
    def load_json(cls, path: Path):
        with path.open() as f:
            return cls.from_dict(json.load(f))

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)
            f.write("\n")


@dataclass
class RunModelConfig(JsonConfigMixin):
    scenario: str = "forward"
    obs: str = "state"
    reward: str = "dense"
    encoder: str = "scratch"
    encoder_path: str | None = None
    lidar_beams: int = 16
    controller: str | None = None
    model: str | None = None
    episodes: int = 10
    render: bool = False


@dataclass
class TrainConfig(JsonConfigMixin):
    scenario: str = "forward"
    obs: str = "state"
    reward: str = "dense"
    encoder: str = "scratch"
    encoder_path: str | None = None
    timesteps: int = 200_000
    n_envs: int = 1
    lidar_beams: int = 16
    device: str = "auto"
    pretrain_steps: int = 10_000
    pretrain_epochs: int = 30
    eval_episodes: int = 10
    eval_freq: int = 10_000
    normalized_eval_freq: int = 30_000
    image_scale_start: float = 0.0
    image_scale_end: float = 1.0
    image_scale_warmup_steps: int = 20_000
    image_scale_ramp_steps: int = 100_000
    state_scale_start: float = 1.0
    state_scale_end: float = 1.0
    state_scale_warmup_steps: int = 120_000
    state_scale_ramp_steps: int = 100_000
    render: bool = False
    retry_on_failure: bool = False


@dataclass
class EvalConfig(JsonConfigMixin):
    scenario: str = "forward"
    obs: str = "state"
    reward: str = "dense"
    encoder: str = "scratch"
    encoder_path: str | None = None
    lidar_beams: int = 16
    model: str | None = None
    episodes: int = 20
    output_csv: Path | None = None
    retry_on_failure: bool = False


RUN_MODEL_FIELDS = (
    FieldSpec("scenario", "Scenario", "choice", SCENARIOS),
    FieldSpec("obs", "Observation", "choice", OBS_CHOICES),
    FieldSpec("reward", "Reward", "choice", ALL_REWARD_MODES),
    FieldSpec("encoder", "Encoder", "choice", ENCODER_MODES),
    FieldSpec("encoder_path", "Encoder Path", "path"),
    FieldSpec("lidar_beams", "Lidar Beams", "int", min_value=1),
    FieldSpec("controller", "Controller", "choice_optional", CONTROLLER_CHOICES),
    FieldSpec("model", "Model Path", "path"),
    FieldSpec("episodes", "Episodes", "int", min_value=1),
    FieldSpec("render", "Render", "bool"),
)

TRAIN_FIELDS = (
    FieldSpec("scenario", "Scenario", "choice", SCENARIOS),
    FieldSpec("obs", "Observation", "choice", OBS_CHOICES),
    FieldSpec("reward", "Reward", "choice", ALL_REWARD_MODES),
    FieldSpec("encoder", "Encoder", "choice", ENCODER_MODES),
    FieldSpec("encoder_path", "Encoder Path", "path"),
    FieldSpec("timesteps", "Timesteps", "int", min_value=1),
    FieldSpec("n_envs", "Parallel Envs", "int", min_value=1),
    FieldSpec("lidar_beams", "Lidar Beams", "int", min_value=1),
    FieldSpec("device", "Device", "str"),
    FieldSpec("pretrain_steps", "Pretrain Steps", "int", min_value=0),
    FieldSpec("pretrain_epochs", "Pretrain Epochs", "int", min_value=0),
    FieldSpec("eval_episodes", "Eval Episodes", "int", min_value=1),
    FieldSpec("eval_freq", "Eval Freq", "int", min_value=1),
    FieldSpec("normalized_eval_freq", "Norm Eval Freq", "int", min_value=1),
    FieldSpec("image_scale_start", "Image Scale Start", "float", min_value=0.0),
    FieldSpec("image_scale_end", "Image Scale End", "float", min_value=0.0),
    FieldSpec("image_scale_warmup_steps", "Image Scale Warmup", "int", min_value=0),
    FieldSpec("image_scale_ramp_steps", "Image Scale Ramp", "int", min_value=1),
    FieldSpec("state_scale_start", "State Scale Start", "float", min_value=0.0),
    FieldSpec("state_scale_end", "State Scale End", "float", min_value=0.0),
    FieldSpec("state_scale_warmup_steps", "State Scale Warmup", "int", min_value=0),
    FieldSpec("state_scale_ramp_steps", "State Scale Ramp", "int", min_value=1),
    FieldSpec("render", "Render", "bool"),
    FieldSpec("retry_on_failure", "Retry On Failure", "bool"),
)

EVAL_FIELDS = (
    FieldSpec("scenario", "Scenario", "choice", SCENARIOS),
    FieldSpec("obs", "Observation", "choice", OBS_CHOICES),
    FieldSpec("reward", "Reward", "choice", ALL_REWARD_MODES),
    FieldSpec("encoder", "Encoder", "choice", ENCODER_MODES),
    FieldSpec("encoder_path", "Encoder Path", "path"),
    FieldSpec("lidar_beams", "Lidar Beams", "int", min_value=1),
    FieldSpec("model", "Model Path", "path"),
    FieldSpec("episodes", "Episodes", "int", min_value=1),
    FieldSpec("output_csv", "Output CSV", "path"),
    FieldSpec("retry_on_failure", "Retry On Failure", "bool"),
)


def reward_choices_for_scenario(scenario: str) -> tuple[str, ...]:
    return REVERSE_REWARD_MODES if "reverse" in scenario else FORWARD_REWARD_MODES


def _coerce_value(value: Any, annotation: Any) -> Any:
    if annotation is None:
        return value

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is not None and type(None) in args:
        if value in (None, ""):
            return None
        non_none_args = [arg for arg in args if arg is not type(None)]
        if len(non_none_args) == 1:
            return _coerce_value(value, non_none_args[0])

    if annotation is Path:
        return value if isinstance(value, Path) or value in (None, "") else Path(value)
    if annotation is int:
        return value if isinstance(value, int) else int(value)
    if annotation is float:
        return value if isinstance(value, float) else float(value)
    if annotation is bool:
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"true", "1", "yes", "on"}:
            return True
        if text in {"false", "0", "no", "off"}:
            return False
    if annotation is str:
        return None if value is None else str(value)
    return value


def obs_tag(obs: str, encoder: str = "scratch", lidar_beams: int = 16) -> str:
    if obs == "bev" and encoder != "scratch":
        return f"{obs}_{encoder}"
    if obs == "lidar" and lidar_beams != 16:
        return f"lidar_{lidar_beams}"
    return obs


def extract_config_path(argv: Iterable[str] | None) -> Path | None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=None)
    parsed, _ = pre_parser.parse_known_args(list(argv) if argv is not None else None)
    return parsed.config


def apply_config_file_defaults(
    parser: argparse.ArgumentParser,
    argv: Iterable[str] | None,
    config_cls: type[JsonConfigMixin],
) -> Path | None:
    config_path = extract_config_path(argv)
    if config_path is None:
        return None
    loaded = config_cls.load_json(config_path)
    parser.set_defaults(**loaded.to_dict())
    return config_path


def add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a JSON config file. CLI flags override values loaded from the file.",
    )


def validate_common_config(scenario: str, obs: str, reward: str, encoder: str) -> list[str]:
    errors: list[str] = []
    if not scenario:
        errors.append("Scenario is required.")
    elif scenario not in SCENARIOS:
        errors.append(f"Unknown scenario: {scenario}")
    if not obs:
        errors.append("Observation type is required.")
    elif obs not in OBS_CHOICES:
        errors.append(f"Unknown observation type: {obs}")
    if not encoder:
        errors.append("Encoder mode is required.")
    elif encoder not in ENCODER_MODES:
        errors.append(f"Unknown encoder mode: {encoder}")
    if encoder != "scratch" and obs != "bev":
        errors.append("BEV encoder/ablation modes only apply when obs='bev'.")
    if not reward:
        errors.append("Reward is required.")
    elif scenario in SCENARIOS and reward not in reward_choices_for_scenario(scenario):
        allowed = ", ".join(reward_choices_for_scenario(scenario))
        errors.append(f"Reward '{reward}' is invalid for scenario '{scenario}'. Choose from: {allowed}.")
    return errors


def validate_run_model_config(config: RunModelConfig) -> list[str]:
    errors = validate_common_config(config.scenario, config.obs, config.reward, config.encoder)
    if config.episodes < 1:
        errors.append("Episodes must be at least 1.")
    if config.controller and config.controller not in CONTROLLER_CHOICES:
        errors.append(f"Unknown controller: {config.controller}")
    if config.controller == "fpp" and "reverse" in config.scenario:
        errors.append("Controller 'fpp' is only valid for forward scenarios.")
    if config.controller == "rpp" and "reverse" not in config.scenario:
        errors.append("Controller 'rpp' is only valid for reverse scenarios.")
    return errors


def validate_train_config(config: TrainConfig) -> list[str]:
    errors = validate_common_config(config.scenario, config.obs, config.reward, config.encoder)
    if config.timesteps < 1:
        errors.append("Timesteps must be at least 1.")
    if config.n_envs < 1:
        errors.append("Parallel env count must be at least 1.")
    if config.eval_episodes < 1:
        errors.append("Eval episodes must be at least 1.")
    if config.eval_freq < 1 or config.normalized_eval_freq < 1:
        errors.append("Evaluation frequencies must be at least 1.")
    if config.render and config.n_envs > 1:
        errors.append("Rendering during training only works sensibly with n_envs=1.")
    return errors


def validate_eval_config(config: EvalConfig) -> list[str]:
    errors = validate_common_config(config.scenario, config.obs, config.reward, config.encoder)
    if config.episodes < 1:
        errors.append("Episodes must be at least 1.")
    return errors

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from termcolor import colored

from lerobot.common.wandb_utils import cfg_to_group
from lerobot.utils.constants import PRETRAINED_MODEL_DIR

if TYPE_CHECKING:
    from lerobot.configs.train import TrainPipelineConfig


def _build_mlflow_tags(cfg: "TrainPipelineConfig") -> dict[str, str]:
    """Build a dictionary of MLflow run tags from the training configuration."""
    raw_tags = cfg_to_group(cfg, return_list=True)
    tags: dict[str, str] = {"group": cfg_to_group(cfg, return_list=False)}  # type: ignore[dict-item]
    for item in raw_tags:
        if ":" in item:
            key, val = item.split(":", 1)
            tags[key] = val
    return tags


class MLflowLogger:
    """A helper class to log metrics, configs, and artifacts using MLflow."""

    def __init__(self, cfg: "TrainPipelineConfig"):
        self.cfg = cfg.mlflow
        self.log_dir = cfg.output_dir
        self.job_name = cfg.job_name
        self.env_fps = cfg.env.fps if cfg.env else None
        self._group = cfg_to_group(cfg)
        self._active_run: Any = None

        import mlflow

        # Resolve tracking URI without writing environment secrets back into cfg.mlflow.tracking_uri,
        # so credentials in MLFLOW_TRACKING_URI are never serialized into train_config.json or stdout logs.
        env_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
        effective_artifact_root = self.cfg.local_artifact_root

        if self.cfg.tracking_uri:
            mlflow.set_tracking_uri(self.cfg.tracking_uri)
        elif env_tracking_uri:
            mlflow.set_tracking_uri(env_tracking_uri)
        else:
            mlflow_dir = Path(self.log_dir) / "mlflow"
            mlflow_dir.mkdir(parents=True, exist_ok=True)
            mlflow.set_tracking_uri(f"sqlite:///{mlflow_dir / 'mlflow.db'}")
            if effective_artifact_root is None:
                effective_artifact_root = str(mlflow_dir / "artifacts")

        if effective_artifact_root:
            from mlflow.store.artifact.artifact_repository_registry import _artifact_repository_registry
            from mlflow.store.artifact.local_artifact_repo import LocalArtifactRepository

            Path(effective_artifact_root).mkdir(parents=True, exist_ok=True)
            base_uri = f"file://{str(effective_artifact_root).rstrip('/')}"
            _artifact_repository_registry.register(
                "mlflow-artifacts",
                lambda artifact_uri, *args, **kwargs: LocalArtifactRepository(
                    artifact_uri.replace("mlflow-artifacts:", base_uri, 1)
                ),
            )

        experiment_name = self.cfg.experiment_name
        if mlflow.get_experiment_by_name(experiment_name) is None:
            if effective_artifact_root:
                mlflow.create_experiment(
                    name=experiment_name,
                    artifact_location=f"mlflow-artifacts:/{experiment_name}",
                )
            else:
                mlflow.create_experiment(name=experiment_name)
        mlflow.set_experiment(experiment_name)

        run_id = self.cfg.run_id if (self.cfg.run_id or cfg.resume) else None
        run_name = None if run_id else (self.cfg.run_name or self.job_name)
        tags = _build_mlflow_tags(cfg) if self.cfg.add_tags else None

        self._active_run = mlflow.start_run(
            run_id=run_id,
            run_name=run_name,
            tags=tags,
            description=self.cfg.notes,
        )
        cfg.mlflow.run_id = self._active_run.info.run_id
        self._mlflow = mlflow

        if run_id is None:
            try:
                self._mlflow.log_dict(cfg.to_dict(), "train_config.json")
            except Exception as e:
                logging.warning(f"Failed to log train_config.json artifact to MLflow: {e}")

        logging.info(
            colored(
                f"Logs will be synced with MLflow (experiment={experiment_name}, run_id={cfg.mlflow.run_id}).",
                "blue",
                attrs=["bold"],
            )
        )

    def log_policy(self, checkpoint_dir: Path):
        """Logs the checkpointed policy to MLflow artifacts."""
        if self.cfg.disable_artifact:
            return

        step_id = checkpoint_dir.name
        pretrained_model_dir = checkpoint_dir / PRETRAINED_MODEL_DIR
        artifact_path = f"checkpoints/{step_id}"

        adapter_model_file = pretrained_model_dir / "adapter_model.safetensors"
        standard_model_file = pretrained_model_dir / SAFETENSORS_SINGLE_FILE

        try:
            if adapter_model_file.exists():
                self._mlflow.log_artifact(str(adapter_model_file), artifact_path=artifact_path)
                adapter_config_file = pretrained_model_dir / "adapter_config.json"
                if adapter_config_file.exists():
                    self._mlflow.log_artifact(str(adapter_config_file), artifact_path=artifact_path)
                config_file = pretrained_model_dir / "config.json"
                if config_file.exists():
                    self._mlflow.log_artifact(str(config_file), artifact_path=artifact_path)
            elif standard_model_file.exists():
                self._mlflow.log_artifact(str(standard_model_file), artifact_path=artifact_path)
            else:
                logging.warning(
                    f"No {SAFETENSORS_SINGLE_FILE} or adapter_model.safetensors found in {pretrained_model_dir}. "
                    "Skipping model artifact upload to MLflow."
                )
        except Exception as e:
            logging.warning(f"Failed to log policy checkpoint to MLflow: {e}")

    def log_dict(
        self, d: dict, step: int | None = None, mode: str = "train", custom_step_key: str | None = None
    ):
        if mode not in {"train", "eval"}:
            raise ValueError(mode)
        if step is None and custom_step_key is None:
            raise ValueError("Either step or custom_step_key must be provided.")

        step_val = step if custom_step_key is None else d[custom_step_key]
        metrics: dict[str, float] = {}
        for k, v in d.items():
            if custom_step_key is not None and k == custom_step_key:
                continue
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            metrics[f"{mode}/{k}"] = float(v)

        if metrics:
            self._mlflow.log_metrics(metrics, step=step_val)

    def log_video(self, video_path: str, step: int, mode: str = "train"):
        if mode not in {"train", "eval"}:
            raise ValueError(mode)

        try:
            self._mlflow.log_artifact(str(video_path), artifact_path=f"{mode}/videos/step_{step}")
        except Exception as e:
            logging.warning(f"Failed to log video to MLflow: {e}")

    def close(self, status: str = "FINISHED"):
        """Ends the active MLflow run."""
        if self._active_run is not None:
            try:
                self._mlflow.end_run(status=status)
            finally:
                self._active_run = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        status = "FAILED" if exc_type is not None else "FINISHED"
        self.close(status=status)

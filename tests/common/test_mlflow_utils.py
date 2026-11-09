import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from lerobot.common.mlflow_utils import MLflowLogger
from lerobot.configs.default import MLflowConfig


def _make_mock_mlflow(run_id: str = "run-123"):
    mlflow = MagicMock()
    mlflow.get_experiment_by_name.return_value = None
    active_run = MagicMock()
    active_run.info.run_id = run_id
    mlflow.start_run.return_value = active_run
    return mlflow


def _make_train_cfg(tmp_path, mlflow_cfg: MLflowConfig | None = None, resume: bool = False):
    return SimpleNamespace(
        mlflow=mlflow_cfg or MLflowConfig(enable=True),
        output_dir=tmp_path,
        job_name="act_custom",
        env=None,
        policy=SimpleNamespace(type="act"),
        is_reward_model_training=False,
        seed=1000,
        dataset=SimpleNamespace(repo_id="intrinsic/rl_transitions"),
        resume=resume,
        to_dict=lambda: {"job_name": "act_custom", "seed": 1000},
    )


def test_mlflow_config_defaults():
    cfg = MLflowConfig()

    assert cfg.enable is False
    assert cfg.experiment_name == "lerobot"
    assert cfg.run_name is None
    assert cfg.run_id is None
    assert cfg.tracking_uri is None
    assert cfg.local_artifact_root is None
    assert cfg.disable_artifact is True
    assert cfg.add_tags is True
    assert cfg.notes is None


def test_mlflow_logger_uses_env_tracking_uri_without_leaking_to_cfg(monkeypatch, tmp_path):
    mlflow = _make_mock_mlflow(run_id="cloud-run-id")
    artifact_registry = MagicMock()
    local_repo_cls = MagicMock()

    monkeypatch.setitem(sys.modules, "mlflow", mlflow)
    monkeypatch.setitem(
        sys.modules,
        "mlflow.store.artifact.artifact_repository_registry",
        SimpleNamespace(_artifact_repository_registry=artifact_registry),
    )
    monkeypatch.setitem(
        sys.modules,
        "mlflow.store.artifact.local_artifact_repo",
        SimpleNamespace(LocalArtifactRepository=local_repo_cls),
    )
    secret_uri = "postgresql+psycopg2://user:super_secret@10.0.0.1:5432/mlflow"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", secret_uri)

    artifact_root = tmp_path / "mlflow-artifacts"
    mlflow_cfg = MLflowConfig(
        enable=True,
        experiment_name="lerobot-exp",
        run_name="wf-12345",
        local_artifact_root=str(artifact_root),
    )
    cfg = _make_train_cfg(tmp_path, mlflow_cfg=mlflow_cfg)

    logger = MLflowLogger(cfg)

    # Verify tracking URI was passed to mlflow but NEVER stored on cfg.mlflow.tracking_uri
    mlflow.set_tracking_uri.assert_called_once_with(secret_uri)
    assert cfg.mlflow.tracking_uri is None
    assert cfg.mlflow.run_id == "cloud-run-id"

    # Verify artifact repository registration and experiment creation
    artifact_registry.register.assert_called_once()
    mlflow.create_experiment.assert_called_once_with(
        name="lerobot-exp",
        artifact_location="mlflow-artifacts:/lerobot-exp",
    )
    mlflow.set_experiment.assert_called_once_with("lerobot-exp")
    mlflow.start_run.assert_called_once()
    assert mlflow.start_run.call_args.kwargs["run_name"] == "wf-12345"
    assert mlflow.start_run.call_args.kwargs["run_id"] is None
    mlflow.log_dict.assert_called_once_with({"job_name": "act_custom", "seed": 1000}, "train_config.json")

    logger.close()
    mlflow.end_run.assert_called_once_with(status="FINISHED")


def test_mlflow_logger_resumes_existing_run(monkeypatch, tmp_path):
    mlflow = _make_mock_mlflow(run_id="existing-run-id")
    monkeypatch.setitem(sys.modules, "mlflow", mlflow)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

    mlflow_cfg = MLflowConfig(enable=True, run_id="existing-run-id", run_name="wf-12345")
    cfg = _make_train_cfg(tmp_path, mlflow_cfg=mlflow_cfg, resume=True)

    MLflowLogger(cfg)

    assert mlflow.start_run.call_args.kwargs["run_id"] == "existing-run-id"
    assert mlflow.start_run.call_args.kwargs["run_name"] is None
    mlflow.log_dict.assert_not_called()


def test_mlflow_logger_log_dict_batches_scalars(monkeypatch, tmp_path):
    mlflow = _make_mock_mlflow()
    monkeypatch.setitem(sys.modules, "mlflow", mlflow)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

    cfg = _make_train_cfg(tmp_path)
    logger = MLflowLogger(cfg)

    logger.log_dict(
        {
            "loss": 0.25,
            "lr": 1e-4,
            "steps": 100,
            "flag": True,
            "nested": {"ignored": 1.0},
        },
        step=100,
        mode="train",
    )

    mlflow.log_metrics.assert_called_once_with(
        {"train/loss": 0.25, "train/lr": 1e-4, "train/steps": 100.0},
        step=100,
    )

    with pytest.raises(ValueError, match="invalid"):
        logger.log_dict({"loss": 1.0}, step=1, mode="invalid")

    with pytest.raises(ValueError, match="Either step or custom_step_key"):
        logger.log_dict({"loss": 1.0}, step=None, mode="train")

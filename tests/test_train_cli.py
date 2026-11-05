"""Training CLI forwards explicit early stopping and preserves saved resume settings."""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train_dual


@pytest.fixture
def training_call(tmp_path, monkeypatch):
    data = tmp_path / "paired.yaml"
    data.write_text("channels: 6\n", encoding="utf-8")
    observed = {}

    class Model:
        def __init__(self, config):
            observed["model"] = config

        def train(self, **kwargs):
            observed["train"] = kwargs

    monkeypatch.setattr(train_dual, "YOLO", Model)
    monkeypatch.setattr(sys, "argv", ["train_dual.py", "--device", "cpu", "--data", str(data), "--scratch"])
    return observed


@pytest.mark.parametrize("extra,expected", [([], 0), (["--patience", "30"], 30)])
def test_new_run_early_stopping_and_mala_selection(training_call, extra, expected):
    sys.argv += ["--fusion", "defa-mala-core", *extra]
    train_dual.main()
    assert Path(training_call["model"]).name == "yolo26n-defa-mala-core.yaml"
    assert training_call["train"]["patience"] == expected


def test_resume_does_not_override_checkpoint_patience(training_call):
    sys.argv += ["--resume", "saved.pt", "--patience", "30"]
    train_dual.main()
    assert training_call["model"] == "saved.pt"
    assert training_call["train"] == {"resume": True, "device": "cpu"}


def test_negative_patience_fails_before_model_construction(training_call, capsys):
    sys.argv += ["--patience", "-1"]
    with pytest.raises(SystemExit) as exc:
        train_dual.main()
    assert exc.value.code == 2
    assert "--patience must be nonnegative" in capsys.readouterr().err
    assert "model" not in training_call

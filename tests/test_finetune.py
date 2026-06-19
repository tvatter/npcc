"""Tests for ``npcc.finetune`` (TabPFN meta-training harness).

Config/save-name/transform/evaluate checks are hermetic; the full meta-train +
load round-trip is gated on a real TabPFN (``TABPFN_TOKEN`` + cached checkpoint),
mirroring ``test_pfnr_bicop.py::test_real_tabpfn_smoke``.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch

from npcc.eval import ConditionalGridSpec
from npcc.finetune.config import FinetuneConfig
from npcc.finetune.evaluate import ClaytonReference, conditional_report
from npcc.finetune.loop import _transform_targets
from npcc.finetune.save_load import save_finetuned


class TestFinetuneConfig:
  def test_smoke_preset_is_tiny_cpu(self) -> None:
    cfg = FinetuneConfig.smoke()
    assert cfg.device == "cpu"
    assert cfg.n_datasets == 2
    assert cfg.max_steps == 2
    assert "v2.5" in cfg.checkpoint_name

  @pytest.mark.parametrize(
    "make",
    [
      lambda: FinetuneConfig(n_datasets=0),
      lambda: FinetuneConfig(rows_per_dataset=3),
      lambda: FinetuneConfig(query_fraction=0.0),
      lambda: FinetuneConfig(query_fraction=1.0),
      lambda: FinetuneConfig(epochs=0),
      lambda: FinetuneConfig(n_estimators_finetune=0),
      lambda: FinetuneConfig(checkpoint_name="finetuned.pth"),
    ],
  )
  def test_validation_raises(self, make: Callable[[], object]) -> None:
    with pytest.raises(ValueError):
      make()


class TestTransformTargets:
  def test_identity(self) -> None:
    y = np.array([0.2, 0.5, 0.8])
    np.testing.assert_allclose(_transform_targets(y, "identity", 1e-6), y)

  def test_logit(self) -> None:
    y = np.array([0.25, 0.5, 0.75])
    expected = np.log(y / (1.0 - y))
    np.testing.assert_allclose(_transform_targets(y, "logit", 1e-6), expected)

  def test_probit_matches_inverse_normal_cdf(self) -> None:
    y = np.array([0.5])
    assert _transform_targets(y, "probit", 1e-6)[0] == pytest.approx(
      0.0, abs=1e-9
    )

  def test_unknown_transform_raises(self) -> None:
    with pytest.raises(ValueError):
      _transform_targets(np.array([0.5]), "bogus", 1e-6)


class TestSaveLoadNameRule:
  def test_rejects_non_v2_5_filename(self, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="v2.5"):
      save_finetuned(object(), tmp_path / "finetuned.pth")


class TestEvaluate:
  def test_clayton_reference_tau_schedule(self) -> None:
    ref = ClaytonReference(tau_min=0.1, tau_max=0.9, x_min=0.0, x_max=1.0)
    np.testing.assert_allclose(ref.tau_of_x(np.array([0.0, 1.0])), [0.1, 0.9])

  def test_clayton_reference_density_integrates_to_one(self) -> None:
    ref = ClaytonReference()
    g = (np.arange(60) + 0.5) / 60
    uu, vv = (a.ravel() for a in np.meshgrid(g, g, indexing="ij"))
    x = np.full(len(uu), 0.5)
    c = ref.true_density(uu, vv, x)
    assert c.mean() == pytest.approx(1.0, abs=0.05)

  def test_clayton_reference_requires_x(self) -> None:
    with pytest.raises(ValueError):
      ClaytonReference().true_density(np.array([0.5]), np.array([0.5]), None)

  def test_conditional_report_zero_when_equal(self) -> None:
    ref = ClaytonReference()
    spec = ConditionalGridSpec(
      u_levels=(0.3, 0.7),
      v_levels=(0.3, 0.7),
      x_values=np.array([[0.2], [0.8]]),
    )
    metrics = conditional_report(ref.true_density, ref.true_density, spec)
    assert metrics["ISE"] == pytest.approx(0.0)
    assert metrics["IAE"] == pytest.approx(0.0)
    assert metrics["KL"] == pytest.approx(0.0)


# ===========================================================================
# Real meta-train + load round-trip (opt-in via TABPFN_TOKEN; CPU is fine)
# ===========================================================================


def test_meta_train_and_load_roundtrip(tmp_path: Path) -> None:
  dotenv = pytest.importorskip("dotenv")
  dotenv.load_dotenv()
  if not os.getenv("TABPFN_TOKEN"):
    pytest.skip("TABPFN_TOKEN not set; skipping fine-tune integration test.")

  pytest.importorskip("tabpfn")
  pv = pytest.importorskip("pyvinecopulib")
  from tabpfn.errors import TabPFNLicenseError

  from npcc import PFNRBicop
  from npcc.finetune import meta_train

  cfg = FinetuneConfig.smoke(output_dir=tmp_path / "smoke")
  try:
    ckpt = meta_train(cfg)
  except TabPFNLicenseError as exc:
    pytest.skip(f"TabPFN authentication unavailable: {exc}")

  assert ckpt.exists() and "v2.5" in ckpt.name

  cop = pv.Bicop(family=pv.BicopFamily.clayton, parameters=np.array([[3.0]]))
  uv = cop.simulate(150, seeds=[1, 2, 3])
  model = PFNRBicop(method="criterion", finetuned_path=ckpt, device="cpu")
  model.fit(uv[:, 0], uv[:, 1])
  assert model.v_given_ux_.model_kwargs["model_path"] == str(ckpt)

  dens = model.pdf(np.array([0.3, 0.5, 0.7]), np.array([0.4, 0.5, 0.6]))
  assert dens.shape == (3,)
  assert np.all(np.isfinite(dens)) and np.all(dens > 0)


class TestClaytonReferenceSimulate:
  def test_simulate_shapes_and_range(self) -> None:
    ref = ClaytonReference()
    x, u, v = ref.simulate(400, np.random.default_rng(0))
    assert x.shape == (400,) and u.shape == (400,) and v.shape == (400,)
    assert u.min() >= 0.0 and u.max() <= 1.0
    assert v.min() >= 0.0 and v.max() <= 1.0


class TestCli:
  def test_build_config_smoke(self) -> None:
    import argparse

    from npcc.finetune.__main__ import _build_config

    args = argparse.Namespace(
      smoke=True,
      output_dir="runs/x",
      n_datasets=9,
      rows_per_dataset=9,
      epochs=9,
      learning_rate=1.0,
      n_estimators=9,
      freeze_backbone=True,
      device="cuda",
      transform="logit",
      seed=0,
    )
    cfg = _build_config(args)
    assert cfg.device == "cpu" and cfg.n_datasets == 2  # smoke preset wins
    assert (
      cfg.freeze_backbone is True
    )  # flag threads even into the smoke preset

  def test_build_config_full(self) -> None:
    import argparse

    from npcc.finetune.__main__ import _build_config

    args = argparse.Namespace(
      smoke=False,
      output_dir="runs/exp",
      n_datasets=16,
      rows_per_dataset=128,
      epochs=3,
      learning_rate=2e-5,
      n_estimators=4,
      freeze_backbone=False,
      device="cpu",
      transform="probit",
      seed=7,
    )
    cfg = _build_config(args)
    assert (
      cfg.n_datasets == 16 and cfg.epochs == 3 and cfg.transform == "probit"
    )

  def test_main_invokes_meta_train(
    self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
  ) -> None:
    import npcc.finetune.__main__ as cli

    captured: dict[str, FinetuneConfig] = {}

    def fake_meta_train(cfg: FinetuneConfig, prior: object) -> Path:
      captured["cfg"] = cfg
      return tmp_path / "finetuned_v2.5.pth"

    monkeypatch.setattr(cli, "meta_train", fake_meta_train)
    monkeypatch.setattr(
      "sys.argv", ["prog", "--smoke", "--output-dir", str(tmp_path)]
    )
    cli.main()
    assert captured["cfg"].device == "cpu"


class TestLazyImports:
  def test_getattr_resolves_public_names(self) -> None:
    import npcc.finetune as ft

    assert callable(ft.meta_train)
    assert callable(ft.build_base_regressor)
    assert callable(ft.save_finetuned)

  def test_getattr_unknown_raises(self) -> None:
    import npcc.finetune as ft

    with pytest.raises(AttributeError):
      _ = ft.does_not_exist


class TestTrainableParameters:
  """_trainable_parameters: full vs decoder-head-only (frozen backbone)."""

  @staticmethod
  def _toy_model() -> torch.nn.Module:
    model = torch.nn.Module()
    model.add_module("transformer_encoder", torch.nn.Linear(4, 4))  # backbone
    model.add_module(
      "decoder_dict", torch.nn.ModuleDict({"standard": torch.nn.Linear(4, 8)})
    )  # head
    return model

  def test_full_returns_all_params(self) -> None:
    from npcc.finetune.loop import _trainable_parameters

    model = self._toy_model()
    params = _trainable_parameters(model, freeze_backbone=False)
    assert len(params) == len(list(model.parameters()))

  def test_head_only_freezes_backbone(self) -> None:
    from npcc.finetune.loop import _trainable_parameters

    model = self._toy_model()
    params = _trainable_parameters(model, freeze_backbone=True)
    head = model.get_submodule("decoder_dict")
    backbone = model.get_submodule("transformer_encoder")
    assert {id(p) for p in params} == {id(p) for p in head.parameters()}
    assert all(not p.requires_grad for p in backbone.parameters())
    assert all(p.requires_grad for p in head.parameters())

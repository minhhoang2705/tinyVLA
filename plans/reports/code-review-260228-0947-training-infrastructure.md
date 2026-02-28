# Code Review: feat/training-infrastructure → master

**Branch:** `feat/training-infrastructure`
**Reviewer:** Automated strict engineering review
**Date:** 2026-02-28
**Files reviewed:** `src/vla/training/lightning_module.py`, `scripts/train.py`, `src/vla/data/datamodule_lightning.py`, `configs/train/default.yaml`, `tests/unit/test_training.py`

---

## Summary

Training infrastructure is structurally sound. `VLALightningModule` correctly wraps `VLAModel`, defers only trainable params to the optimizer, and uses cosine-warmup scheduling. `train.py` has clean function decomposition. The `VLADataModule` fix is correct. Three bugs and one policy violation require resolution before merge.

**Verdict: NEEDS FIXES (3 bugs, 1 policy violation)**

---

## Critical Issues

### 1. Committed training artifact — `csv_logs/` in git [POLICY VIOLATION]

`csv_logs/version_0/hparams.yaml` and `metrics.csv` were committed. `.gitignore` covers `logs/` and `lightning_logs/` but not `csv_logs/`. These are runtime training outputs — they do not belong in source control and will cause merge conflicts as logs rotate.

**Fix required:**
```
# Add to .gitignore
csv_logs/
```
Then remove from tracked files:
```bash
git rm -r --cached csv_logs/
```

### 2. EarlyStopping cannot be disabled from config [BUG]

`train.py:_build_callbacks()`:
```python
early_stop_cfg = train_cfg.get("early_stopping", {})
if early_stop_cfg:
    callbacks.append(EarlyStopping(...))
```
`configs/train/default.yaml` always defines `early_stopping: {patience: 10, min_delta: 0.001}`, so `early_stop_cfg` is always truthy. There is no way to disable early stopping short of removing the YAML key entirely. No `enabled: false` toggle exists.

**Fix required:** Add an `enabled` flag:
```yaml
# configs/train/default.yaml
early_stopping:
  enabled: true
  patience: 10
  min_delta: 0.001
```
```python
if early_stop_cfg.get("enabled", False):
    callbacks.append(EarlyStopping(...))
```

### 3. EarlyStopping monitor reads from wrong config section [BUG]

`train.py:_build_callbacks()`:
```python
EarlyStopping(
    monitor=checkpoint_cfg.get("monitor", "val/loss"),
    mode=checkpoint_cfg.get("mode", "min"),
)
```
The `monitor` and `mode` for `EarlyStopping` are sourced from `checkpoint_cfg` (the `checkpoint:` sub-key), not from `early_stopping`. This is silent coupling — if `checkpoint.monitor` is changed to e.g. `val/action_mse`, early stopping also silently changes target. They are logically independent concerns.

**Fix required:** Read `monitor`/`mode` from `early_stop_cfg` with fallback to checkpoint:
```python
EarlyStopping(
    monitor=early_stop_cfg.get("monitor", checkpoint_cfg.get("monitor", "val/loss")),
    mode=early_stop_cfg.get("mode", checkpoint_cfg.get("mode", "min")),
    ...
)
```

---

## Important Issues

### 4. `configure_optimizers` — overly broad exception [MODERATE]

`lightning_module.py:178-185`:
```python
try:
    total_steps = int(self.trainer.estimated_stepping_batches)
except Exception:
    total_steps = 100_000
    logger.warning(...)
```
`configure_optimizers` is called by PL _after_ the trainer is attached — `self.trainer` exists. The broad `except Exception` will silently swallow `AttributeError` from a typo or a real `RuntimeError` from misconfigured datamodule, masking root causes. Use a specific guard instead:

```python
if self.trainer is not None:
    total_steps = int(self.trainer.estimated_stepping_batches)
else:
    total_steps = 100_000
    logger.warning(...)
```

### 5. Dummy dataset default too small [MODERATE]

`VLADataModule.__init__` defaults `total_samples=1000` → 800 train / 200 val. `train.py._build_datamodule` passes `data_cfg.get("num_samples", 1000)` and no `num_samples` key exists in any config. Running `python scripts/train.py` with dummy data silently produces a trivially small training set. The old hardcoded `10000` was more appropriate for meaningful runs.

**Fix:** Add `num_samples: 10000` to the dummy data config or change default to `10000`.

### 6. Test dataset size not parameterized [MODERATE]

`datamodule_lightning.py:172`:
```python
test_size = 1000  # hardcoded, not self.total_samples
```
Train+val use `self.total_samples`; test still hardcodes 1000. Passing `total_samples=100` yields 80/20 train/val but 1000 test samples — impossible in practice, and will `IndexError` if `DummyVLADataset` generates fewer samples than the test loader requests.

### 7. `lightning_module.py` exceeds 200 LOC limit [MINOR]

File is 201 lines — one over the project's hard limit from `CLAUDE.md`. Extract `configure_optimizers` into a helper or split into two files.

---

## Minor Issues

### 8. Redundant hyperparameter assignment after `save_hyperparameters`

`lightning_module.py:62-64`:
```python
self.save_hyperparameters(ignore=["model_cfg"])
self.learning_rate = learning_rate   # redundant — already in self.hparams
self.weight_decay = weight_decay     # redundant
self.warmup_steps = warmup_steps     # redundant
```
`save_hyperparameters()` stores these in `self.hparams` and PL auto-creates attributes. The manual assignments are not wrong but create two sources of truth.

### 9. Union type syntax requires Python 3.10+

`lightning_module.py:72`:
```python
target_actions: torch.Tensor | None = None
```
Python 3.10+ syntax. Project is Python 3.8+ compatible per environment setup. Use `Optional[torch.Tensor]` for consistency with the rest of the codebase.

### 10. Config breaking change undocumented

`configs/train/default.yaml` renamed `optimizer.lr` → `learning_rate` and `optimizer.weight_decay` → `weight_decay`. Any external script reading `cfg.train.optimizer.lr` will silently use the wrong key. The `max_epochs` / `epochs` dual-key is noted as backwards compat but `optimizer.*` removal is not.

---

## Tests Assessment

**Coverage: Adequate for critical invariants**

| Test | Quality | Gap |
|------|---------|-----|
| Backbone frozen after init | ✓ | — |
| Optimizer excludes frozen params | ✓ | — |
| `training_step` returns scalar loss | ✓ | — |
| `val/loss` logged with `sync_dist=True` | ✓ | — |
| `from_dict` / `from_hydra` config paths | ✗ | No test |
| `configure_optimizers` fallback path | ✗ | No test |
| `_build_callbacks` with early stopping disabled | ✗ | No test |

---

## Positive Observations

- `_shared_step` DRY pattern is correct and clean
- `get_trainable_params()` delegation is architecturally sound
- Lazy import of `lerobot` in `datamodule_lightning.py` correctly avoids hard dependency
- `pin_memory=True` + `persistent_workers` DataLoader config is appropriate
- `pl.seed_everything(workers=True)` correctly seeds DataLoader workers

---

## Action Items Before Merge

| # | Severity | Action |
|---|----------|--------|
| 1 | Critical | Add `csv_logs/` to `.gitignore`, remove from git tracking |
| 2 | Bug | Add `enabled` flag to early stopping config + guard in `_build_callbacks` |
| 3 | Bug | Fix EarlyStopping monitor to read from `early_stopping` config section |
| 4 | Moderate | Narrow `except Exception` in `configure_optimizers` |
| 5 | Moderate | Fix dummy dataset default `total_samples` to `10000` |
| 6 | Moderate | Parameterize test dataset size with `self.total_samples` |
| 7 | Minor | Trim `lightning_module.py` to ≤200 LOC |
| 8 | Minor | Remove redundant `self.learning_rate =` assignments |
| 9 | Minor | Replace `Tensor | None` with `Optional[Tensor]` |

---

## Unresolved Questions

1. Is there a `configs/data/dummy.yaml` planned? Currently `num_samples` has no config home.
2. Should `csv_logs/` be replaced by Hydra's built-in output directory? The current setup bypasses Hydra's CWD isolation.
3. Is Python 3.8 compatibility a hard requirement, or has the baseline moved to 3.10+?

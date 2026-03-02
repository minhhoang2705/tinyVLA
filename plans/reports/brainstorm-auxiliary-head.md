# Brainstorm: Auxiliary Affordance Head for VLAModel

## Ground truth verdict
`observation.state` in LeRobot PushT already contains `[block_x, block_y, block_angle]` — it's currently silently discarded by the dataset adapter. No new data generation is needed.

## Recommended approach: Supervised Affordance Head
- Small MLP `(768 → 256 → 3)` attached to the mean-pooled fused latents.
- Predicts block position + angle via MSE loss.
- Gated by the existing `auxiliary_loss_weight` config (already in `VLAConfig`).
- Files to touch: `vla_configs.py`, `vla_base.py`, `lerobot_dataset.py`, `lightning_module.py`, `collate_batch_samples.py`

## Common pitfalls to watch
1. **Normalization:** Normalize `observation.state` coordinates (env pixel space ~0-512) to `[-1, 1]` before MSE computation.
2. **Metrics:** Log `train/action_loss` and `train/aux_loss` separately — don't let aux dominate the standard loss logs.
3. **Weighting:** Start `auxiliary_loss_weight=0.1`, not 1.0.
4. **Compilation:** Don't compile the affordance head alongside `action_head` in the same `torch.compile` call — compile them separately or skip compiling the affordance head.
5. **Backward Compatibility:** The `"state"` key in the batch should be **Optional** throughout — if there is no state key in the batch, compute no loss, and do not crash.
# n24_ext3000: budget extension of n24_knee97 (round 13)

Continuation of `n24_knee97` from its update_001000 checkpoint to 3000 updates.

- Source checkpoint: `n24_knee97/checkpoints/v6q_nocomm_gru_macpo_softc8_knee97/update_001000.msgpack` (COPIED here; original untouched).
- Verified before staging: the config rebuilt from the current launcher at updates=1000 has
  fingerprint `2b6c017047a159bb9018242b68a10b5f313af2ab12c32533bd9a1aa25e3c4c4f`, equal to the original checkpoint metadata. That proves the config used
  here differs from the original run in `total_updates` (1000 -> 3000) ONLY.
- The copied metadata's fingerprint was rewritten to `47808b89b7dd6ddb2c1222a9e0130e9fd9fc6dc3816ef582c06e2f0d310e744c` (the updates=3000 config) so the
  trainer's compatibility check passes against this staged copy.
- anneal_lr=True: on resume the LR schedule is linear(lr -> 0 at u3000); with ~1000 optimizer
  steps restored, training restarts near 2/3 lr. This is NOT identical to a from-scratch 3000-
  update run -- the first 1000 updates ran under the old schedule. Protocol is identical for
  both arms, so the comparison between them is unaffected.
- Evaluate with `--updates 3000` so the evaluator rebuilds the matching fingerprint.
- Pre-registered judgment conditions: PLAN.md round 13.

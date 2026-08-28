# b8_ext3000: budget extension of b8_mappopen001_knee97 (round 13)

Continuation of `b8_mappopen001_knee97` from its update_001000 checkpoint to 3000 updates.

- Source checkpoint: `b8_mappopen001_knee97/checkpoints/v6q_nocomm_gru_mappopen001_knee97/update_001000.msgpack` (COPIED here; original untouched).
- Verified before staging: the config rebuilt from the current launcher at updates=1000 has
  fingerprint `913a5270a414177181a41c7f703d3a5830e2554e5fc85615a754d037dcf68a11`, equal to the original checkpoint metadata. That proves the config used
  here differs from the original run in `total_updates` (1000 -> 3000) ONLY.
- The copied metadata's fingerprint was rewritten to `1ae595bba6b52b88759ab1a0c037d9b493269c61ad02812d9a739e8049c0386d` (the updates=3000 config) so the
  trainer's compatibility check passes against this staged copy.
- anneal_lr=True: on resume the LR schedule is linear(lr -> 0 at u3000); with ~1000 optimizer
  steps restored, training restarts near 2/3 lr. This is NOT identical to a from-scratch 3000-
  update run -- the first 1000 updates ran under the old schedule. Protocol is identical for
  both arms, so the comparison between them is unaffected.
- Evaluate with `--updates 3000` so the evaluator rebuilds the matching fingerprint.
- Pre-registered judgment conditions: PLAN.md round 13.

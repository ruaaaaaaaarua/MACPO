# Compare experiment actor weights (5000 episodes, seed=30)

Training run: `result/compare_mappo_matrpo_hypermarl/` (2026-06-22)

| File | Algorithm | Format |
|------|-----------|--------|
| `mappo_ia_actor_seed30.npz` | MAPPO independent actors + CTDE | Flax actor tensors (4 independent actor MLPs) |
| `hypermarl_mappo_actor_seed30.npz` | HyperMARL-MAPPO | Flax actor hypernet + embeddings + log_std |
| `matrpo_actor_agent{0-3}.pt` | MATRPO | PyTorch `state_dict` per agent |

Each `.npz` has a matching `.meta.json` with key names and source checkpoint path.

Budget: `TOTAL_TIMESTEPS=120000` = 5000 episodes × 24 steps.

Load JAX actor npz example:

```python
import numpy as np
data = np.load("mappo_ia_actor_seed30.npz")
print(list(data.files))
```

Load MATRPO actor example:

```python
import torch
from baselines.MATRPO.networks import Actor
ckpt = torch.load("matrpo_actor_agent0.pt", map_location="cpu")
actor = Actor(ckpt["obs_dim"], ckpt["action_dim"], hidden_dim=ckpt["hidden_dim"])
actor.load_state_dict(ckpt["state_dict"])
```

# Circle Packing

The first version is problematic. The script labled final is the best.

The best solution out there: https://github.com/algorithmicsuperintelligence/openevolve/issues/156



For the algorithmic version, run:

```bash
python3 ./circle_packing_jax.py \
  --n=26 \
  --seed=42 \
  --png_filename="__DELETE_ME__.png" \
  --initial_radius=0.01 \
  --max_margin_param=2.0 \
  --optimizer=adam \
  --lr=1e-4 \
  --newton_alpha=1.0 \
  --newton_damping=1.0 \
  --newton_max_delta=1.0 \
  --penalty_weight=100.0 \
  --n_steps=10000 \
  --log_every=500 \
  --learn_centers=true \
  --sgd_momentum=0.9 \
  --adamw_weight_decay=1e-4 \
  --candidates_multiplier=5
```


For the gradient-based (using PyTorch) version, run:

```bash
python3 ./circle_packing_pytorch.py \
  --n=26 \
  --seed=42 \
  --png_filename="result_pytorch.png" \
  --initial_radius=0.05 \
  --max_margin_param=1.0
```

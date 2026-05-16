# Circle Packing

The first version is problematic. The script labled final is the best.

The best solution out there: https://github.com/algorithmicsuperintelligence/openevolve/issues/156



For the algorithmic version, run:

```bash
python3 ./circle_packing_final.py \
	--n=26 \
	--seed=42 \
	--num_iterations=1000 \
	--png_filename="result.png"
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

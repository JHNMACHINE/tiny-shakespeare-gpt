# tiny-shakespeare-gpt

A small character-level GPT (4 layers, 192 wide, about 1.8M parameters)
trained on Tiny Shakespeare under [Ravex](https://github.com/JHNMACHINE/ravex),
so a run checkpoints, resumes and forks.

```bash
pip install ravex -r requirements.txt
RAVEX_STORAGE_PATH=runs/mine python train.py --steps 2000
```

`gpuzero.toml` declares the scripts an agent may start, with their default
parameters.

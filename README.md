# E2E-RL Launcher

## GUI Launcher

Start the desktop launcher with:

```bash
python gui_launcher.py
```

The launcher is a separate configuration tool. It does not embed pygame. When you click `Run`, it writes a JSON config file and launches one of:

- `run_model.py`
- `train.py`
- `eval.py`

as a subprocess, then streams the subprocess output back into the GUI.

## CLI With Config Files

All three entry points now support `--config`:

```bash
python run_model.py --config /path/to/run_config.json
python train.py --config /path/to/train_config.json
python eval.py --config /path/to/eval_config.json
```

CLI flags still work and override values loaded from the config file.

Examples:

```bash
python run_model.py --config configs/demo_run.json --episodes 3
python train.py --config configs/forward_state_train.json --timesteps 50000
python eval.py --config configs/reverse_eval.json --output_csv results/custom_eval/episodes.csv
```

## SSH X11 Forwarding

This setup works over SSH X forwarding as long as both the launcher and pygame are started on the remote machine and displayed through X11 locally.

Typical usage:

```bash
ssh -X user@host
python gui_launcher.py
```

or:

```bash
ssh -Y user@host
python gui_launcher.py
```

The launcher starts child processes with `os.environ.copy()`, so environment variables such as `DISPLAY` are inherited by the launched pygame process.

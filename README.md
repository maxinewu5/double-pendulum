# Inverted Double Pendulum on Cart

## Documentation

See [docs/report.md](https://github.com/maxinewu5/double-pendulum/blob/main/docs/report.md) for the project writeup and results discussion.

## Setup

From the `double-pendulum/` directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you want MuJoCo rendering, use `mjpython` from the MuJoCo Python install when running the rendered experiment scripts.

## Run Experiments

The scripts in `experiments/` are the main entrypoints for running controllers. The policy implementations themselves live in `policy/`, and each `run.py` script sets the controller parameters used for that experiment. If you want to tune gains, thresholds, or swing-up behavior, start by editing the corresponding `experiments/.../run.py` file.

Standalone LQR:

```bash
./.venv/bin/mjpython experiments/lqr/run.py
```

Swing-up with LQR handoff:

```bash
./.venv/bin/mjpython experiments/swingup_lqr/run.py
```

Baseline swing-up only:

```bash
./.venv/bin/mjpython experiments/swingup/run.py
```

Rendered experiment scripts usually save plots or debug figures under `outputs/`.

## Validation

Named validation scenarios:

```bash
python3 scenarios/run_scenarios.py
```

Angle basin at zero initial velocity:

```bash
python3 validate/angle_basin.py
```

Velocity basin at fixed angle anchors:

```bash
python3 validate/velocity_basin.py
```


## Project Layout

- `experiments/`: runnable experiment scripts
- `policy/`: controller implementations
- `validate/`: basin and recovery validation scripts
- `scenarios/`: named scenario suites
- `search/`: grid-search utilities
- `outputs/`: generated plots and CSVs

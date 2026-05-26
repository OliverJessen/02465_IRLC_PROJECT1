# Setup Guide

This guide is for anyone who wants to run the exercises, tests, or exam code in this repository with `uv`.

## Quick Start

From the repository root:

```bash
uv sync
uv run python -c "import irlc; print('irlc import ok')"
```

If both commands work, the environment is ready.

To run one week of tests:

```bash
uv run python -m unittest irlc.tests.tests_week01
```

To run all weekly exercise tests:

```bash
uv run python -m unittest irlc.tests.tests_week01 irlc.tests.tests_week02 irlc.tests.tests_week03 irlc.tests.tests_week04 irlc.tests.tests_week05 irlc.tests.tests_week06 irlc.tests.tests_week07 irlc.tests.tests_week08 irlc.tests.tests_week09 irlc.tests.tests_week10 irlc.tests.tests_week11 irlc.tests.tests_week12 irlc.tests.tests_week13
```

## First-Time Setup

1. Install `uv`.

   On macOS or Linux:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   Then close and reopen the terminal, or follow the shell setup message printed by the installer.

2. Clone or download this repository.

3. Open a terminal in the repository root, the folder containing `pyproject.toml`.

4. Create the environment:

   ```bash
   uv sync
   ```

5. Check that Python and the course package work:

   ```bash
   uv run python --version
   uv run python -c "import numpy, scipy, torch, gymnasium, irlc; print('imports ok')"
   ```

This repo is configured for Python 3.12 through `.python-version` and `pyproject.toml`. `uv` should install and use the correct Python automatically.

## Daily Use

Always run Python through `uv run` from the repository root:

```bash
uv run python path/to/script.py
```

For example:

```bash
uv run python irlc/ex00/old/ex00.py
uv run python -m unittest irlc.tests.tests_week03
```

Avoid running plain `python` unless you have manually activated the correct virtual environment. Plain `python` may use a different environment and fail to find `irlc` or the required packages.

## Setup Without uv

Using `uv` is recommended for this repository because it uses `pyproject.toml`, `uv.lock`, and `.python-version`. If you cannot or do not want to use `uv`, you can use a normal Python virtual environment instead.

You need Python 3.12.

Create and activate a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements_pip.txt
```

Check that the environment works:

```bash
python -c "import numpy, scipy, torch, gymnasium, irlc; print('imports ok')"
```

Run tests:

```bash
python -m unittest irlc.tests.tests_week01
```

When using this setup, remember to activate the environment every time you open a new terminal:

```bash
source .venv/bin/activate
```

This method can work well, but it is less reproducible than `uv` because package versions may drift depending on when and where `pip install` is run.

## Offline / Exam Preparation

Before the exam, while you still have internet:

```bash
uv sync
uv run python -c "import numpy, scipy, torch, gymnasium, irlc; print('imports ok')"
uv run python -m unittest irlc.tests.tests_week01
```

Then test that the cached environment works offline:

```bash
uv sync --offline
uv run --offline python -c "import numpy, scipy, torch, gymnasium, irlc; print('offline imports ok')"
```

If those commands pass before the exam, you should be able to use the repository without internet, as long as you keep the repository folder, `.venv`, `uv.lock`, and your `uv` cache intact.

Do not delete these before the exam:

- `.venv`
- `uv.lock`
- `~/.cache/uv`

The `.venv` folder contains the local Python environment. The `uv` cache contains downloaded packages used by `uv sync --offline`.

## Common Problems

### `uv: command not found`

Install `uv`, then restart the terminal:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### `ModuleNotFoundError: No module named 'irlc'`

Most likely you are either outside the repository root or using plain `python`.

Check your location:

```bash
pwd
ls pyproject.toml
```

Then run through `uv`:

```bash
uv run python -c "import irlc; print('ok')"
```

### `uv sync --offline` fails

First run a normal online sync:

```bash
uv sync
```

Then retry:

```bash
uv sync --offline
```

If offline sync still fails, the package cache may be missing or incomplete. Reconnect to the internet and run `uv sync` again before relying on the environment offline.

### `uv sync` changes or breaks the environment

Use the lockfile:

```bash
uv sync --locked
```

If you only want to check the existing environment without updating anything:

```bash
uv run --offline python -c "import irlc; print('environment works')"
```

### Pygame prints a `pkg_resources is deprecated` warning

This warning is harmless for the exercises. If tests continue after the warning, you can ignore it.

### Some exam tests fail with `NotImplementedError`

That usually means the exam question file is still a starter file. The weekly exercise tests can pass even if unsolved exam starter files fail.

## Useful Test Commands

Run one week:

```bash
uv run python -m unittest irlc.tests.tests_week05
```

Run all weeks:

```bash
uv run python -m unittest irlc.tests.tests_week01 irlc.tests.tests_week02 irlc.tests.tests_week03 irlc.tests.tests_week04 irlc.tests.tests_week05 irlc.tests.tests_week06 irlc.tests.tests_week07 irlc.tests.tests_week08 irlc.tests.tests_week09 irlc.tests.tests_week10 irlc.tests.tests_week11 irlc.tests.tests_week12 irlc.tests.tests_week13
```

Run midterm 2023A and 2023B tests:

```bash
uv run python -m unittest irlc.exam.midterm2023a.midterm2023a_tests irlc.exam.midterm2023b.midterm2023b_tests
```

Run a quick import check:

```bash
uv run python -c "import numpy, scipy, torch, gymnasium, irlc; print('ok')"
```

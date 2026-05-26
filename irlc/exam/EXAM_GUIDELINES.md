# Exam Practice Guidelines

This folder is now unpacked so you can solve the old exam and midterm tasks locally.
The starter files are intentionally unsolved: the places you need to work on are the
`question_*.py` files in each exam directory.

## What To Edit

Use this command to list the open tasks:

```bash
rg -n "TODO|NotImplemented|raise NotImplemented" irlc/exam/*/question_*.py
```

The current practice sets are:

```text
irlc/exam/midterm2023a/
irlc/exam/midterm2023b/
irlc/exam/exam2023spring/
irlc/exam/exam2024spring/
irlc/exam/exam2025spring/
```

Each directory contains:

- `question_*.py`: the files you should solve.
- `*_tests.py`: local tests for the questions.
- helper files such as `dp.py`, `mdp.py`, `inventory.py`, or `policy_evaluation.py`.
- `unitgrade_data/`: reference data used by the tests.
- `multiple_choice_answers.py`: fill this in if the exam asks for multiple choice answers.

The `solution/` directories are still present in the repository. Avoid opening them if
you want proper exam practice.

## Suggested Workflow

1. Pick one exam directory and read its PDF first.
2. Open only the corresponding `question_*.py` files.
3. Solve one function at a time.
4. Run that exam's tests after each small change.
5. If a test fails, inspect the relevant `*_tests.py` file to see the expected function
   signature, input shape, and numerical tolerance.
6. Once an exam passes, move to the next one.

## Test Commands

Run a single exam:

```bash
uv run python -m unittest irlc.exam.midterm2023a.midterm2023a_tests
uv run python -m unittest irlc.exam.midterm2023b.midterm2023b_tests
uv run python -m unittest irlc.exam.exam2023spring.exam2023spring_tests
uv run python -m unittest irlc.exam.exam2024spring.exam2024spring_tests
uv run python -m unittest irlc.exam.exam2025spring.exam2025spring_tests
```

Run all unpacked exams:

```bash
uv run python -m unittest \
  irlc.exam.midterm2023a.midterm2023a_tests \
  irlc.exam.midterm2023b.midterm2023b_tests \
  irlc.exam.exam2023spring.exam2023spring_tests \
  irlc.exam.exam2024spring.exam2024spring_tests \
  irlc.exam.exam2025spring.exam2025spring_tests
```

Starter files are expected to fail because of `NotImplementedError`. Treat that as a
checklist: each removed `NotImplementedError` should correspond to a piece you solved.

## Reusable Helper Code

Generic helper code lives in:

```text
irlc/exam/exam_helpers.py
```

You can import from it inside a `question_*.py` file, for example:

```python
from irlc.exam.exam_helpers import finite_horizon_dp, expected_stage_cost
```

The helper file is meant to save mechanical coding, not replace reading the problem.
The usual exam workflow is:

1. Read the problem text and identify the model.
2. Implement the exam-specific pieces such as `A`, `g`, `f`, `Pw`, `gN`, or `Psr`.
3. Use a helper for the standard computation.
4. Check the result against the local tests.

Useful helpers by topic:

- Finite-horizon DP / inventory:
  `expected_next_state`, `expected_stage_cost`, `finite_horizon_dp`,
  `evaluate_finite_horizon_policy`.
- Small MDPs:
  `expected_reward`, `mdp_action_value`, `greedy_mdp_action`,
  `best_action_lookahead`, `value_iteration_small`.
- Bandits, TD, Q-learning:
  `sample_average_action_values`, `constant_alpha_action_values`,
  `td_errors`, `td0_update`, `greedy_action_from_q`, `epsilon_greedy_action`,
  `q_learning_update`, `q_learning_trajectory`.
- Control:
  `euler_simulate`, `rk4_simulate`, `affine_residual`, `linear_policy_action`,
  `pid_last_action`.

Be careful with sign conventions. The DP helpers default to cost minimization, while
the MDP and RL helpers default to reward maximization.

## Offline Check

This project has been verified to run from the local environment with `uv` in offline
mode. Before an exam, you can check the same thing with:

```bash
uv run --offline python -c "import irlc, gymnasium, numpy, scipy, matplotlib, pygame, sympy, seaborn, torch, unitgrade; print('offline imports OK')"
```

You can also run the solved weekly exercise tests offline:

```bash
uv run --offline python -m unittest \
  irlc.tests.tests_week01 \
  irlc.tests.tests_week02 \
  irlc.tests.tests_week03 \
  irlc.tests.tests_week04 \
  irlc.tests.tests_week05 \
  irlc.tests.tests_week06 \
  irlc.tests.tests_week07 \
  irlc.tests.tests_week08 \
  irlc.tests.tests_week09 \
  irlc.tests.tests_week10 \
  irlc.tests.tests_week11 \
  irlc.tests.tests_week12 \
  irlc.tests.tests_week13
```

For real exam use, keep the project folder, `.venv/`, `uv.lock`, and your `uv` cache
intact. Avoid running dependency-changing commands such as `uv add`, `uv remove`, or
`uv lock --upgrade` during the exam unless you know you have internet access.

If you want to sync the environment before or during an offline exam, prefer:

```bash
uv sync --offline --frozen
```

This uses the existing `uv.lock` and local cache without attempting dependency updates.
At the time this guide was written, plain `uv sync --dry-run` reported that the
environment was already up to date.

## Practical Tips

- Keep changes small and rerun tests often.
- Prefer using the helper classes already provided in the same exam directory.
- Compare with the weekly exercises when a topic looks familiar, especially dynamic
  programming, MDPs, LQR/control, bandits, and Q-learning.
- Do not change the tests or `unitgrade_data/` while practicing.
- If a function returns a number, check whether the test uses `assertAlmostEqual`;
  small floating-point differences are normal, but large differences usually mean the
  recurrence, dynamics, or indexing is off.
- If a question file has imports at the top, use those local helper modules before
  writing a new implementation from scratch.

## Notes

`exam2026spring` is not unpacked because this repository currently does not include a
corresponding `_problems_nosol.zip` archive for it.

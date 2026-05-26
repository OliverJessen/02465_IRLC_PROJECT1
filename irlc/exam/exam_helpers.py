"""
Small reusable helpers for IRLC exam coding questions.

The functions here are intentionally generic. They do not know any specific old exam
constants, but they cover the mechanics that tend to repeat: finite-horizon DP,
small MDP lookahead/value iteration, bandit/TD/Q-learning updates, and simple control
simulation utilities.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np


def _as_list(items: Iterable[Any]) -> list[Any]:
    return list(items)


def _pick_key(scores: Mapping[Any, float], mode: str = "min", tie_break: str = "first") -> Any:
    """Pick a key from a score dictionary with deterministic tie handling."""
    if not scores:
        raise ValueError("Cannot pick from an empty score dictionary")

    target = min(scores.values()) if mode == "min" else max(scores.values())
    tied = [key for key, value in scores.items() if abs(value - target) <= 1e-12]

    if tie_break == "first":
        return tied[0]
    if tie_break == "min":
        return min(tied)
    if tie_break == "max":
        return max(tied)
    raise ValueError("tie_break must be one of: 'first', 'min', 'max'")


# ---------------------------------------------------------------------------
# Finite-horizon DP helpers


def expected_next_state(model: Any, x: Any, u: Any, k: int = 0) -> float:
    """Compute E[f(x, u, W, k)] for a DP model with Pw/f methods."""
    return sum(p * model.f(x, u, w, k) for w, p in model.Pw(x, u, k).items())


def expected_stage_cost(model: Any, x: Any, u: Any, k: int = 0) -> float:
    """Compute E[g(x, u, W, k)] for a DP model with Pw/g methods."""
    return sum(p * model.g(x, u, w, k) for w, p in model.Pw(x, u, k).items())


def dp_q_value(model: Any, J_next: Mapping[Any, float], x: Any, u: Any, k: int, gamma: float = 1.0) -> float:
    """Compute the one-step DP Q-value for a finite-horizon stochastic DP model."""
    return sum(
        p * (model.g(x, u, w, k) + gamma * J_next[model.f(x, u, w, k)])
        for w, p in model.Pw(x, u, k).items()
    )


def finite_horizon_dp(
    model: Any,
    *,
    gamma: float = 1.0,
    mode: str = "min",
    tie_break: str = "first",
) -> tuple[list[dict[Any, float]], list[dict[Any, Any]]]:
    """
    Generic finite-horizon DP for models with N, S(k), A(x,k), Pw, f, g, and gN.

    Returns (J, pi), where J[k][x] is the value and pi[k][x] is the selected action.
    Use mode="min" for cost minimization and mode="max" for reward maximization.
    """
    N = model.N
    J: list[dict[Any, float]] = [{} for _ in range(N + 1)]
    pi: list[dict[Any, Any]] = [{} for _ in range(N)]
    J[N] = {x: model.gN(x) for x in model.S(N)}

    for k in range(N - 1, -1, -1):
        for x in model.S(k):
            scores = {u: dp_q_value(model, J[k + 1], x, u, k, gamma=gamma) for u in model.A(x, k)}
            u_star = _pick_key(scores, mode=mode, tie_break=tie_break)
            J[k][x] = scores[u_star]
            pi[k][x] = u_star
    return J, pi


def evaluate_finite_horizon_policy(
    model: Any,
    policy: Sequence[Mapping[Any, Any]] | Callable[[int, Any], Any],
    *,
    gamma: float = 1.0,
) -> list[dict[Any, float]]:
    """Evaluate a deterministic finite-horizon policy for a DP model."""
    N = model.N
    J: list[dict[Any, float]] = [{} for _ in range(N + 1)]
    J[N] = {x: model.gN(x) for x in model.S(N)}

    for k in range(N - 1, -1, -1):
        for x in model.S(k):
            u = policy(k, x) if callable(policy) else policy[k][x]
            J[k][x] = dp_q_value(model, J[k + 1], x, u, k, gamma=gamma)
    return J


# ---------------------------------------------------------------------------
# Small MDP helpers


def expected_reward(mdp: Any, s: Any, a: Any) -> float:
    """Compute E[R | s, a] for an MDP whose Psr returns {(sp, r): probability}."""
    return sum(p * r for (_, r), p in mdp.Psr(s, a).items())


def mdp_action_value(
    mdp: Any,
    s: Any,
    a: Any,
    V: Mapping[Any, float] | None = None,
    *,
    gamma: float = 1.0,
    default_value: float = 0.0,
) -> float:
    """Compute sum p(sp,r|s,a) * (r + gamma * V[sp])."""
    V = {} if V is None else V
    return sum(p * (r + gamma * V.get(sp, default_value)) for (sp, r), p in mdp.Psr(s, a).items())


def greedy_mdp_action(
    mdp: Any,
    s: Any,
    V: Mapping[Any, float] | None = None,
    *,
    gamma: float = 1.0,
    tie_break: str = "first",
) -> Any:
    """Pick the action with highest one-step action value."""
    scores = {a: mdp_action_value(mdp, s, a, V, gamma=gamma) for a in mdp.A(s)}
    return _pick_key(scores, mode="max", tie_break=tie_break)


def lookahead_action_value(
    mdp: Any,
    s: Any,
    a: Any,
    *,
    depth: int,
    gamma: float = 1.0,
    terminal_value: Mapping[Any, float] | None = None,
    tie_break: str = "first",
) -> float:
    """
    Value of taking action a now and then acting greedily for depth-1 more steps.

    depth=1 means only the immediate expected reward is used.
    """
    if depth <= 0:
        terminal_value = {} if terminal_value is None else terminal_value
        return terminal_value.get(s, 0.0)

    def V(sp: Any, remaining: int) -> float:
        if remaining <= 0 or (hasattr(mdp, "is_terminal") and mdp.is_terminal(sp)):
            terminal_value_ = {} if terminal_value is None else terminal_value
            return terminal_value_.get(sp, 0.0)
        scores = {
            ap: lookahead_action_value(
                mdp,
                sp,
                ap,
                depth=remaining,
                gamma=gamma,
                terminal_value=terminal_value,
                tie_break=tie_break,
            )
            for ap in mdp.A(sp)
        }
        return scores[_pick_key(scores, mode="max", tie_break=tie_break)]

    return sum(p * (r + gamma * V(sp, depth - 1)) for (sp, r), p in mdp.Psr(s, a).items())


def best_action_lookahead(
    mdp: Any,
    s: Any,
    *,
    depth: int,
    gamma: float = 1.0,
    terminal_value: Mapping[Any, float] | None = None,
    tie_break: str = "first",
) -> Any:
    """Pick the best action using finite-depth lookahead."""
    scores = {
        a: lookahead_action_value(
            mdp,
            s,
            a,
            depth=depth,
            gamma=gamma,
            terminal_value=terminal_value,
            tie_break=tie_break,
        )
        for a in mdp.A(s)
    }
    return _pick_key(scores, mode="max", tie_break=tie_break)


def value_iteration_small(
    mdp: Any,
    *,
    gamma: float,
    states: Iterable[Any] | None = None,
    tol: float = 1e-10,
    max_iters: int = 10000,
    tie_break: str = "first",
) -> tuple[dict[Any, Any], dict[Any, float]]:
    """Simple value iteration for small finite MDPs."""
    if states is None:
        if hasattr(mdp, "nonterminal_states"):
            states = mdp.nonterminal_states
        elif hasattr(mdp, "states"):
            states = mdp.states
        else:
            raise ValueError("Pass states=... for MDPs without a states/nonterminal_states attribute")

    states_list = _as_list(states)
    V = {s: 0.0 for s in states_list}
    pi = {s: _as_list(mdp.A(s))[0] for s in states_list}

    for _ in range(max_iters):
        delta = 0.0
        V_next = {}
        for s in states_list:
            scores = {a: mdp_action_value(mdp, s, a, V, gamma=gamma) for a in mdp.A(s)}
            a_star = _pick_key(scores, mode="max", tie_break=tie_break)
            V_next[s] = scores[a_star]
            pi[s] = a_star
            delta = max(delta, abs(V_next[s] - V[s]))
        V = V_next
        if delta < tol:
            break
    return pi, V


# ---------------------------------------------------------------------------
# Bandit, TD, and Q-learning helpers


def greedy_action_from_q(
    q_values: Mapping[Any, float],
    state: Any,
    actions: Iterable[Any],
    *,
    default: float = 0.0,
    tie_break: str = "first",
) -> Any:
    """Pick argmax_a Q(state, a). q_values should usually use (state, action) keys."""
    scores = {a: q_values.get((state, a), default) for a in actions}
    return _pick_key(scores, mode="max", tie_break=tie_break)


def epsilon_greedy_action(
    q_values: Mapping[Any, float],
    state: Any,
    actions: Iterable[Any],
    epsilon: float,
    *,
    rng: np.random.Generator | None = None,
    default: float = 0.0,
    tie_break: str = "first",
) -> Any:
    """Epsilon-greedy action selection."""
    rng = np.random.default_rng() if rng is None else rng
    actions_list = _as_list(actions)
    if rng.random() < epsilon:
        return actions_list[int(rng.integers(len(actions_list)))]
    return greedy_action_from_q(q_values, state, actions_list, default=default, tie_break=tie_break)


def sample_average_action_values(
    n_actions: int,
    actions: Sequence[int],
    rewards: Sequence[float],
    *,
    initial: float = 0.0,
) -> dict[int, float]:
    """Compute sample-average action values from action/reward histories."""
    counts = {a: 0 for a in range(n_actions)}
    sums = {a: 0.0 for a in range(n_actions)}
    for a, r in zip(actions, rewards):
        counts[a] += 1
        sums[a] += r
    return {a: sums[a] / counts[a] if counts[a] else initial for a in range(n_actions)}


def constant_alpha_action_values(
    n_actions: int,
    actions: Sequence[int],
    rewards: Sequence[float],
    alpha: float,
    *,
    initial: float = 0.0,
) -> dict[int, float]:
    """Compute nonstationary bandit estimates Q <- Q + alpha * (R - Q)."""
    Q = {a: float(initial) for a in range(n_actions)}
    for a, r in zip(actions, rewards):
        Q[a] += alpha * (r - Q[a])
    return Q


def td_errors(values: Mapping[Any, float], states: Sequence[Any], rewards: Sequence[float], gamma: float) -> list[float]:
    """Compute TD(0) errors delta_t = r_{t+1} + gamma V(s_{t+1}) - V(s_t)."""
    return [
        rewards[t] + gamma * values.get(states[t + 1], 0.0) - values.get(states[t], 0.0)
        for t in range(len(rewards))
    ]


def td0_update(
    values: Mapping[Any, float],
    states: Sequence[Any],
    rewards: Sequence[float],
    gamma: float,
    alpha: float,
    *,
    batched: bool = False,
) -> dict[Any, float]:
    """Perform TD(0) updates. batched=True computes all deltas before changing V."""
    V = dict(values)
    deltas = td_errors(V, states, rewards, gamma) if batched else None
    for t, r in enumerate(rewards):
        s = states[t]
        if batched:
            delta = deltas[t]
        else:
            sp = states[t + 1]
            delta = r + gamma * V.get(sp, 0.0) - V.get(s, 0.0)
        V[s] = V.get(s, 0.0) + alpha * delta
    return V


def q_learning_update(
    q_values: Mapping[Any, float],
    state: Any,
    action: Any,
    reward: float,
    next_state: Any,
    next_actions: Iterable[Any],
    alpha: float,
    gamma: float,
    *,
    default: float = 0.0,
) -> float:
    """Return the updated value for one Q-learning transition."""
    q_old = q_values.get((state, action), default)
    best_next = max(q_values.get((next_state, a), default) for a in next_actions)
    return q_old + alpha * (reward + gamma * best_next - q_old)


def q_learning_trajectory(
    transitions: Sequence[tuple[Any, Any, float]],
    actions: Iterable[Any],
    alpha: float,
    gamma: float,
    *,
    default: float = 0.0,
) -> dict[tuple[Any, Any], float]:
    """
    Apply Q-learning to a trajectory of (state, action, reward) triples.

    The next state for transition t is taken from transitions[t + 1][0], so the last
    triple only supplies the final next state.
    """
    q_values: dict[tuple[Any, Any], float] = {}
    actions_list = _as_list(actions)
    for t, (state, action, reward) in enumerate(transitions[:-1]):
        next_state = transitions[t + 1][0]
        q_values[state, action] = q_learning_update(
            q_values,
            state,
            action,
            reward,
            next_state,
            actions_list,
            alpha,
            gamma,
            default=default,
        )
    return q_values


# ---------------------------------------------------------------------------
# Control helpers


def euler_step(f: Callable[[np.ndarray, Any], np.ndarray], x: Sequence[float], u: Any, dt: float) -> np.ndarray:
    """One explicit Euler step x <- x + dt * f(x, u)."""
    x_arr = np.asarray(x, dtype=float)
    return x_arr + dt * np.asarray(f(x_arr, u), dtype=float)


def euler_simulate(
    f: Callable[[np.ndarray, Any], np.ndarray],
    x0: Sequence[float],
    u: Any,
    dt: float,
    N: int,
) -> np.ndarray:
    """Simulate N explicit Euler steps and return the final state."""
    x = np.asarray(x0, dtype=float)
    for _ in range(N):
        x = euler_step(f, x, u, dt)
    return x


def rk4_step(f: Callable[[np.ndarray, Any], np.ndarray], x: Sequence[float], u: Any, dt: float) -> np.ndarray:
    """One classical fourth-order Runge-Kutta step for xdot=f(x,u)."""
    x_arr = np.asarray(x, dtype=float)
    k1 = np.asarray(f(x_arr, u), dtype=float)
    k2 = np.asarray(f(x_arr + dt * k1 / 2, u), dtype=float)
    k3 = np.asarray(f(x_arr + dt * k2 / 2, u), dtype=float)
    k4 = np.asarray(f(x_arr + dt * k3, u), dtype=float)
    return x_arr + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def rk4_simulate(
    f: Callable[[np.ndarray, Any], np.ndarray],
    x0: Sequence[float],
    u: Any,
    dt: float,
    N: int,
) -> np.ndarray:
    """Simulate N RK4 steps and return the final state."""
    x = np.asarray(x0, dtype=float)
    for _ in range(N):
        x = rk4_step(f, x, u, dt)
    return x


def affine_residual(A: np.ndarray, B: np.ndarray, x: Sequence[float], u: Sequence[float], x_next: Sequence[float]) -> np.ndarray:
    """Return d in the affine approximation x_next = A x + B u + d."""
    return np.asarray(x_next, dtype=float) - np.asarray(A) @ np.asarray(x, dtype=float) - np.asarray(B) @ np.asarray(u, dtype=float)


def linear_policy_action(L: np.ndarray, l: np.ndarray | float, x: Sequence[float]) -> np.ndarray:
    """Evaluate u = L x + l."""
    return np.asarray(L) @ np.asarray(x, dtype=float) + np.asarray(l)


def pid_last_action(
    xs: Sequence[float],
    xstar: float,
    *,
    Kp: float = 0.0,
    Ki: float = 0.0,
    Kd: float = 0.0,
    dt: float = 1.0,
    smooth_last_two_derivatives: bool = False,
) -> float:
    """Return the final PID action for a sequence of observations."""
    integral = 0.0
    previous_error = 0.0
    errors: list[float] = []
    u = 0.0

    for x in xs:
        error = xstar - x
        errors.append(error)
        integral += dt * error

        if smooth_last_two_derivatives and len(errors) >= 3:
            d1 = (errors[-1] - errors[-2]) / dt
            d2 = (errors[-2] - errors[-3]) / dt
            derivative = (d1 + d2) / 2
        else:
            derivative = (error - previous_error) / dt

        u = Kp * error + Ki * integral + Kd * derivative
        previous_error = error

    return float(u)

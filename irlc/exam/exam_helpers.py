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


def policy_evaluation_small(
    mdp: Any,
    policy: Mapping[Any, Any] | Callable[[Any], Any],
    *,
    gamma: float,
    states: Iterable[Any] | None = None,
    tol: float = 1e-10,
    max_iters: int = 10000,
) -> dict[Any, float]:
    """Iterative policy evaluation for a small finite MDP and deterministic policy."""
    if states is None:
        if hasattr(mdp, "nonterminal_states"):
            states = mdp.nonterminal_states
        elif hasattr(mdp, "states"):
            states = mdp.states
        else:
            raise ValueError("Pass states=... for MDPs without a states/nonterminal_states attribute")

    states_list = _as_list(states)
    V = {s: 0.0 for s in states_list}
    for _ in range(max_iters):
        delta = 0.0
        for s in states_list:
            a = policy(s) if callable(policy) else policy[s]
            v_old = V[s]
            V[s] = mdp_action_value(mdp, s, a, V, gamma=gamma)
            delta = max(delta, abs(v_old - V[s]))
        if delta < tol:
            break
    return V


def policy_iteration_small(
    mdp: Any,
    *,
    gamma: float,
    states: Iterable[Any] | None = None,
    tol: float = 1e-10,
    max_iters: int = 1000,
    tie_break: str = "first",
) -> tuple[dict[Any, Any], dict[Any, float]]:
    """Policy iteration for a small finite MDP with deterministic policies."""
    if states is None:
        if hasattr(mdp, "nonterminal_states"):
            states = mdp.nonterminal_states
        elif hasattr(mdp, "states"):
            states = mdp.states
        else:
            raise ValueError("Pass states=... for MDPs without a states/nonterminal_states attribute")

    states_list = _as_list(states)
    pi = {s: _as_list(mdp.A(s))[0] for s in states_list}
    V = {s: 0.0 for s in states_list}
    for _ in range(max_iters):
        V = policy_evaluation_small(mdp, pi, gamma=gamma, states=states_list, tol=tol)
        stable = True
        for s in states_list:
            old = pi[s]
            pi[s] = greedy_mdp_action(mdp, s, V, gamma=gamma, tie_break=tie_break)
            stable = stable and pi[s] == old
        if stable:
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


def ucb_action(
    q_values: Mapping[Any, float],
    counts: Mapping[Any, int],
    t: int,
    c: float,
    actions: Iterable[Any] | None = None,
    *,
    tie_break: str = "first",
) -> Any:
    """Pick an action using the UCB action-selection rule."""
    actions_list = _as_list(actions if actions is not None else q_values.keys())
    for a in actions_list:
        if counts.get(a, 0) == 0:
            return a
    scores = {
        a: q_values.get(a, 0.0) + c * np.sqrt(np.log(t) / counts[a])
        for a in actions_list
    }
    return _pick_key(scores, mode="max", tie_break=tie_break)


def softmax_preferences(preferences: Mapping[Any, float], actions: Iterable[Any] | None = None) -> dict[Any, float]:
    """Convert action preferences into softmax probabilities."""
    actions_list = _as_list(actions if actions is not None else preferences.keys())
    values = np.asarray([preferences.get(a, 0.0) for a in actions_list], dtype=float)
    values = values - np.max(values)
    probs = np.exp(values)
    probs = probs / probs.sum()
    return {a: float(p) for a, p in zip(actions_list, probs)}


def gradient_bandit_update(
    preferences: Mapping[Any, float],
    action: Any,
    reward: float,
    alpha: float,
    *,
    baseline: float = 0.0,
    actions: Iterable[Any] | None = None,
) -> dict[Any, float]:
    """One gradient-bandit preference update."""
    H = dict(preferences)
    actions_list = _as_list(actions if actions is not None else H.keys())
    probs = softmax_preferences(H, actions_list)
    for a in actions_list:
        indicator = 1.0 if a == action else 0.0
        H[a] = H.get(a, 0.0) + alpha * (reward - baseline) * (indicator - probs[a])
    return H


def discounted_returns(rewards: Sequence[float], gamma: float) -> list[float]:
    """Return G_t for every time t in a reward sequence."""
    returns = [0.0 for _ in rewards]
    g = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        g = rewards[t] + gamma * g
        returns[t] = g
    return returns


def mc_state_returns(
    states: Sequence[Any],
    rewards: Sequence[float],
    gamma: float,
    *,
    first_visit: bool = True,
) -> dict[Any, list[float]]:
    """
    Map each state to Monte Carlo returns observed after visits to that state.

    states should have length len(rewards)+1, where rewards[t] is R_{t+1}.
    """
    returns = discounted_returns(rewards, gamma)
    out: dict[Any, list[float]] = {}
    seen = set()
    for t, s in enumerate(states[:-1]):
        if first_visit and s in seen:
            continue
        seen.add(s)
        out.setdefault(s, []).append(returns[t])
    return out


def mc_value_estimates(
    states: Sequence[Any],
    rewards: Sequence[float],
    gamma: float,
    *,
    first_visit: bool = True,
) -> dict[Any, float]:
    """Monte Carlo value estimates from one episode by averaging visit returns."""
    observed = mc_state_returns(states, rewards, gamma, first_visit=first_visit)
    return {s: float(np.mean(gs)) for s, gs in observed.items()}


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


def sarsa_update(
    q_values: Mapping[Any, float],
    state: Any,
    action: Any,
    reward: float,
    next_state: Any,
    next_action: Any,
    alpha: float,
    gamma: float,
    *,
    done: bool = False,
    default: float = 0.0,
) -> float:
    """Return the updated value for one SARSA transition."""
    q_old = q_values.get((state, action), default)
    q_next = 0.0 if done else q_values.get((next_state, next_action), default)
    return q_old + alpha * (reward + gamma * q_next - q_old)


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


def double_q_learning_update(
    q1: Mapping[Any, float],
    q2: Mapping[Any, float],
    state: Any,
    action: Any,
    reward: float,
    next_state: Any,
    next_actions: Iterable[Any],
    alpha: float,
    gamma: float,
    *,
    update: int = 1,
    done: bool = False,
    default: float = 0.0,
    tie_break: str = "first",
) -> tuple[dict[Any, float], dict[Any, float]]:
    """
    One tabular Double Q-learning update.

    update=1 updates q1 using q1 for selection and q2 for evaluation. update=2 swaps roles.
    """
    Q1, Q2 = dict(q1), dict(q2)
    if update not in (1, 2):
        raise ValueError("update must be 1 or 2")

    target_q, eval_q = (Q1, Q2) if update == 1 else (Q2, Q1)
    old = target_q.get((state, action), default)
    if done:
        target = reward
    else:
        scores = {a: target_q.get((next_state, a), default) for a in next_actions}
        best_next = _pick_key(scores, mode="max", tie_break=tie_break)
        target = reward + gamma * eval_q.get((next_state, best_next), default)
    target_q[state, action] = old + alpha * (target - old)
    return (target_q, eval_q) if update == 1 else (eval_q, target_q)


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


def n_step_return(
    rewards: Sequence[float],
    gamma: float,
    n: int,
    *,
    bootstrap_value: float = 0.0,
) -> float:
    """Compute G_{t:t+n} from rewards[t:t+n] plus an optional bootstrapped value."""
    total = 0.0
    for i, r in enumerate(rewards[:n]):
        total += (gamma**i) * r
    if len(rewards) >= n:
        total += (gamma**n) * bootstrap_value
    return total


def td_lambda_update(
    values: Mapping[Any, float],
    states: Sequence[Any],
    rewards: Sequence[float],
    gamma: float,
    alpha: float,
    lamb: float,
    *,
    replacing: bool = False,
) -> dict[Any, float]:
    """Tabular TD(lambda) value update for one episode."""
    V = dict(values)
    E: dict[Any, float] = {}
    for t, r in enumerate(rewards):
        s, sp = states[t], states[t + 1]
        delta = r + gamma * V.get(sp, 0.0) - V.get(s, 0.0)
        for key in list(E):
            E[key] *= gamma * lamb
        E[s] = 1.0 if replacing else E.get(s, 0.0) + 1.0
        for key, trace in E.items():
            V[key] = V.get(key, 0.0) + alpha * delta * trace
    return V


def sarsa_lambda_update(
    q_values: Mapping[Any, float],
    states: Sequence[Any],
    actions: Sequence[Any],
    rewards: Sequence[float],
    gamma: float,
    alpha: float,
    lamb: float,
    *,
    replacing: bool = False,
    default: float = 0.0,
) -> dict[Any, float]:
    """Tabular SARSA(lambda) update for one episode."""
    Q = dict(q_values)
    E: dict[tuple[Any, Any], float] = {}
    for t, r in enumerate(rewards):
        s, a = states[t], actions[t]
        done = t + 1 >= len(actions)
        q_next = 0.0 if done else Q.get((states[t + 1], actions[t + 1]), default)
        delta = r + gamma * q_next - Q.get((s, a), default)
        for key in list(E):
            E[key] *= gamma * lamb
        E[s, a] = 1.0 if replacing else E.get((s, a), 0.0) + 1.0
        for key, trace in E.items():
            Q[key] = Q.get(key, default) + alpha * delta * trace
    return Q


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


def finite_difference_jacobian(
    f: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x: Sequence[float],
    u: Sequence[float],
    *,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Numerically approximate A=df/dx and B=df/du around (x,u)."""
    x_arr = np.asarray(x, dtype=float)
    u_arr = np.asarray(u, dtype=float)
    y0 = np.asarray(f(x_arr, u_arr), dtype=float)
    A = np.zeros((y0.size, x_arr.size))
    B = np.zeros((y0.size, u_arr.size))

    for i in range(x_arr.size):
        dx = np.zeros_like(x_arr)
        dx[i] = eps
        A[:, i] = (np.asarray(f(x_arr + dx, u_arr), dtype=float) - np.asarray(f(x_arr - dx, u_arr), dtype=float)) / (2 * eps)
    for i in range(u_arr.size):
        du = np.zeros_like(u_arr)
        du[i] = eps
        B[:, i] = (np.asarray(f(x_arr, u_arr + du), dtype=float) - np.asarray(f(x_arr, u_arr - du), dtype=float)) / (2 * eps)
    return A, B


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


# ---------------------------------------------------------------------------
# Linear function approximation helpers


def linear_value(features: Sequence[float], weights: Sequence[float]) -> float:
    """Compute x^T w."""
    return float(np.asarray(features, dtype=float) @ np.asarray(weights, dtype=float))


def linear_q_values(
    feature_fn: Callable[[Any, Any], Sequence[float]],
    weights: Sequence[float],
    state: Any,
    actions: Iterable[Any],
) -> dict[Any, float]:
    """Compute approximate q(s,a;w) for all actions."""
    return {a: linear_value(feature_fn(state, a), weights) for a in actions}


def semi_gradient_q_update(
    weights: Sequence[float],
    features: Sequence[float],
    target: float,
    prediction: float,
    alpha: float,
) -> np.ndarray:
    """Semi-gradient update w <- w + alpha * (target - prediction) * features."""
    w = np.asarray(weights, dtype=float)
    x = np.asarray(features, dtype=float)
    return w + alpha * (target - prediction) * x


def semi_gradient_sarsa_update(
    weights: Sequence[float],
    feature_fn: Callable[[Any, Any], Sequence[float]],
    state: Any,
    action: Any,
    reward: float,
    next_state: Any,
    next_action: Any,
    alpha: float,
    gamma: float,
    *,
    done: bool = False,
) -> np.ndarray:
    """One linear semi-gradient SARSA update."""
    w = np.asarray(weights, dtype=float)
    x = np.asarray(feature_fn(state, action), dtype=float)
    prediction = linear_value(x, w)
    target = reward if done else reward + gamma * linear_value(feature_fn(next_state, next_action), w)
    return semi_gradient_q_update(w, x, target, prediction, alpha)


def semi_gradient_sarsa_lambda_update(
    weights: Sequence[float],
    trace: Sequence[float],
    features: Sequence[float],
    target: float,
    prediction: float,
    alpha: float,
    gamma: float,
    lamb: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One linear semi-gradient SARSA(lambda) weight/trace update."""
    w = np.asarray(weights, dtype=float)
    z = gamma * lamb * np.asarray(trace, dtype=float) + np.asarray(features, dtype=float)
    return w + alpha * (target - prediction) * z, z

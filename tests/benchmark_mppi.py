"""Performance benchmarks for MPPI, SMPPI, and KMPPI.

Run with: python tests/benchmark_mppi.py
Produces timing and solution quality results for various configurations
to track performance before and after optimization refactors.

Solution quality metrics:
  - accumulated_cost: Total running cost over a multi-step trajectory
  - final_dist: Euclidean distance from goal at end of trajectory
  - control_smoothness: Sum of |u_{t+1} - u_t| (lower = smoother)
"""
import time
import json
import sys
import torch
from pytorch_mppi import MPPI, SMPPI, KMPPI
from pytorch_mppi.mppi import RBFKernel

DEVICE = "cpu"
DTYPE = torch.double
SEED = 42

# Also benchmark on CUDA if available
DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")

# ---------------------------------------------------------------------------
# Test environments (no external deps)
# ---------------------------------------------------------------------------
B_MATRIX = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=DTYPE)
GOAL = torch.tensor([2.0, 2.0], dtype=DTYPE)


def _make_dynamics(device):
    b = B_MATRIX.to(device=device)

    def dynamics(state, action):
        return state + action @ b.T

    return dynamics


def _make_cost(device):
    goal = GOAL.to(device=device)

    def cost(state, action):
        dx = goal - state
        return (dx ** 2).sum(dim=-1)

    return cost


def _make_terminal_cost(device):
    goal = GOAL.to(device=device)

    def terminal_cost(states, actions):
        dx = goal - states[..., -1, :]
        return (dx ** 2).sum(dim=-1)

    return terminal_cost


# Higher-dimensional environment
def _make_dynamics_nd(device, nx, nu):
    def dynamics(state, action):
        delta = torch.zeros_like(state)
        delta[..., :nu] = action
        return state + delta

    return dynamics


def _make_cost_nd(device, nx):
    def cost(state, action):
        return (state ** 2).sum(dim=-1)

    return cost


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------
def benchmark_command(ctrl, state, num_warmup=3, num_iters=20):
    """Benchmark the command() method, returning stats in seconds."""
    # Warmup
    for _ in range(num_warmup):
        ctrl.command(state, shift_nominal_trajectory=False)

    if state.device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(num_iters):
        ctrl.reset()
        s = state.clone()
        if state.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        ctrl.command(s)
        if state.device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times = sorted(times)
    # Drop fastest and slowest 10%
    trim = max(1, len(times) // 10)
    trimmed = times[trim:-trim] if len(times) > 2 * trim else times
    mean_t = sum(trimmed) / len(trimmed)
    min_t = times[0]
    max_t = times[-1]
    return {"mean_s": mean_t, "min_s": min_t, "max_s": max_t, "num_iters": num_iters}


def benchmark_multi_step(ctrl, state, dynamics_fn, num_steps=20, num_warmup=2, num_iters=5):
    """Benchmark a full control loop of num_steps."""
    for _ in range(num_warmup):
        ctrl.reset()
        s = state.clone()
        for _ in range(num_steps):
            a = ctrl.command(s)
            s = dynamics_fn(s.unsqueeze(0), a.unsqueeze(0)).squeeze(0)

    if state.device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(num_iters):
        ctrl.reset()
        s = state.clone()
        if state.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(num_steps):
            a = ctrl.command(s)
            s = dynamics_fn(s.unsqueeze(0), a.unsqueeze(0)).squeeze(0)
        if state.device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    mean_t = sum(times) / len(times)
    return {"mean_s": mean_t, "min_s": min(times), "max_s": max(times),
            "per_step_s": mean_t / num_steps, "num_steps": num_steps}


def evaluate_quality(ctrl, state, dynamics_fn, cost_fn, goal, num_steps=20, num_trials=5):
    """Evaluate solution quality over multiple seeded trials.

    Returns dict with:
      accumulated_cost: mean total running cost over trajectory
      final_dist: mean Euclidean distance to goal at end
      control_smoothness: mean sum of |u_{t+1} - u_t|
      per_trial: list of per-trial results for variance analysis
    """
    per_trial = []
    for trial in range(num_trials):
        torch.manual_seed(SEED + trial)
        ctrl.reset()
        s = state.clone()
        total_cost = 0.0
        actions = []
        for _ in range(num_steps):
            a = ctrl.command(s)
            actions.append(a.clone())
            c = cost_fn(s.unsqueeze(0), a.unsqueeze(0)).item()
            total_cost += c
            s = dynamics_fn(s.unsqueeze(0), a.unsqueeze(0)).squeeze(0)
        final_dist = (s - goal).norm().item()
        actions_t = torch.stack(actions)
        control_diff = actions_t.diff(dim=0).abs().sum().item()
        per_trial.append({
            "accumulated_cost": total_cost,
            "final_dist": final_dist,
            "control_smoothness": control_diff,
        })

    acc_costs = [t["accumulated_cost"] for t in per_trial]
    dists = [t["final_dist"] for t in per_trial]
    smooths = [t["control_smoothness"] for t in per_trial]
    return {
        "accumulated_cost_mean": sum(acc_costs) / len(acc_costs),
        "accumulated_cost_std": (sum((x - sum(acc_costs)/len(acc_costs))**2 for x in acc_costs) / len(acc_costs)) ** 0.5,
        "final_dist_mean": sum(dists) / len(dists),
        "final_dist_std": (sum((x - sum(dists)/len(dists))**2 for x in dists) / len(dists)) ** 0.5,
        "control_smoothness_mean": sum(smooths) / len(smooths),
        "control_smoothness_std": (sum((x - sum(smooths)/len(smooths))**2 for x in smooths) / len(smooths)) ** 0.5,
        "per_trial": per_trial,
    }


# ---------------------------------------------------------------------------
# Benchmark configurations
# ---------------------------------------------------------------------------
def run_benchmarks():
    results = {}

    for device in DEVICES:
        print(f"\n{'='*60}")
        print(f"Device: {device}")
        print(f"{'='*60}")

        dynamics = _make_dynamics(device)
        cost = _make_cost(device)
        terminal = _make_terminal_cost(device)
        noise_sigma = torch.eye(2, dtype=DTYPE, device=device)
        start = torch.tensor([-3.0, -2.0], dtype=DTYPE, device=device)

        # --- Vary num_samples (K) ---
        print(f"\n--- MPPI: Varying K (T=15) ---")
        for K in [50, 100, 500, 1000, 5000]:
            torch.manual_seed(SEED)
            ctrl = MPPI(dynamics, cost, 2, noise_sigma,
                        num_samples=K, horizon=15, device=device, lambda_=1.0)
            res = benchmark_command(ctrl, start)
            key = f"{device}/mppi/K={K}_T=15"
            results[key] = res
            print(f"  K={K:>5d}: {res['mean_s']*1000:>8.2f}ms (min={res['min_s']*1000:.2f}ms)")

        # --- Vary horizon (T) ---
        print(f"\n--- MPPI: Varying T (K=500) ---")
        for T in [5, 10, 15, 30, 50]:
            torch.manual_seed(SEED)
            ctrl = MPPI(dynamics, cost, 2, noise_sigma,
                        num_samples=500, horizon=T, device=device, lambda_=1.0)
            res = benchmark_command(ctrl, start)
            key = f"{device}/mppi/K=500_T={T}"
            results[key] = res
            print(f"  T={T:>5d}: {res['mean_s']*1000:>8.2f}ms (min={res['min_s']*1000:.2f}ms)")

        # --- MPPI with features ---
        print(f"\n--- MPPI: Feature variations (K=500, T=15) ---")
        feature_configs = [
            ("base", {}),
            ("terminal_cost", {"terminal_state_cost": terminal}),
            ("noise_abs_cost", {"noise_abs_cost": True}),
            ("bounded", {"u_max": torch.tensor([1.0, 1.0], dtype=DTYPE, device=device)}),
            ("M=3", {"rollout_samples": 3, "rollout_var_cost": 0.1}),
            ("null_action", {"sample_null_action": True}),
        ]
        for name, extra_kwargs in feature_configs:
            torch.manual_seed(SEED)
            ctrl = MPPI(dynamics, cost, 2, noise_sigma,
                        num_samples=500, horizon=15, device=device, lambda_=1.0,
                        **extra_kwargs)
            res = benchmark_command(ctrl, start)
            key = f"{device}/mppi_feat/{name}"
            results[key] = res
            print(f"  {name:<20s}: {res['mean_s']*1000:>8.2f}ms")

        # --- SMPPI ---
        print(f"\n--- SMPPI (K=500, T=15) ---")
        for w in [1.0, 5.0, 10.0]:
            torch.manual_seed(SEED)
            ctrl = SMPPI(dynamics, cost, 2, noise_sigma,
                         num_samples=500, horizon=15, device=device, lambda_=1.0,
                         w_action_seq_cost=w)
            res = benchmark_command(ctrl, start)
            key = f"{device}/smppi/w={w}"
            results[key] = res
            print(f"  w={w:<5.1f}: {res['mean_s']*1000:>8.2f}ms")

        # --- KMPPI ---
        print(f"\n--- KMPPI (K=500, T=15) ---")
        for nsp in [3, 5, 7]:
            torch.manual_seed(SEED)
            ctrl = KMPPI(dynamics, cost, 2, noise_sigma,
                         num_samples=500, horizon=15, device=device, lambda_=1.0,
                         num_support_pts=nsp, kernel=RBFKernel(sigma=2.0))
            res = benchmark_command(ctrl, start)
            key = f"{device}/kmppi/nsp={nsp}"
            results[key] = res
            print(f"  support_pts={nsp}: {res['mean_s']*1000:>8.2f}ms")

        # --- MPPI vs SMPPI vs KMPPI comparison ---
        print(f"\n--- Comparison: MPPI vs SMPPI vs KMPPI (K=500, T=15) ---")
        for label, ctrl_cls, extra in [
            ("MPPI", MPPI, {}),
            ("SMPPI", SMPPI, {"w_action_seq_cost": 5.0}),
            ("KMPPI", KMPPI, {"num_support_pts": 5, "kernel": RBFKernel(sigma=2.0)}),
        ]:
            torch.manual_seed(SEED)
            ctrl = ctrl_cls(dynamics, cost, 2, noise_sigma,
                            num_samples=500, horizon=15, device=device, lambda_=1.0,
                            **extra)
            res = benchmark_command(ctrl, start)
            key = f"{device}/compare/{label}"
            results[key] = res
            print(f"  {label:<8s}: {res['mean_s']*1000:>8.2f}ms")

        # --- Multi-step control loop ---
        print(f"\n--- Multi-step loop: 20 steps (K=500, T=15) ---")
        for label, ctrl_cls, extra in [
            ("MPPI", MPPI, {}),
            ("SMPPI", SMPPI, {"w_action_seq_cost": 5.0}),
            ("KMPPI", KMPPI, {"num_support_pts": 5, "kernel": RBFKernel(sigma=2.0)}),
        ]:
            torch.manual_seed(SEED)
            ctrl = ctrl_cls(dynamics, cost, 2, noise_sigma,
                            num_samples=500, horizon=15, device=device, lambda_=1.0,
                            **extra)
            res = benchmark_multi_step(ctrl, start, dynamics, num_steps=20)
            key = f"{device}/loop/{label}"
            results[key] = res
            print(f"  {label:<8s}: {res['mean_s']*1000:>8.2f}ms total, "
                  f"{res['per_step_s']*1000:.2f}ms/step")

        # --- Higher dimensional ---
        print(f"\n--- Higher dimensional (nx=10, nu=3, K=500, T=15) ---")
        nx, nu = 10, 3
        sigma_nd = torch.eye(nu, dtype=DTYPE, device=device)
        dyn_nd = _make_dynamics_nd(device, nx, nu)
        cost_nd = _make_cost_nd(device, nx)
        start_nd = torch.randn(nx, dtype=DTYPE, device=device)
        torch.manual_seed(SEED)
        ctrl = MPPI(dyn_nd, cost_nd, nx, sigma_nd,
                    num_samples=500, horizon=15, device=device, lambda_=1.0)
        res = benchmark_command(ctrl, start_nd)
        key = f"{device}/mppi/nx=10_nu=3"
        results[key] = res
        print(f"  nx=10, nu=3: {res['mean_s']*1000:>8.2f}ms")

        # ---------------------------------------------------------------
        # Solution quality evaluation
        # ---------------------------------------------------------------
        goal = GOAL.to(device=device)

        print(f"\n--- Solution quality: MPPI vs SMPPI vs KMPPI (K=500, T=15, 20 steps, 5 trials) ---")
        print(f"  {'Method':<8s}  {'Accum Cost':>12s}  {'Final Dist':>12s}  {'Ctrl Smooth':>12s}")
        for label, ctrl_cls, extra in [
            ("MPPI", MPPI, {}),
            ("SMPPI", SMPPI, {"w_action_seq_cost": 5.0}),
            ("KMPPI", KMPPI, {"num_support_pts": 5, "kernel": RBFKernel(sigma=2.0)}),
        ]:
            torch.manual_seed(SEED)
            ctrl = ctrl_cls(dynamics, cost, 2, noise_sigma,
                            num_samples=500, horizon=15, device=device, lambda_=1.0,
                            **extra)
            qres = evaluate_quality(ctrl, start, dynamics, cost, goal, num_steps=20, num_trials=5)
            key = f"{device}/quality/{label}"
            results[key] = qres
            print(f"  {label:<8s}  {qres['accumulated_cost_mean']:>9.2f}\u00b1{qres['accumulated_cost_std']:<5.2f}"
                  f"  {qres['final_dist_mean']:>9.4f}\u00b1{qres['final_dist_std']:<7.4f}"
                  f"  {qres['control_smoothness_mean']:>9.3f}\u00b1{qres['control_smoothness_std']:<6.3f}")

        print(f"\n--- Solution quality: varying K (T=15, 20 steps, 5 trials) ---")
        print(f"  {'K':<6s}  {'Accum Cost':>12s}  {'Final Dist':>12s}  {'Ctrl Smooth':>12s}")
        for K in [50, 100, 500, 1000]:
            torch.manual_seed(SEED)
            ctrl = MPPI(dynamics, cost, 2, noise_sigma,
                        num_samples=K, horizon=15, device=device, lambda_=1.0)
            qres = evaluate_quality(ctrl, start, dynamics, cost, goal, num_steps=20, num_trials=5)
            key = f"{device}/quality/mppi_K={K}"
            results[key] = qres
            print(f"  K={K:<4d}  {qres['accumulated_cost_mean']:>9.2f}\u00b1{qres['accumulated_cost_std']:<5.2f}"
                  f"  {qres['final_dist_mean']:>9.4f}\u00b1{qres['final_dist_std']:<7.4f}"
                  f"  {qres['control_smoothness_mean']:>9.3f}\u00b1{qres['control_smoothness_std']:<6.3f}")

        print(f"\n--- Solution quality: varying T (K=500, 20 steps, 5 trials) ---")
        print(f"  {'T':<6s}  {'Accum Cost':>12s}  {'Final Dist':>12s}  {'Ctrl Smooth':>12s}")
        for T in [5, 10, 15, 30]:
            torch.manual_seed(SEED)
            ctrl = MPPI(dynamics, cost, 2, noise_sigma,
                        num_samples=500, horizon=T, device=device, lambda_=1.0)
            qres = evaluate_quality(ctrl, start, dynamics, cost, goal, num_steps=20, num_trials=5)
            key = f"{device}/quality/mppi_T={T}"
            results[key] = qres
            print(f"  T={T:<4d}  {qres['accumulated_cost_mean']:>9.2f}\u00b1{qres['accumulated_cost_std']:<5.2f}"
                  f"  {qres['final_dist_mean']:>9.4f}\u00b1{qres['final_dist_std']:<7.4f}"
                  f"  {qres['control_smoothness_mean']:>9.3f}\u00b1{qres['control_smoothness_std']:<6.3f}")

    return results


def _serialize(obj):
    """Make results JSON-serializable by converting non-serializable values."""
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_serialize(v) for v in obj]
    elif isinstance(obj, float):
        return obj
    elif isinstance(obj, int):
        return obj
    else:
        return float(obj)


def print_summary(results):
    print(f"\n{'='*60}")
    print("SUMMARY — TIMING (all times in milliseconds)")
    print(f"{'='*60}")
    for key in sorted(results.keys()):
        r = results[key]
        if "mean_s" in r:
            print(f"  {key:<45s}  mean={r['mean_s']*1000:>8.2f}  min={r['min_s']*1000:>8.2f}")

    print(f"\n{'='*60}")
    print("SUMMARY — SOLUTION QUALITY")
    print(f"{'='*60}")
    print(f"  {'Key':<40s}  {'Accum Cost':>12s}  {'Final Dist':>12s}  {'Smoothness':>12s}")
    for key in sorted(results.keys()):
        r = results[key]
        if "accumulated_cost_mean" in r:
            print(f"  {key:<40s}  {r['accumulated_cost_mean']:>12.2f}  {r['final_dist_mean']:>12.4f}  {r['control_smoothness_mean']:>12.3f}")


if __name__ == "__main__":
    results = run_benchmarks()
    print_summary(results)

    # Save results to JSON for comparison
    outpath = "benchmark_results.json"
    with open(outpath, "w") as f:
        json.dump(_serialize(results), f, indent=2)
    print(f"\nResults saved to {outpath}")

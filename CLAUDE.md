# pytorch_mppi

Model Predictive Path Integral (MPPI) control library using approximate dynamics in PyTorch.
Implements batched trajectory sampling for GPU-accelerated model-based control.

## Project Structure

```
src/pytorch_mppi/
  __init__.py          # Exports MPPI, SMPPI, KMPPI
  mppi.py              # Core implementation (~625 lines): MPPI, SMPPI, KMPPI, run_mppi
  autotune.py          # Hyperparameter tuning infrastructure (CMA-ES local optimizer)
  autotune_global.py   # Ray Tune global search integration
  autotune_qd.py       # Quality Diversity optimization (CMA-ME via pyribs)
tests/
  pendulum.py                       # MPPI on true pendulum dynamics (gym)
  pendulum_approximate.py           # MPPI with learned neural network dynamics
  pendulum_approximate_continuous.py # Continuous angle representation variant
  smooth_mppi.py                    # Visual comparison of MPPI, SMPPI, KMPPI
  auto_tune_parameters.py           # Hyperparameter tuning example
  test_batch_wrapper.py             # Unit tests for handle_batch_input
```

## Architecture

### Class Hierarchy
- **MPPI** - Base class. Batched trajectory sampling with importance-weighted control update (Algorithm 2, Williams et al. 2017).
- **SMPPI(MPPI)** - Smooth MPPI. Lifts control space to penalize action rate of change. Maintains separate `action_sequence` and `U` (control differences).
- **KMPPI(MPPI)** - Kernel MPPI. Samples fewer support points, interpolates to full trajectory via RBF kernel. Uses `functorch.vmap` for batched interpolation.

### Key Data Flow (per `command()` call)
1. `shift_nominal_trajectory()` - Roll U forward, append u_init
2. `_compute_perturbed_action_and_noise()` - Sample K noise trajectories (K x T x nu), add to U, bound
3. `_compute_rollout_costs(perturbed_actions)` - **Hot loop**: iterate T timesteps, call user dynamics+cost each step
4. `_compute_total_cost_batch()` - Combine rollout cost + perturbation cost
5. `_compute_weighting()` - Softmax-like exponential weighting (omega)
6. Update U with weighted sum of noise perturbations

### Key Dimensions
- **K** = `num_samples` (trajectory samples, typically 100-1000)
- **T** = `horizon` (timesteps, typically 15-30)
- **M** = `rollout_samples` (stochastic dynamics replicates, usually 1)
- **nu** = control dimensions, **nx** = state dimensions

### User-Provided Functions
- `dynamics(state, action) -> next_state` — state is K x nx, action is K x nu
- `running_cost(state, action) -> cost` — cost is K x 1
- `terminal_state_cost(states, actions) -> cost` — optional, states is K x T x nx
- Wrapped by `@handle_batch_input(n=2)` from `arm_pytorch_utilities`

## Dependencies
- **torch** — core tensor operations
- **arm_pytorch_utilities** — `handle_batch_input` decorator for flexible batch dimensions
- **functorch** — `vmap` used in KMPPI for batched kernel interpolation
- **numpy** — minimal use (only in `get_params()` for display and in autotune)
- Optional: cma, ray[tune], bayesian-optimization, hyperopt (for autotune)

## Planned Optimization Refactor

Goal: Remove Python loops and make code compatible with `torch.compile`, similar to what was done for `pytorch_kinematics`.

### Performance-Critical Hot Path
The main bottleneck is `_compute_rollout_costs()` (mppi.py:254-267):
```python
for t in range(T):
    u = self.u_scale * perturbed_actions[:, t].repeat(self.M, 1, 1)
    next_state = self._dynamics(state, u, t)
    next_state = self._sample_specific_dynamics(next_state, state, u, t)
    state = next_state
    c = self._running_cost(state, u, t)
    cost_samples = cost_samples + c
    states.append(state)
    actions.append(u)
actions = torch.stack(actions, dim=-2)
states = torch.stack(states, dim=-2)
```

Similarly `get_rollouts()` (mppi.py:357-361) has a horizon loop.

### torch.compile Blockers
1. **Horizon for-loop with list appends** — `states.append()` / `torch.stack()` pattern. Fix: pre-allocate tensors, use index assignment.
2. **Shape-dependent control flow** (mppi.py:244): `if self.state.shape == (K, self.nx)`. Fix: use attribute flags or always reshape.
3. **`@handle_batch_input` decorator** — wraps dynamics/cost with runtime shape inspection. May need compile-friendly alternative or be applied outside compiled region.
4. **Optional feature branching** — `if self.terminal_state_cost`, `if self.M > 1`, `if self.specific_action_sampler is not None`. Fix: use guards at init time or compile separate variants.
5. **`from functorch import vmap`** — should migrate to `torch.vmap` (functorch is merged into PyTorch core).

### Vectorization Opportunities
- The horizon loop is inherently sequential (state[t+1] depends on state[t]), so it cannot be parallelized across T. However, `torch.compile` can still optimize the unrolled loop if graph breaks are eliminated.
- Pre-allocating `states` and `actions` tensors avoids dynamic list building.
- The `_compute_weighting` softmax computation is already vectorized.
- Noise sampling and action cost computation are already fully vectorized.

### Already Optimized
- Noise sampling: fully batched via `MultivariateNormal.rsample((K, T))`
- Action cost: matrix multiply K x T x nu @ nu x nu
- KMPPI kernel interpolation: uses vmap for batch kernel solve
- Cost weighting: vectorized exp + normalize

### Migration Notes
- `functorch.vmap` → `torch.vmap` (available since PyTorch 2.0)
- Consider whether `arm_pytorch_utilities.handle_batch_input` can be replaced with simpler reshape logic to reduce external dependencies
- User dynamics/cost functions must also be compile-friendly for full graph compilation; consider documenting this requirement or providing a fallback path

## Development

```shell
pip install -e .           # Dev install
pip install -e .[test]     # With test deps
KMP_DUPLICATE_LIB_OK=TRUE pytest tests/test_mppi.py tests/test_batch_wrapper.py -v  # Run tests
KMP_DUPLICATE_LIB_OK=TRUE python tests/benchmark_mppi.py  # Run benchmarks (saves benchmark_results.json)
```

Note: `KMP_DUPLICATE_LIB_OK=TRUE` is needed on this machine due to duplicate OpenMP libraries.

## Testing

### Test files
- `tests/test_mppi.py` — 64 tests covering MPPI/SMPPI/KMPPI correctness and solution quality
- `tests/test_batch_wrapper.py` — 2 tests for handle_batch_input decorator
- `tests/benchmark_mppi.py` — Timing + solution quality benchmarks

### Test categories in test_mppi.py
- **TestMPPI** (26): shapes, bounds, features, determinism, state handling
- **TestSMPPI** (11): SMPPI-specific behavior, smoothness, action sequences
- **TestKMPPI** (12): kernel interpolation, support points, stability
- **TestSpecificActionSampler** (1): custom sampler integration
- **TestEdgeCases** (5): numpy input, high-dim, 1-sample, float32
- **TestSolutionQuality** (9): goal convergence, cost bounds, determinism, bounded actions

### Baseline performance (CPU, K=500, T=15)
| Controller | per-command | 20-step loop |
|---|---|---|
| MPPI | 0.63ms | 12.9ms |
| SMPPI | 0.68ms | 13.7ms |
| KMPPI | 1.05ms | 20.5ms |

### Baseline solution quality (CPU, K=500, T=15, 20 steps, 5 trials)
| Controller | Accum Cost | Final Dist | Control Smoothness |
|---|---|---|---|
| MPPI | 113.5 +/- 17.6 | 1.59 +/- 0.96 | 57.8 +/- 9.0 |
| KMPPI | 111.0 +/- 12.2 | 1.61 +/- 0.58 | 25.9 +/- 3.2 |

Note: SMPPI quality is highly environment-dependent; it requires careful tuning (action bounds, terminal cost) per environment. KMPPI achieves similar cost to MPPI but with 2x smoother control.

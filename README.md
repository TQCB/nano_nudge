# nano-nudge

> A lightweight Python library for surrogate-based Bayesian Optimization.

`nano-nudge` provides a simple, understandable implementation of Bayesian Optimization for finding the optimal parameters of expensive-to-evaluate functions.

## Core Concepts

Bayesian Optimization works by building a probabilistic model (a "surrogate model") of the objective function. This model is then used to intelligently select the most promising points to evaluate next.

`nano-nudge` is built around a few key components:
*   **`SurrogateBayesianOptimizer`**: The main class that orchestrates the optimization process.
*   **`SurrogateModel`**: A model that learns the shape of your objective function from past trials.
*   **`Acquisition Function`**: A function (e.g., Expected Improvement) that uses the surrogate model's predictions to quantify how "promising" a candidate point is.
*   **`Trial`**: An object that stores a parameter suggestion and its resulting value.

## Demo

Optimizing a function with `nano-nudge` is straightforward. Here is a complete example of finding the maximum of a simple function.

```python
import numpy as np
from nano_nudge.optimizer import SurrogateBayesianOptimizer

def objective(trial):
  x = trial.suggest_float(min=-10, max=10)
  return np.exp((-x**2)/2)/np.sqrt(2*np.pi)

optimizer = SurrogateBayesianOptimizer(objective=objective, n_trials=30, direction='maximize')
optimizer.optimize()

print("\n")
print("Best trial:")
print(optimizer.history.get_best_trial())
```
```bash
Trial number: 0 | Suggestion: 7.769077156726528 | Value: 3.120436310418126e-14
Trial number: 1 | Suggestion: 8.678406672808347 | Value: 1.764093319457085e-17
Trial number: 2 | Suggestion: 1.1376615768300837 | Value: 0.20886326722844653
(...)
Trial number: 27 | Suggestion: 0.6785836994168211 | Value: 0.31689764243866037
Trial number: 28 | Suggestion: -0.6012088625574608 | Value: 0.3329827535734924
Trial number: 29 | Suggestion: 0.43840897881871577 | Value: 0.3623880247819842


Best trial:
Trial number: 16 | Suggestion: 0.03761830347918682 | Value: 0.39866010130132035
```

## Future Work

`nano-nudge` is in active development. Here are some of the planned features:

- **Parallel Optimization**: Implement q-Expected Improvement (q-EI) to allow for evaluating multiple trials in parallel.
- **More Model**: Allow the use of other surrogate models
- **Better Handling of Search-Space Bounds and Sampling**: Currently the solution to handle these syttems is inelegant and requires significant overhauling.
- **Tree-Structured Parzen Estimation**: Implement TPE as an alternative bayesian optimisation strategy
- **Expanded Search Space**: Add support for categorical and integer hyperparameters, in addition to continuous ones.
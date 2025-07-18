import numpy as np
from nano_nudge.optimizer import ParzenOptimizer

def objective(trial):
  x = trial.suggest_float(min=-10, max=10)
  return np.exp((-x**2)/2)/np.sqrt(2*np.pi)

optimizer = ParzenOptimizer(objective=objective, n_trials=100, direction='maximize')
optimizer.optimize()

print("\n")
print("Best trial:")
print(optimizer.history.get_best_trial())
from typing import Callable

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

def expected_improvement(x_mu: float, x_std: float, f_x_plus: float) -> float:
  z = (x_mu - f_x_plus) / x_std
  result =  (x_mu - f_x_plus) * norm.cdf(z) + x_std * norm.pdf(z)
  return float(result)

def multi_start_lbfgs(
  objective_function: Callable,
  search_space_bounds: list[tuple[float, float]],
  n_restarts: int=25,
  ) -> float:
  """
    Optimize a function using multi-start L-BFGS-B.
    
    Args:
      objective_function: A function to minimize.
      x_plus: The best point found so far.
      f_x_plus: The value at the best point found so far.
      search_space_bounds: The bounds of the search space.
      n_restarts: The number of random starting points for the optimizer.
      
    Returns:
      The point that maximizes the function.
  """

  best_x = None
  best_value = np.inf # EI is >= 0 so -1 makes sense, but won't work for other functions

  # Generate random starting points from our bounds
  n_dims = len(search_space_bounds)
  random_starts = np.random.uniform(
    low=[b[0] for b in search_space_bounds],
    high=[b[1] for b in search_space_bounds],
    size=(n_restarts, n_dims)
  )

  # Find best x, starting from all start points
  for start_point in random_starts:
    result = minimize(
      fun=objective_function,
      x0=start_point,
      method='L-BFGS-B',
      bounds=search_space_bounds
    )
    
    if result.fun < best_value:
      best_value = result.fun
      best_x = result.x
    
  return float(best_x)
from typing import Callable, Literal

import numpy as np

from .kde import KDE
from .surrogate_model import SurrogateModel
from .trial import Trial, TrialHistory, FrozenTrial
from .acquisition import expected_improvement, multi_start_lbfgs

type t_Direction = Literal["minimize", "maximize"]

class GammaSchedule:
  def __init__(
      self,
      start: float,
      end: float,
      max_steps: int,
      method: str = 'linear',
      ):
    self.start = start
    self.end = end
    self.max_steps = max_steps

    self.current_step = 0

    self.values = self.calculate_values(method)

  def calculate_values(self, method):
    if method == 'linear':
      values = np.linspace(self.start, self.end, self.max_steps)
      return values

  def step(self):
    self.current_step += 1
    return self.values[self.current_step]


class Optimizer:
  def __init__(
      self,
      objective: Callable[[Trial], float],
      n_trials: int,
      direction: t_Direction,
  ):
    self.history = TrialHistory()

    self.objective = objective
    self.n_trials = n_trials
    self.direction = direction

class OptimizerWithStartup(Optimizer):
  def __init__(
      self,
      objective: Callable[[Trial], float],
      n_trials: int,
      direction: t_Direction,
      n_startup_trials: int,
  ):
    super().__init__(objective, n_trials, direction)

    if n_startup_trials > n_trials:
      raise ValueError("You cannot have more startup trials than trials: increase n_trials or decrease n_startup_trials.")
    
    self.n_startup_trials = n_startup_trials

  def run_startup_trials(self):
    for i in range(self.n_startup_trials):
      trial = Trial()
      value = self.objective(trial)
      
      frozen_trial = FrozenTrial(i, trial, value)
      self.history.add_trial(frozen_trial)
      print(frozen_trial)

class SurrogateBayesianOptimizer(OptimizerWithStartup):
  def __init__(
    self,
    objective: Callable[[Trial], float],
    n_trials: int,
    direction: t_Direction,
    n_startup_trials: int=10,
    acquisition_function: Callable[[float, float, float], float]=expected_improvement
    ):

    super().__init__(objective, n_trials, direction, n_startup_trials)
    
    self.acquisition_function = acquisition_function
    self.surrogate_model = SurrogateModel()
    
    self.surrogate_objective = self.create_surrogate_objective(direction)
  
  def fit_surrogate_model(self):
    x, y = self.history.convert_to_xy()
    self.surrogate_model.fit(x, y)
    
  def create_surrogate_objective(self, direction) -> Callable[[float, float], float]:
    def directionless_surrogate_objective(x, f_x_plus):
      prediction = self.surrogate_model.predict(x)
      x_mu, x_sigma = prediction.mu, prediction.sigma
      x_sigma += 1e-8
      
      result = self.acquisition_function(x_mu, x_sigma, f_x_plus)
      
      return result
    
    if direction == "minimize":
      def surrogate_objective(x: float, f_x_plus: float) -> float:
        return directionless_surrogate_objective(x, f_x_plus)
      
    if direction == "maximize":
      def surrogate_objective(x: float, f_x_plus: float) -> float:
        return -directionless_surrogate_objective(x, f_x_plus)

    return surrogate_objective
      
  def run_optimization_loop(self):
    for i in range(self.n_startup_trials, self.n_trials):
      self.fit_surrogate_model()
      
      f_x_plus = self.history.get_f_x_plus()
      suggestion = multi_start_lbfgs(lambda x: self.surrogate_objective(x, f_x_plus), search_space_bounds=[(-5, 5)])
      suggestion = float(suggestion[0])
      
      trial = Trial(suggestion)
      value = self.objective(trial)
      
      frozen_trial = FrozenTrial(i, trial, value)
      self.history.add_trial(frozen_trial)
      
      print(frozen_trial)
    
  def optimize(self):
    self.run_startup_trials()
    self.run_optimization_loop()

class ParzenOptimizer(OptimizerWithStartup):
  def __init__(
      self,
      objective: Callable[[Trial], float],
      n_trials: int,
      direction: t_Direction,
      n_startup_trials: int=20,
      gamma_start: float=0.25,
      gamma_end: float=0.05,
      sampling_size: int=100,
  ):
    super().__init__(objective, n_trials, direction, n_startup_trials)
    self.gamma_schedule = GammaSchedule(gamma_start, gamma_end, n_trials)
    self.sampling_size = sampling_size

  def split_array(self, arr, gamma):
    thresh = int(gamma * len(arr))
    
    x1 = arr[-thresh:]
    x2 = arr[:-thresh]

    return x1, x2

  def split_points_lg(self) -> tuple[np.ndarray, np.ndarray]:
    x, y = self.history.convert_to_xy()

    sorted_idx = np.argsort(y)
    sorted_x = x[sorted_idx]

    gamma = self.gamma_schedule.step()
    x_l, x_g = self.split_array(sorted_x, gamma)

    return x_l, x_g
  
  def generate_distributions(self, x_l, x_g):
    l = KDE(x_l)
    g = KDE(x_g)

    return l, g
  
  def evaluate_improvement(self, l_samples):
    l_of_x = self.l.evaluate(l_samples)
    g_of_x = self.g.evaluate(l_samples)

    improvement = l_of_x / g_of_x
    return improvement

  def run_optimization_loop(self):
    for i in range(self.n_startup_trials, self.n_trials):
      x_l, x_g = self.split_points_lg()
      self.l, self.g = self.generate_distributions(x_l, x_g)

      l_samples = self.l.sample(self.sampling_size)
      sample_improvements = self.evaluate_improvement(l_samples)

      suggestion = l_samples[np.argmax(sample_improvements)]
      suggestion = float(suggestion)

      trial = Trial(suggestion)
      value = self.objective(trial)

      frozen_trial = FrozenTrial(i, trial, value)
      self.history.add_trial(frozen_trial)

      print(frozen_trial)

  def optimize(self):
    self.run_startup_trials()
    self.run_optimization_loop()
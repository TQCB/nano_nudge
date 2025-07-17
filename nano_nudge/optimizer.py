from .surrogate_model import SurrogateModel
from .trial import Trial, TrialHistory, FrozenTrial
from .acquisition import expected_improvement, multi_start_lbfgs

class Optimizer:
  def __init__(
      self,
      objective,
      n_trials,
      direction,
  ):
    self.history = TrialHistory()

    self.objective = objective
    self.n_trials = n_trials
    self.direction = direction

  def run_startup_trials(self):
    for i in range(self.n_startup_trials):
      trial = Trial()
      value = self.objective(trial)
      
      frozen_trial = FrozenTrial(i, trial, value)
      self.history.add_trial(frozen_trial)
      print(frozen_trial)

class SurrogateBayesianOptimizer(Optimizer):
  def __init__(
    self,
    objective,
    n_trials,
    direction,
    n_startup_trials=10,
    acquisition_function=expected_improvement
    ):

    super().__init__(objective, n_trials, direction)

    if n_startup_trials > n_trials:
      raise ValueError("You cannot have more startup trials than trials: increase n_trials or decrease n_startup_trials.")
    
    self.acquisition_function = acquisition_function
    
    self.n_startup_trials = n_startup_trials
    
    self.surrogate_model = SurrogateModel()
    
    self.surrogate_objective = self.create_surrogate_objective(direction)
      
  def fit_surrogate_model(self):
    x, y = self.history.convert_to_xy()
    self.surrogate_model.fit(x, y)
    
  def create_surrogate_objective(self, direction):
    def directionless_surrogate_objective(x, f_x_plus):
      prediction = self.surrogate_model.predict(x)
      x_mu, x_sigma = prediction.mu, prediction.sigma
      x_sigma += 1e-8
      
      result = self.acquisition_function(x_mu, x_sigma, f_x_plus)
      
      return result
    
    surrogate_objective = lambda x, f_x_plus: directionless_surrogate_objective(x, f_x_plus)
    if direction == "maximize":
      surrogate_objective = lambda x, f_x_plus: -directionless_surrogate_objective(x, f_x_plus)

    return surrogate_objective
      
  def run_optimization_loop(self):
    for i in range(self.n_startup_trials, self.n_trials):
      self.fit_surrogate_model()
      
      f_x_plus = self.history.get_f_x_plus()
      suggestion = multi_start_lbfgs(lambda x: self.surrogate_objective(x, f_x_plus), search_space_bounds=[[-5, 5]])
      suggestion = float(suggestion[0])
      
      trial = Trial(suggestion)
      value = self.objective(trial)
      
      frozen_trial = FrozenTrial(i, trial, value)
      self.history.add_trial(frozen_trial)
      
      print(frozen_trial)
    
  def optimize(self):
    self.run_startup_trials()
    self.run_optimization_loop()

class ParzenOptimizer(Optimizer):
  def __init__(
      objective,
      n_trials,
      direction,
  ):
    super().__init__(objective, n_trials, direction)
    pass
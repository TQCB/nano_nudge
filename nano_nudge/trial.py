from typing import Optional

import numpy as np

class Trial:
  def __init__(self, next_suggestion: Optional[float] =None):
    self.suggestion = next_suggestion
  
  def suggest_float(self, min: float, max: float) -> float:
    # if trial was created with a suggested value, then we suggest that directly
    if self.suggestion:
      return self.suggestion
    
    self.suggestion = np.random.uniform(min, max)
    return self.suggestion
  
class FrozenTrial:
  def __init__(self, iteration: int, trial: Trial, value: float):
    self.iteration = iteration
    self.suggestion = trial.suggestion
    self.value = value
  
  def __repr__(self):
    return f"Trial number: {self.iteration} | Suggestion: {self.suggestion} | Value: {self.value}"
  
class TrialHistory:
  def __init__(self):
    self.history: list[FrozenTrial] = []
    
  def add_trial(self, frozen_trial: FrozenTrial):
    self.history.append(frozen_trial)
    
  def convert_to_xy(self):
    x = [trial.suggestion for trial in self.history]
    y = [trial.value for trial in self.history]

    x = np.array(x).reshape(-1, 1)
    y = np.array(y)
    
    return x, y
  
  def get_f_x_plus(self) -> float:
    values = [trial.value for trial in self.history]
    return np.max(values)
    
  def get_best_trial(self):
    values = [trial.value for trial in self.history]
    best_trial = self.history[np.argmax(values)]
    return best_trial
    
  def __repr__(self):
    return "".join([frozen_trial.__repr__() for frozen_trial in self.history])
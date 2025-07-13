import numpy as np

def trial_history_to_train_data(trial_history):
  x = [trial['trial'].suggestion for trial in trial_history]
  y = [trial['value'] for trial in trial_history]
  
  x = np.array(x).reshape(-1, 1)
  
  return x, y

def find_x_plus(trial_history):
  values = [trial['value'] for trial in trial_history]
  best_idx = np.argmax(values)
  
  x_plus = trial_history[best_idx]['trial'].suggestion
  f_x_plus = values[best_idx]
  
  return x_plus, f_x_plus
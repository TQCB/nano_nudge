import numpy as np
from sklearn.ensemble import RandomForestRegressor

class Prediction:
  def __init__(self, mu, sigma):
    self.mu = mu
    self.sigma = sigma

class SurrogateModel:
  def __init__(self):
    self.model = RandomForestRegressor(n_estimators=100, max_depth=4)
    
  def fit(self, x, y):
    self.model.fit(x, y)
    
  def score(self, x, y):
    return self.model.score(x, y)
    
  def predict(self, x):
    x = x.reshape(-1, 1)
    predictions = []
    for estimator in self.model.estimators_:
      predictions.append(estimator.predict(x))
    
    mu = np.mean(predictions)
    sigma = np.std(predictions)
    ensemble_prediction = Prediction(mu, sigma)
    return ensemble_prediction
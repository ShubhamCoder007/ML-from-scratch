import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

#Get your Data set
#
##################

class SGDRegressor:

  def __init__(self) -> None:
    self.coef_ = None
    self.intercept_ = None

  def fit(self, X_train, y_train, lr=0.01, epochs=100):
    self.intercept_ = 0
    self.coef_ = np.ones(X_train.shape[1])

    for _ in range(epochs):
      #randomly sample the row
      idx = np.random.randint(0, X_train.shape[0])

      y_hat = self.predict(X_train[idx])
      intercept_der = -2*(y_train[idx] - y_hat)
      self.intercept_ = self.intercept_ - lr*intercept_der

      coef_der = -2*np.dot((y_train[idx]-y_hat), X_train[idx])
      self.coef_ = self.coef_ - lr*coef_der

  def predict(self, X_test):
    return np.dot(self.coef_, X_test.T) + self.intercept_

  def __str__(self) -> str:
    return f"Fitting complete:-\ncoeficients:{self.coef_}\nintercept:{self.intercept_}"



#usage
sgd = SGDRegressor()
sgd.fit(X_train, y_train,lr=0.2,epochs=1000)
print(sgd)

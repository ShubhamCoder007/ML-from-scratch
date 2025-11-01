class GDRegressor:

  def __init__(self) -> None:
    self.coef_ = None
    self.intercept_ = None

  def fit(self, X_train, y_train, lr=0.1, epochs=100):
    self.coef_ = np.ones(X_train.shape[1])
    self.intercept_ = 0

    for _ in range(epochs):
      # y_hat = np.dot(X_train, self.coef_) + self.intercept_
      y_hat = self.predict(X_train)
      intercept_der = -2 * np.mean((y_train - y_hat))
      self.intercept_ = self.intercept_ - lr*intercept_der

      coef_der = -2 * np.dot((y_train - y_hat), X_train) / X_train.shape[1]
      self.coef_ = self.coef_ - lr*coef_der

  def predict(self, X_test):
    return np.dot(X_test, self.coef_) + self.intercept_

  def __str__(self) -> str:
    return f"Fitting complete:-\ncoeficients:{self.coef_}\nintercept:{self.intercept_}"
    

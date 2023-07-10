import numpy as np
class Logistic_Regression ():
  
  # declaring learning rate and number of iteration(Hyperparameters)
  def __init__(self,learning_rate,no_of_iteration):

    self.learning_rate = learning_rate
    self.no_of_iteration = no_of_iteration


  # fit the function to train our model with the dataset 
  def fit(self,X,Y):

    # no of data points in the dataset (number of rows) ==> m
    # no of input features in the dataset (number of columns )==> n
    self.m , self.n = X.shape

    #initiate the weight and bais value 
    self.w = np.zeros(self.n)
    self.b = 0

    self.X = X
    self.Y = Y

    # implementing gradient descent 
    for i in range(self.no_of_iteration):
      self.update_weight()


  def update_weight(self):

    # we need the formular for y_hat first (sigma function)

    Y_hat = 1 / (1 + np.exp( - (self.X.dot(self.w) + self.b) ))  # wx + b

    # for the derivatives

    dw = (1/self.m)*np.dot(self.X.T, (Y_hat - self.Y)) # taking the transpose number of column of one matrix should match no of row in the next
                                                       # X = [769 x 8]  Y = [769 x 1]

    db = (1/self.m)*np.sum(Y_hat - self.Y)
    
    # updating the weight and bais using gradient descent equation

    self.w = self.w - self.learning_rate * dw

    self.b = self.b - self.learning_rate * db

    # Sigmoid Equation & Decision Boundary
  def predict(self, X):  
    
    Y_pred =  1 / (1 + np.exp( - (X.dot(self.w) + self.b) )) 
    Y_pred = np.where(Y_pred > 0.5 , 1 , 0)
    return Y_pred

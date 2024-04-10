import math
import torch
import numpy as np

class LinearModel:

    def __init__(self):
        self.w = None 
        self.p_w = None

    def score(self, X):
        """
        Compute the scores for each data point in the feature matrix X. 
        The formula for the ith entry of s is s[i] = <self.w, x[i]>. 

        If self.w currently has value None, then it is necessary to first initialize self.w to a random value. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            s torch.Tensor: vector of scores. s.size() = (n,)
        """

        # if weights is empty, return 1d tensor with random numbers
        if self.w is None: 
            self.w = torch.rand((X.size()[1]))

        # matrix multiplication X and weights
        return torch.matmul(X, self.w)

    def predict(self, X):
        """
        Compute the predictions for each data point in the feature matrix X. The prediction for the ith 
        data point is either 0 or 1. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            y_hat, torch.Tensor: vector predictions in {0.0, 1.0}. y_hat.size() = (n,)
        """

        # compute scores
        scores = self.score(X)

        # yhat = 1 if score >= 0; 1 otherwise
        y_hat = 1.0 * (scores >= 0)
        return y_hat
    
class LogisticRegression(LinearModel):
    
    def loss(self, X,y):
        """
        Compute empirical risk using logistic loss function 
        
        ARGUMENTS:
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 
            y, torch.Tensor: target vector, {0,1}. y is shape (n, ).

        RETURNS:
            loss, float: value of the model's loss
        """
        # helper definitions
        score = self.score(X)
        sigmoid = torch.sigmoid(score)

        # loss function
        loss = -y * torch.log(sigmoid) - (1-y)*torch.log(1-sigmoid)
        return torch.mean(loss)
    
    def grad(self, X, y):
        """
        Compute the gradient of the empirical risk L(w)
        
        ARGUMENTS:
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 
            y, torch.Tensor: target vector, {0,1}. n x 1 dimensional.

        RETURNS:
            grad, float: 
        """

        # helper definitions
        score = self.score(X)
        sigmoid = torch.sigmoid(score)

        # gradient loss function
        g = (sigmoid - y)[:, None] # convert tensor with shape (n,) to shape (n,1)
        grad = g * X

        # return mean gradient
        mean = torch.mean(grad, dim = 0)
        return mean
    
class GradientDescentOptimizer:

    def __init__(self, model):
        self.model = model

    def step(self, X, y, alpha, beta):
        """
        Compute a step of the gradient update
        
        ARGUEMNTS:
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 
            y, torch.Tensor: target vector, {0,1}. n x 1 dimensional.
            alpha, float: learning rate parameter
            beta, float: learning rate parameter. WHen beta = 0 we have 
            regular gradient descent.
        """
        gradient = self.model.grad(X,y)
        loss = self.model.loss(X,y)
        currentWeight = self.model.w

        # If first update, previous update does not exist so just alpha * gradient
        if self.model.p_w == None:
            self.model.w -= alpha * gradient

        # all other updates, compute step
        else:
            self.model.w = currentWeight -1*alpha*gradient + beta*(currentWeight - self.model.p_w)
        
        # update previous weight value
        self.model.p_w = currentWeight

        return loss
import numpy as np
import util
import matplotlib.pyplot as plt
import math
from random import *

def main(train_path, valid_path, save_path,image_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***

    # Train a logistic regression classifier
    LR = LogisticRegression()
    LR.plms_fit(x_train, y_train)

    # Plot decision boundary on top of validation set set
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    util.plot(x_valid, y_valid, LR.theta, image_path, correction=1.0)
    # Use np.savetxt to save predictions on eval set to save_path
    np.savetxt(save_path,LR.predict(x_valid,y_valid))
    # *** END CODE HERE ***


class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=100000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def plms_fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        def logisticGradient(w,x,y):
            g1=np.array([0.0 for k in range(len(w))])
            g=np.reshape(g1, (len(g1), 1))
            for k in range(len(y)):
                x_k_new = np.reshape(x[k], (len(x[k]), 1))
                g+=(-1/len(y))*(y[k]-1/(1+np.exp(-np.dot(np.transpose(w),x_k_new))))*x_k_new
            return g

        delta=1e4
        t=0
        if self.theta==None:
            self.theta=np.ones_like(x[0,:])
        self.theta=np.reshape(self.theta,(len(self.theta),1))
        while (np.linalg.norm(delta)>self.eps and t<self.max_iter):
            old_theta = self.theta
            grad=logisticGradient(old_theta, x, y)
            delta = self.step_size * grad
            var = abs(np.linalg.norm(delta, axis=0)) / delta.shape[0]
            noise = np.where(abs(old_theta) < var, np.random.normal(0, var, delta.shape), np.zeros_like(delta))
            update = (delta * np.sign(old_theta)+noise) * old_theta
            self.theta = old_theta - update
            t+=1
            print(self.theta)
            if t%1000==0:
                print(t)
        print(t)
        print(self.theta)
        print(np.min(np.abs(self.theta)))
        # *** END CODE HERE ***

    def newton_fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        def logisticGradient(w,x,y):
            g1=np.array([0.0 for k in range(len(w))])
            g=np.reshape(g1, (len(g1), 1))
            for k in range(len(y)):
                x_k_new = np.reshape(x[k], (len(x[k]), 1))
                g+=(-1/len(y))*(y[k]-1/(1+np.exp(-np.dot(np.transpose(w),x_k_new))))*x_k_new
            return g

        def logisticHessian(w,x,y):
            h=np.zeros((len(w),len(w)))
            for k in range(len(y)):
                x_k_new = np.reshape(x[k], (len(x[k]), 1))
                h+=(1/len(y))*np.dot(x_k_new,np.transpose(x_k_new))*(1/(1+np.exp(-np.dot(np.transpose(w),x_k_new))))*(1-(1/(1+np.exp(-np.dot(np.transpose(w),x_k_new)))))
            return h

        delta=1e4
        t=0
        if self.theta==None:
            self.theta=np.array([0.0 for k in range(len(x[0]))])
        self.theta=np.reshape(self.theta,(len(self.theta),1))
        while delta>self.eps and t<self.max_iter:
            old_theta=self.theta
            self.theta=old_theta-self.step_size*np.dot(np.linalg.inv(logisticHessian(old_theta,x,y)),logisticGradient(old_theta,x,y))
            delta=np.sum(np.abs(self.theta - old_theta))
            t+=1
            if t % 1000 == 0:
                print(t)
        print(t)
        print(self.theta)
        # *** END CODE HERE ***

    def predict(self, x, y):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        results=np.array([0.0 for k in range(len(x))])
        FP=0
        FN=0
        for k in range(len(x)):
            new_x_k=np.reshape(x[k],(len(x[k]),1))
            results[k]=1/(1+np.exp(-np.dot(np.transpose(self.theta),new_x_k)))
            if np.dot(np.transpose(self.theta),new_x_k)>=0:
                if y[k]==0.0:
                    FP+=1
            elif np.dot(np.transpose(self.theta),new_x_k)<0:
                if y[k]==1.0:
                    FN+=1
        print(FP,FN,(FP+FN)/len(y),1-(FP+FN)/len(y))
        return results
        # *** END CODE HERE ***

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='logreg_pred_0001.txt',
         image_path='logreg01.png')

    #main(train_path='ds2_train.csv',
    #     valid_path='ds2_valid.csv',
    #     save_path='logreg_pred_0002.txt',
    #     image_path='logreg02.png')

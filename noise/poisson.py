import numpy as np
import p_util as util
import matplotlib.pyplot as plt
import multiprocessing
import time

def run_trial(i, lr=0.25):
    time.sleep(0.01*i)
    print("\n\nStarting trial number", i)
    
    # Load training set
    ideal = PoissonRegression(theta_0 = np.random.normal(0,2,4))

    cov = np.random.normal(0,.1,(4,4))
    cov = cov.dot(cov.T)

    x_train = np.random.multivariate_normal(np.zeros(4),cov,2500)
    y_train = np.random.poisson(ideal.predict(x_train), 2500)

    y = np.zeros((5,2))

    # Fit LMS model
    clf = PoissonRegression(step_size=lr)
    #x[0,i,:] = clf.fit(x_train, y_train)   
    y[0,:] = clf.fit(x_train, y_train)
    print("Finished with part 0 of trial", i)

    # Fit Gradient Variance % LMS model
    percent_clf = PoissonRegression(step_size=lr)
    #x[1,i,:] = percent_clf.fit(x_train, y_train, True)
    y[1,:] = percent_clf.fit(x_train, y_train, True)
    print("Finished with part 1 of trial", i)

    # Fit Constant Variance % LMS model
    for j,var in enumerate([1e-3,1e-4,1e-5]):
        norm_clf = PoissonRegression(step_size=lr)
        #x[j+2,i,:] = norm_clf.fit(x_train, y_train, True, var)
        y[j+2,:] = norm_clf.fit(x_train, y_train, True, var)
        print("Finished with part", j+2, "of trial", i)

    # Zero loss deviation against LMS loss
    for j in range(4,-1,-1):
        #x[j,i,0] -= x[0,i,0]
        y[j,0] -= y[0,0]

    return i, y

def main():
    """Problem: Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        save_path: Path to save predictions.
    """

    # Number of trials to run in experiment
    n = 8
    x = np.zeros((5,n,2))
    #x = np.load('vardata_15.npy')

    # Run each trial on a different processor
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        for i, y in pool.imap_unordered(run_trial, range(n)):
            x[:,i,:] = y

    # Save convergence/loss data from experiment for later analysis
    np.save('vardata_'+format(n)+'.npy', x)

    # Graph data
    util.plot_points(x[:,:,0], x[:,:,1], 'vardata_'+format(n)+'.png')


class PoissonRegression:
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, step_size=1e-5, max_iter=5000, eps=1e-4,
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
    
    def fit(self, x, y, percent=False, constvar=0):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        self.theta = np.ones_like(x[0,:])
        conv = self.max_iter
        for i in range(self.max_iter):

            #Stochastic GD:
            #xi = x[int(i % y.size),:]
            #yi = y[int(i % y.size)]
            #if np.dot(xi, self.theta) > 700:
            #    continue
            #delta = np.multiply(xi, yi - np.exp(np.dot(xi, self.theta)))*self.step_size

            #Full Batch GD:
            delta = np.dot(x.T, y - [np.exp(z) for z in np.dot(x, self.theta)])*self.step_size/y.size
            noise = np.zeros_like(delta)
            var = abs(np.linalg.norm(delta)*(constvar == 0) + constvar) if percent else 0
            if percent:
                noise = np.random.normal(0,var,delta.size)
                delta = np.multiply(delta, np.absolute(self.theta))

            #Normalize:
            #if norm:
                #delta /= np.linalg.norm(self.theta)

            self.theta += delta + np.where(abs(self.theta) < var, noise, np.zeros_like(delta))

            #if i%500 == 0:
                #print(".", end="")
            if np.linalg.norm(delta) < self.eps:
                #print("\n", self.theta)
                print("Converged after", i, "iterations")
                conv = i
                break

        loss = -np.dot(np.dot(x, self.theta), y)
        for i in range(y.size):
            loss += np.sum([np.log(k) for k in range(2, int(y[i]))]) + np.exp(np.dot(x[i], self.theta))
        print("Loss on Training Data:", loss/y.size)
        return loss/y.size, conv

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Floating-point prediction for each input, shape (n_examples,).
        """
        return np.exp(np.dot(x, self.theta))

if __name__ == '__main__':
    main()

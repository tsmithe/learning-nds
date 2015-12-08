#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

# RBF class not implemented yet...
#
# class RBF:
#     def __init__(self, means, covariances, A, B, b, Q):


###
### Example timeseries: the Tinkerbell map (it's differentiable!)
###

def u(k, dim=2):
    """This is just an empty input function for now.
    """
    return np.zeros(dim)


def F(x, u):
    x_ = np.zeros(2)
    x_[0] = x[0]**2 - x[1]**2 + 0.9*x[0] - 0.6013*x[1]
    x_[1] = 2*x[0]*x[1] + 2*x[0] + 0.5*x[1]
    return x_


def dF_dx(x, u):
    dF0_dx0 = 2*x[0] + 0.9
    dF1_dx0 = 2*x[1] + 2
    dF0_dx1 = -2*x[1] - 0.6013
    dF1_dx1 = 2*x[0] + 0.5
    return np.array([[dF0_dx0, dF0_dx1],
                     [dF1_dx0, dF1_dx1]])
    

def H(x):
    return x


def dH_dx(x):
    return np.eye(2)


###
### Helper functions to generate a noisy timeseries, train a filter,
### and plot the results
###

def generate_timeseries(stop=10000, x0=np.array([-0.72, -0.64]),
                        R_v=np.eye(2)*0, R_n=np.eye(2)*0.001):
    """
    stop is the number of iterations;
    x0 is the initial condition;
    R_v is the process noise (zero by default);
    R_n is the observation noise.
    """
    dim = 2 # Number of dimensions for the system
    U, Y = [], []

    x = x0
    for k in range(stop):
        U.append(u(k, dim))
        x = F(x, U[-1]) + np.random.multivariate_normal(np.zeros(dim), R_v)
        Y.append(H(x) + np.random.multivariate_normal(np.zeros(dim), R_n))

    return U, Y, R_v, R_n


def train_filter(filt, U, Y):
    """
    filt is your filter class, U the input vector, and Y the output vector.
    """
    n = len(U)
    X = []
    for i in range(n):
        x_hat_bar, P_bar = filt.time_update(U[i])
        x_hat, P = filt.measurement_update(Y[i], P_bar, x_hat_bar)
        print(i, x_hat)
        X.append(x_hat)
    return np.array(X)


def plot_states(F, U, X_hat, x0=np.array([-0.72, -0.64])):
    """
    Given input U, state transition function F, initial conditions x0,
    and state estimates X_hat, plot the estimates and the true states
    one one figure.

    NB: This assumes the states X are of dimension 2!
    """
    n = len(U)

    X = [x0]
    for i in range(n):
        X.append(F(X[-1], u(i)))
    X = np.array(X)

    fig, ax = plt.subplots()
    ax.plot(X[:, 0], X[:, 1], '.', color='blue')
    ax.plot(X_hat[:, 0], X_hat[:, 1], '+', color='black')
    ax.set_xlim(-2, 1)
    ax.set_ylim(-2, 1)

    return fig, ax


###
### Kalman filter classes, using the notation from Ch.7 in Haykin (2004)
###

class EKF:
    """
    Assume
      x_{k+1} = F(x_k, u_k) + v
      y_k = H(x_k) + n
    where v, n are Gaussian noise processes with mean 0
     and covariances R_v, R_n.    
    """
    def __init__(self, F, dF_dx, H, dH_dx, R_v, R_n, x_hat_0, P_0): 
        """
        x_hat_0 and P_0 are initial estimates of the state mean and covariance.

        dF_dx and dH_dx are the Jacobians of F and H with respect to x.
        """
        self.F = F
        self.dF_dx = dF_dx
        self.H = H
        self.dH_dx = dH_dx
        self.R_v = R_v
        self.R_n = R_n
        self.x_hat = x_hat_0
        self.P = P_0

    def time_update(self, u):
        x_hat_bar = self.F(self.x_hat, u)
        self.A = self.dF_dx(self.x_hat, u)
        A = self.A
        P_bar = A.dot(self.P).dot(A.T) + self.R_v
        return x_hat_bar, P_bar

    def measurement_update(self, y, P_bar, x_hat_bar):
        A = self.A
        C = self.dH_dx(x_hat_bar)
        K = P_bar.dot(C.T).dot(np.linalg.inv(C.dot(P_bar).dot(C.T) + self.R_n))
        x_hat = x_hat_bar + K.dot(y - H(x_hat_bar))
        P = (np.eye(P_bar.shape[0]) - K.dot(C)).dot(P_bar)
        self.x_hat, self.P = x_hat, P
        return x_hat, P


class UKF:
    """
    Assume
      x_{k+1} = F(x_k, u_k) + v
      y_k = H(x_k) + n
    where v, n are Gaussian noise processes with mean 0
     and covariances R_v, R_n.    
    """
    def weights_m(self):
        w = np.zeros(2*self.L + 1)
        w[0] = self.Lambda / (self.L + self.Lambda)
        for i in range(1, 2*self.L + 1):
            w[i] = 1 / (2*(self.L + self.Lambda))
        return w

    def weights_c(self):
        w = self.weights_m #()
        w[0] += 1 - self.alpha**2 + self.beta
        return w

    def __init__(self, F, H, R_v, R_n, x_hat_0, P_0, alpha=1):
        """
        x_hat_0 and P_0 are initial estimates of the state mean and covariance.
        alpha determines the spread of sigma points, and should be small...
        """
        self.F = F
        self.H = H
        self.R_v = R_v
        self.R_n = R_n
        self.x_hat = x_hat_0
        self.P = P_0
        self.L = len(x_hat_0)
        self.M = len(H(x_hat_0))
        self.alpha = alpha
        self.beta = 2
        kappa = 3 - self.L
        self.kappa = kappa
        self.Lambda = alpha**2 * (self.L + kappa) - self.L

        # NB: these don't change while we don't augment the sigma points
        self.weights_m = self.weights_m()
        self.weights_c = self.weights_c()

    def sigma_points(self, x, P):
        # nb: better to *augment* the sigma points at each k, not recompute
        #     but this should work fine
        sqrt = np.linalg.cholesky(P)
        sqrt *= np.sqrt(self.L + self.Lambda)
        sigma_points = np.zeros((2*self.L + 1, self.L))
        sigma_points[0, :] = x
        for i in range(1, self.L+1):
            sigma_points[i, :] = x + sqrt[:, i-1]
        for i in range(self.L):
            sigma_points[self.L+i+1, :] = x - sqrt[:, i]
        return sigma_points

    def time_update(self, u):
        sigma_points = self.sigma_points(self.x_hat, self.P)

        sigma_star = np.zeros((2*self.L + 1, self.L))
        for i in range(2*self.L + 1):
            sigma_star[i] = self.F(sigma_points[i], u)

        x_hat_bar = np.zeros(self.L)
        for i in range(2*self.L + 1):
            x_hat_bar += self.weights_m[i] * sigma_star[i]

        P_bar = np.array(self.R_v)
        for i in range(2*self.L + 1):
            P_bar += self.weights_c[i] * np.outer(
                sigma_star[i] - x_hat_bar,
                sigma_star[i] - x_hat_bar)

        sigma_points = self.sigma_points(x_hat_bar, P_bar)
        
        Y = np.zeros((2*self.L + 1, self.M))
        y_hat_bar = np.zeros(self.M)
        for i in range(2*self.L + 1):
            Y[i] = self.H(sigma_points[i])
        for i in range(2*self.L):
            y_hat_bar += self.weights_m[i] * Y[i]

        self.measurement_params = (sigma_points, Y, y_hat_bar)

        return x_hat_bar, P_bar

    def measurement_update(self, y, P_bar, x_hat_bar):
        sigma_points, Y, y_hat_bar = self.measurement_params

        P_yy = np.array(self.R_n)
        for i in range(2*self.L):
            P_yy += self.weights_c[i] * np.outer(
                Y[i] - y_hat_bar,
                Y[i] - y_hat_bar)

        P_xy = np.zeros((self.L, self.M))
        for i in range(2*self.L):
            P_xy += self.weights_c[i] * np.outer(
                sigma_points[i] - x_hat_bar,
                Y[i] - y_hat_bar)

        K = P_xy.dot(np.linalg.inv(P_yy))

        x_hat = x_hat_bar + K.dot(y - y_hat_bar)
        P = P_bar - K.dot(P_yy).dot(K.T)

        self.x_hat = x_hat
        self.P = P

        return x_hat, P


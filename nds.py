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
        x_hat_bar = filt.time_update(U[i])
        x_hat, P = filt.measurement_update(Y[i], x_hat_bar)
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
        self.P_bar = P_bar
        return x_hat_bar

    def measurement_update(self, y, x_hat_bar):
        P_bar = self.P_bar
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
    # def weights_m(self):
    #     w = np.zeros(2*self.L + 1)
    #     w[0] = self.Lambda / (self.L + self.Lambda)
    #     for i in range(1, 2*self.L + 1):
    #         w[i] = 1 / (2*(self.L + self.Lambda))
    #     return w

    # def weights_c(self):
    #     w = self.weights_m #()
    #     w[0] += 1 - self.alpha**2 + self.beta
    #     return w

    def weights_m(self):
        w = np.ones(2*self.L + 1)
        w[0] = 0.5
        w[1:] = (1 - w[0]) / (2*self.L)
        return w

    def weights_c(self):
        return self.weights_m

    def __init__(self, F, H, R_v, R_n, x_hat_0, P_0, alpha=1.611):
        """
        x_hat_0 and P_0 are initial estimates of the state mean and covariance.


        alpha determines the spread of sigma points, and should be ~1e-3 or so
         However: currently, small alpha (1.61 or less) makes the
                  covariance estimate explode, and it is not clear how to fix
                  this. There is a square-root implementation belo that should
                  not suffer this kind of problem, but it has bugs.
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
        print(np.sum(self.weights_m))

    def sigma_points(self, x, P):
        # nb: better to *augment* the sigma points at each k, not recompute
        #     but this should work fine
        sqrt = np.linalg.cholesky(P)
        sqrt *= np.sqrt(self.L / (1-self.weights_m[0])) #np.sqrt(self.L + self.Lambda)
        sigma_points = np.zeros((2*self.L + 1, self.L))
        sigma_points[0, :] = x
        for i in range(1, self.L+1):
            sigma_points[i, :] = x + sqrt[:, i-1]
        for i in range(self.L):
            sigma_points[self.L+i+1, :] = x - sqrt[:, i]
        return sigma_points

    def time_update(self, u):
        sigma_points = self.sigma_points(self.x_hat, self.P)
        #print("points", np.linalg.norm(sigma_points), np.linalg.norm(self.P))

        sigma_star = np.zeros((2*self.L + 1, self.L))
        for i in range(2*self.L + 1):
            sigma_star[i] = self.F(sigma_points[i], u)

        x_hat_bar = np.zeros(self.L)
        for i in range(2*self.L + 1):
            x_hat_bar += self.weights_m[i] * sigma_star[i]

        P_bar = np.array(self.R_v)
        #print(np.linalg.norm(P_bar), np.linalg.norm(sigma_star))
        for i in range(2*self.L + 1):
            P_bar += self.weights_c[i] * np.outer(
                sigma_star[i] - x_hat_bar,
                sigma_star[i] - x_hat_bar)
            #print(np.linalg.norm(P_bar))

        sigma_points = self.sigma_points(x_hat_bar, P_bar)
        
        Y = np.zeros((2*self.L + 1, self.M))
        y_hat_bar = np.zeros(self.M)
        for i in range(2*self.L + 1):
            Y[i] = self.H(sigma_points[i])
        for i in range(2*self.L + 1):
            y_hat_bar += self.weights_m[i] * Y[i]

        self.measurement_params = (sigma_points, P_bar, Y, y_hat_bar)

        return x_hat_bar

    def measurement_update(self, y, x_hat_bar):
        sigma_points, P_bar, Y, y_hat_bar = self.measurement_params

        P_yy = np.array(self.R_n)
        for i in range(2*self.L + 1):
            P_yy += self.weights_c[i] * np.outer(
                Y[i] - y_hat_bar,
                Y[i] - y_hat_bar)

        P_xy = np.zeros((self.L, self.M))
        for i in range(2*self.L + 1):
            P_xy += self.weights_c[i] * np.outer(
                sigma_points[i] - x_hat_bar,
                Y[i] - y_hat_bar)

        K = P_xy.dot(np.linalg.inv(P_yy))

        x_hat = x_hat_bar + K.dot(y - y_hat_bar)
        P = P_bar - K.dot(P_yy).dot(K.T)

        self.x_hat = x_hat
        self.P = P/2 + P.T/2 + np.eye(self.P.shape[0]) * 0.001

        return x_hat, self.P


class UKF_sqrt:
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
        self.S = np.linalg.cholesky(P_0)
        self.sqrt_R_v = np.linalg.cholesky(R_v)
        self.sqrt_R_n = np.linalg.cholesky(R_n)
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

    def sigma_points(self, x, S):
        # nb: better to *augment* the sigma points at each k, not recompute
        #     but this should work fine
        sqrt = np.sqrt(self.L + self.Lambda) * S
        sigma_points = np.zeros((2*self.L + 1, self.L))
        sigma_points[0, :] = x
        for i in range(1, self.L+1):
            sigma_points[i, :] = x + sqrt[:, i-1]
        for i in range(self.L):
            sigma_points[self.L+i+1, :] = x - sqrt[:, i]
        return sigma_points

    def time_update(self, u):
        sigma_points = self.sigma_points(self.x_hat, self.S)

        sigma_star = np.zeros((2*self.L + 1, self.L))
        for i in range(2*self.L + 1):
            sigma_star[i] = self.F(sigma_points[i], u)

        x_hat_bar = np.zeros(self.L)
        for i in range(2*self.L + 1):
            x_hat_bar += self.weights_m[i] * sigma_star[i]

        S_bar = np.array(sigma_star)
        for i in range(1, 2*self.L + 1):
            S_bar[i] -= x_hat_bar
        S_bar *= np.sqrt(self.weights_c[1])
        #print(S_bar.shape, self.sqrt_R_v.shape)
        S_bar = np.c_[S_bar.T, self.sqrt_R_v]
        q, r = np.linalg.qr(S_bar)
        S_bar = r + np.sqrt(self.weights_c[0]) * np.outer(
            sigma_star[0] - x_hat_bar,
            sigma_star[0] - x_hat_bar)

        sigma_points = self.sigma_points(x_hat_bar, S_bar)
        
        Y = np.zeros((2*self.L + 1, self.M))
        y_hat_bar = np.zeros(self.M)
        for i in range(2*self.L + 1):
            Y[i] = self.H(sigma_points[i])
        for i in range(2*self.L):
            y_hat_bar += self.weights_m[i] * Y[i]

        self.measurement_params = (sigma_points, S_bar, Y, y_hat_bar)

        return x_hat_bar

    def measurement_update(self, y, S_bar, x_hat_bar):
        sigma_points, S_bar, Y, y_hat_bar = self.measurement_params

        S_y = np.array(Y)
        for i in range(1, 2*self.L + 1):
            S_y[i] -= y_hat_bar
        S_y *= np.sqrt(self.weights_c[1])
        S_y = np.c_[S_y, self.sqrt_R_n]
        q, r = np.linalg.qr(S_y)
        S_y = r + np.sqrt(self.weights_c[0]) * np.outer(
            Y[0] - y_hat_bar,
            Y[0] - y_hat_bar)

        P_xy = np.zeros((self.L, self.M))
        for i in range(2*self.L+1):
            P_xy += self.weights_c[i] * np.outer(
                sigma_points[i] - x_hat_bar,
                Y[i] - y_hat_bar)

        # b / a -> linalg.solve(a.T, b.T)
        K = np.linalg.solve(S_y.T, np.linalg.solve(S_y, P_xy.T)).T

        x_hat = x_hat_bar + K.dot(y - y_hat_bar)
        self.x_hat = x_hat

        U = K.dot(S_y)
        S = np.array(S_bar)
        for i in range(U.shape[1]):
            S -= np.outer(U[:, i], U[:, i])
        self.S = S

        P = S.dot(S.T)
        self.P = P
        
        return x_hat, P


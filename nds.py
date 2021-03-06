#!/usr/bin/env python3

from __future__ import division              # For Python 2 users...

import numpy as np
from scipy import linalg as spla
import matplotlib.pyplot as plt

from filterpy.kalman import UnscentedKalmanFilter as UKF2
from filterpy.kalman import MerweScaledSigmaPoints

class RBF:
    def __init__(self, u_size, x_size, z_size):
        """
        self.c[i]
        self.S_det[i]
        self.S_inv[i]
        self.size
        self.I

        And guesses for:
        self.A, self.B, self.b, self.h[i]
        """
        self.I = 0
        self.size = z_size

        self.A = np.ones((z_size, x_size))
        self.B = np.ones((z_size, u_size))
        self.b = np.ones(z_size)

    def make_nonlinear(self, c, S):
        """
        c is a list of centres of the RBFs; S is a list of their covariances
        """
        self.I = len(c)
        self.h = [np.zeros(self.size)] * self.I  # Initial guess for RBF coeffs
        self.c = c
        self.S_det, self.S_inv = [], []
        for i in range(self.I):
            self.S_det.append(np.linalg.det(S[i]))
            self.S_inv.append(np.linalg.inv(S[i]))

    def rho(self, i, x):
        return pow(2*np.pi, -len(x)/2) * pow(self.S_det[i], -0.5) * np.exp(
            -0.5 * (x - self.c[i]).T.dot(self.S_inv[i]).dot(x - self.c[i]))

    def __call__(self, x, u):
        z = np.zeros(self.size)
        for i in range(self.I):
            z += self.h[i] * self.rho(i, x)
        z += self.A.dot(x) + self.B.dot(u) + self.b
        return z

    def update(self, u_data, x_mean_data, z_mean_data, x_z_covar_data):
        """
        u_data is a data matrix for u (rows are the different j);
        x_mean_data is the corresponding data matrix for the means of x;
        z_mean_data the same for the means of z; and
        x_z_covar_data is a list of corresponding covariance matrices for (x,z),
          such that the upper-left blocks are the x covariances, and
          the lower-right blocks are the z covariances.
        """
        J = len(u_data)

        u_size = len(u_data[0])
        x_size = len(x_mean_data[0])
        z_size = len(z_mean_data[0])
        size_max = max(max(u_size, x_size), z_size)

        #
        # Compute all the expectations needed for the linear part of the system
        #

        print("Computing linear expectations")

        sum_u = np.zeros(size_max)
        sum_mean_x = np.zeros(size_max)
        sum_mean_z = np.zeros(size_max)

        sum_u[:u_size] = np.sum(u_data, 0)
        sum_mean_x[:x_size] = np.sum(x_mean_data, 0)
        sum_mean_z[:z_size] = np.sum(z_mean_data, 0)

        sum_mean_x_xT = np.zeros((size_max, size_max))
        for j in range(J):
            sum_mean_x_xT[:x_size, :x_size] += (
                np.outer(x_mean_data[j], x_mean_data[j]) + 
                x_z_covar_data[j][:x_size, :x_size])

        sum_u_uT = np.zeros((size_max, size_max))
        for j in range(J):
            sum_u_uT[:u_size, :u_size] += np.outer(u_data[j], u_data[j])

        sum_mean_u_xT = np.zeros((size_max, size_max))
        for j in range(J):
            sum_mean_u_xT[:u_size, :x_size] += np.outer(u_data[j], x_mean_data[j])

        sum_mean_x_zT = np.zeros((size_max, size_max))
        for j in range(J):
            sum_mean_x_zT[:x_size, :z_size] += (
                np.outer(x_mean_data[j], z_mean_data[j]) + 
                x_z_covar_data[j][:x_size, x_size:])

        sum_mean_u_zT = np.zeros((size_max, size_max))
        for j in range(J):
            sum_mean_u_zT[:u_size, :z_size] += np.outer(u_data[j], z_mean_data[j])

        linear_row1 = np.c_[sum_mean_x_xT, sum_mean_u_xT, sum_mean_x]
        linear_row2 = np.c_[sum_mean_u_xT.T, sum_u_uT, sum_u]
        linear_row3 = np.r_[sum_mean_x, sum_u, J]
        linear_row3 = linear_row3.reshape((1, len(linear_row3)))

        linear_system = np.r_[linear_row1, linear_row2, linear_row3]

        linear_rhs = np.r_[
            sum_mean_x_zT,
            sum_mean_u_zT,
            sum_mean_z.reshape((1, size_max))
        ]

        #
        # Now compute all the expectations depending on the RBF kernels rho
        #

        print("Computing nonlinear expectations")

        x_z_precisions = []
        if self.I > 0:
            for j in range(J):
                x_z_precisions.append(np.linalg.inv(x_z_covar_data[j]))

        sum_mean_rhos = [0] * self.I
        sum_mean_u_rhos = [np.zeros(x_size)] * self.I
        sum_mean_x_rhos = [np.zeros(x_size)] * self.I
        sum_mean_z_rhos = [np.zeros(z_size)] * self.I
        sum_mean_rhos_rhos = np.zeros((self.I, self.I))

        for i in range(self.I):
            for j in range(J):
                sys.stdout.write("\r  %d, %d" % (i, j))
                C_ij = x_z_precisions[j].copy()
                C_ij[:x_size, :x_size] += self.S_inv[i]
                C_ij = np.linalg.inv(C_ij)
                delta_ij = self.c[i].dot(self.S_inv[i].dot(self.c[i]))
                mu_j = np.r_[x_mean_data[j], z_mean_data[j]]
                mu_ij = x_z_precisions[j].dot(mu_j)
                mu_ij += np.r_[self.S_inv[i].dot(self.c[i]), np.zeros(z_size)]
                mu_ij = C_ij.dot(mu_ij)
                delta_ij += mu_j.dot(x_z_precisions[j].dot(mu_j))
                delta_ij -= mu_ij.dot(np.linalg.inv(C_ij).dot(mu_ij))
                beta_ij = pow(2*np.pi, -x_size / 2) * pow(self.S_det[i], -0.5)
                beta_ij *= pow(np.linalg.det(x_z_covar_data[j]), -0.5)
                beta_ij *= pow(np.linalg.det(C_ij), 0.5)
                beta_ij *= np.exp(-0.5 * delta_ij)

                sum_mean_rhos[i] += beta_ij
                sum_mean_u_rhos[i] += beta_ij * u_data[j]
                sum_mean_x_rhos[i] += beta_ij * mu_ij[:x_size]
                sum_mean_z_rhos[i] += beta_ij * mu_ij[:z_size]

                for l in range(self.I):
                    C_ilj = x_z_precisions[j].copy()
                    C_ilj[:x_size, :x_size] += self.S_inv[i] + self.S_inv[l]
                    C_ilj = np.linalg.inv(C_ilj)
                    mu_ilj = x_z_precisions[j].dot(mu_j)
                    mu_ilj += np.r_[
                        self.S_inv[i].dot(self.c[i]), np.zeros(z_size)]
                    mu_ilj += np.r_[
                        self.S_inv[l].dot(self.c[l]), np.zeros(z_size)]
                    mu_ilj = C_ilj.dot(mu_ilj)
                    gamma_ilj = self.c[i].dot(self.S_inv[i].dot(self.c[i]))
                    gamma_ilj += self.c[l].dot(self.S_inv[l].dot(self.c[l]))
                    gamma_ilj += mu_j.dot(x_z_precisions[j].dot(mu_j))
                    gamma_ilj -= mu_ilj.dot(np.linalg.inv(C_ilj).dot(mu_ilj))

                    mean_rho_rho = pow(2*np.pi, -x_size)
                    mean_rho_rho *= pow(np.linalg.det(x_z_covar_data[j]), -0.5)
                    mean_rho_rho *= pow(self.S_det[i], -0.5)
                    mean_rho_rho *= pow(self.S_det[l], -0.5)
                    mean_rho_rho *= pow(np.linalg.det(C_ilj), 0.5)
                    mean_rho_rho *= np.exp(0.5 * gamma_ilj)
                    sum_mean_rhos_rhos[i, l] += mean_rho_rho

        #
        # And use the expectations to compute MLEs for theta and Q
        #

        print("\nComputing MLEs for theta and Q")

        system_size = self.I+x_size+u_size+1
        system = np.zeros((system_size, system_size))
        system[:self.I, :self.I] = sum_mean_rhos_rhos

        for i in range(self.I):
            system[i, self.I:self.I+x_size] = sum_mean_x_rhos[i]
            system[self.I:self.I+x_size, i] = sum_mean_x_rhos[i]

            system[i, self.I+x_size:self.I+x_size+u_size] = sum_mean_u_rhos[i]
            system[self.I+x_size:self.I+x_size+u_size, i] = sum_mean_u_rhos[i]

            system[i, -1] = sum_mean_rhos[i]
            system[-1, i] = sum_mean_rhos[i]

        system[self.I:, self.I:] = linear_system

        rhs = np.zeros((self.I+linear_rhs.shape[0], z_size))
        for i in range(self.I):
            rhs[i] = sum_mean_z_rhos[i]
        rhs[self.I:] = linear_rhs

        # Or perhaps compute the pseudo-inverse with npla.pinv?...
        theta = np.linalg.lstsq(system, rhs)[0]

        for i in range(self.I):
            self.h[i] = theta[i]

        self.A = theta[self.I:self.I+x_size, :z_size]
        self.B = theta[self.I+x_size:self.I+x_size+u_size, :z_size]
        self.b = theta[self.I+x_size+u_size, :z_size]

        # TODO: Q_hat (the MLE for the noise covariance in z)


def make_x_z_covar(x_covar, z_covar):
    return np.r_[
        np.c_[x_covar, np.zeros((x_covar.shape[0], z_covar.shape[1]))],
        np.c_[np.zeros((z_covar.shape[0], x_covar.shape[1])), z_covar]
    ]


def make_x_z_covar_data(x_covars, z_len, precision=10):
    data = []
    for R in x_covars:
        data.append(make_x_z_covar(R, np.eye(z_len)/precision))
    return data


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
    

def H_complicated(x):
    """
    UKF also works well with a complicated output function!
    """
    _ = x**2
    _[0] += np.sin(2*x[1]*x[0])
    _[1] += -3*x[0]**3 + np.log(np.abs(x[0]))
    return _


def H(x):
    return x


def dH_dx(x):
    return np.eye(len(x))


###
### Helper functions to generate a noisy timeseries, train a filter,
### and plot the results
###

def generate_timeseries(F=F, H=H, stop=2000, x0=np.array([-0.72, -0.64]),
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
    R = []
    for i in range(n):
        x_hat_bar = filt.time_update(U[i])
        x_hat, P = filt.measurement_update(Y[i], x_hat_bar)
        print(i, x_hat, P)
        X.append(x_hat)
        R.append(P)
    return np.array(X), R


def train_filter2(filt, U, Y):
    """
    filt is your filter class, U the input vector, and Y the output vector.
    """
    n = len(U)
    X, R = [], []
    for i in range(n):
        print(i)
        filt.predict(fx_args=(U[i],))
        filt.update(Y[i])
        X.append(filt.x.copy())
        R.append(filt.P.copy())
    X = np.array(X)
    X, R, K = filt.rts_smoother(X, R, fx_args=(U[i],))
    return X, R


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


def test_filter(F, H, dF_dx, dH_dx, x0=np.array([-0.72, -0.64])):
    U, Y, R_v, R_n = generate_timeseries(F, H)
    #filt = UKF(F, H, R_v, R_n, x0, np.eye(2)*0.0001)
    #X, R = train_filter(filt, U, Y)
    def fx(x, dt, u):
        return F(x, u)
    filt = UKF2(len(x0), len(Y[0]), 1, H, fx, MerweScaledSigmaPoints(2, 1e-3, 2, 1))
    filt.x = x0
    X, R = train_filter2(filt, U, Y)
    plot_states(F, U, X)
    plt.show()


def test_filter_rbf(F, H, x0=np.array([-0.72, -0.64])):
    U, Y, R_v, R_n = generate_timeseries(F, H)

    c_x = [-1.25, -0.75, -0.25, 0.25]
    c_y = [-1.25, -0.75, -0.25, 0.25]
    c, S = [], []
    for x in c_x:
        for y in c_y:
            c.append(np.array([x, y]))
            S.append(np.eye(2)*0.5)

    rbf = RBF(len(U[0]), 2, len(Y[0]))
    rbf.make_nonlinear(c, S)

    while True:
        # filt = UKF(rbf, H, R_v, R_n, x0, np.eye(2)*0.0001)
        # X, R = train_filter(filt, U, Y)
        def fx(x, dt, u):
            return rbf(x, u)
        filt = UKF2(len(x0), len(Y[0]), 1, H, fx, MerweScaledSigmaPoints(2, 1e-3, 2, 1))
        filt.x = x0
        X, R = train_filter2(filt, U, Y)
        plot_states(F, U, X)
        plt.show()
        covars = make_x_z_covar_data(R, 2)
        rbf.update(U, X, Y, covars)

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
        w[0] = 0.1
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
        #print(np.sum(self.weights_m))

    def sigma_points(self, x, P):
        # nb: better to *augment* the sigma points at each k, not recompute
        #     but this should work fine
        sqrt, err = spla.sqrtm(P, disp=False) # spla.cholesky(P)
        print("!!!", err)
        if err > 1000:
            raise Exception("Error too large")
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

        print("!!!!!!!!!!!!")

        P_yy = np.array(self.R_n)
        for i in range(2*self.L + 1):
            P_yy += self.weights_c[i] * np.outer(
                Y[i] - y_hat_bar,
                Y[i] - y_hat_bar)
        print(P_yy)

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

        print("!!!!!!!!!!!!")

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
        self.S = spla.sqrtm(P_0) #np.linalg.cholesky(P_0)
        self.sqrt_R_v = spla.sqrtm(R_v) #np.linalg.cholesky(R_v)
        self.sqrt_R_n = spla.sqrtm(R_n) #np.linalg.cholesky(R_n)
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


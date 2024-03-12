#!/usr/bin/env python

import os,sys,glob,gzip
import numpy as np
import sklearn.preprocessing as preprocessing
import sklearn.decomposition as decomposition


EPSILON = np.finfo(np.float32).eps


class MVNMF:
    # A single run of mvNMF
    def __init__(
        self, X, n_components,
        init = 'nndsvd',  # init = {‘random’, ‘nndsvd’, ‘nndsvda’, ‘nndsvdar’}
        lambda_tilde = 1e-5,
        delta = 1.0,
        gamma = 1.0,
        max_iter = 200,
        min_iter = 100,
        tol = 1e-4,
        conv_test_freq = 10,
    ):
        if (type(X) != np.ndarray) or (not np.issubdtype(X.dtype, np.floating)):
            X = np.array(X).astype(float)
        self.X = X
        self.n_components = int(n_components)
        self.init = init
        self.lambda_tilde = lambda_tilde
        self.delta = delta
        self.gamma = gamma
        self.max_iter = max_iter
        self.min_iter = min_iter
        self.tol = tol
        self.conv_test_freq = conv_test_freq
        self.W = None
        self.H = None
    
    def initialize(self):
        # init = {‘random’, ‘nndsvd’, ‘nndsvda’, ‘nndsvdar’, ‘custom’}
        # https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html
        W, H = decomposition._nmf._initialize_nmf(self.X, self.n_components, init = self.init)
        W = preprocessing.normalize(W, norm = 'l1', axis = 0)
        self.W = W
        self.H = H
    
    @staticmethod
    def normalize_WH(W, H):
        norm_factor = np.sum(W, axis = 0)
        W = W / norm_factor
        H = H * norm_factor.reshape(-1, 1)
        return W, H
    
    def kl_divergence_X_WH(self, W, H):
        A_data = self.X.ravel()
        B_data = (W @ H).ravel()
        indices = A_data > 0
        A_data = A_data[indices]
        B_data_remaining = B_data[~indices]
        B_data = B_data[indices]
        res = np.sum(A_data * np.log(A_data / B_data) - A_data + B_data) + np.sum(B_data_remaining)
        return res
    
    def volume_log_det(self, W, H):
        K = W.shape[1]
        return np.log10(np.linalg.det(W.T @ W + self.delta * np.eye(K)))
    
    def loss_mvnmf(self, W, H):
        reconstruction_error = self.kl_divergence_X_WH(W, H)
        volume = self.volume_log_det(W, H)
        loss = reconstruction_error + self.Lambda * volume
        return loss, reconstruction_error, volume
    
    def solve(self):
        # reimplementation of https://github.com/parklab/MuSiCal/blob/main/musical/mvnmf.py#L39
        X = self.X
        n_features, n_samples = self.X.shape
        M = n_features
        T = n_samples
        K = self.n_components
        # normalize W, H
        W, H = self.normalize_WH(self.W, self.H)
        # clip small values, EPSILON = np.finfo(np.float32).eps
        W = W.clip(EPSILON)
        H = H.clip(EPSILON)
        # calculate Lambda from labmda_tilde
        reconstruction_error = self.kl_divergence_X_WH(W, H)
        volume = self.volume_log_det(W, H)
        self.Lambda = self.lambda_tilde * reconstruction_error / np.abs(volume)
        loss = reconstruction_error + self.Lambda * volume
        # constants
        ones = np.ones((M, T))
        # baseline of convergence test
        conv_test_baseline = loss
        # main
        gamma = self.gamma
        losses = [loss]
        reconstruction_errors = [reconstruction_error]
        volumes = [volume]
        line_search_steps = []
        gammas = [gamma]
        converged = False
        for n_iter in range(1, self.max_iter + 1):
            # Update H according to 2001 Lee
            H = H * ((W.T @ (X / (W @ H))) / (W.T @ ones))
            H = H.clip(EPSILON)
            # Update W
            Y = np.linalg.inv(W.T @ W + self.delta * np.eye(K))
            Y_plus = np.maximum(Y, 0)
            Y_minus = np.maximum(-Y, 0)
            JHT = ones @ H.T
            LWYm = self.Lambda * (W @ Y_minus)
            LWY = self.Lambda * (W @ (Y_plus + Y_minus))
            numerator = ((JHT - 4 * LWYm) ** 2 + 8 * LWY * ((X / (W @ H)) @ H.T)) ** 0.5 - JHT + 4 * LWYm
            denominator = 4 * LWY
            Wup = W * (numerator / denominator)
            Wup = Wup.clip(EPSILON)
            # Backtracking line search for W
            W_new = (1.0 - gamma) * W + gamma * Wup
            W_new, H_new = self.normalize_WH(W_new, H)
            W_new = W_new.clip(EPSILON)
            H_new = H_new.clip(EPSILON)
            loss, reconstruction_error, volume = self.loss_mvnmf(W_new, H_new)
            line_search_step = 0
            while (loss > losses[-1]) and (gamma > 1e-16):
                gamma = gamma * 0.8
                W_new = (1.0 - gamma) * W + gamma * Wup
                W_new, H_new = self.normalize_WH(W_new, H)
                W_new = W_new.clip(EPSILON)
                H_new = H_new.clip(EPSILON)
                loss, reconstruction_error, volume = self.loss_mvnmf(W_new, H_new)
                line_search_step += 1
            W = W_new
            H = H_new
            line_search_steps.append(line_search_step)
            # update gamma
            gamma = min(gamma * 2.0, self.gamma)
            gammas.append(gamma)
            # loss
            loss, reconstruction_error, volume = self.loss_mvnmf(W, H)
            losses.append(loss)
            reconstruction_errors.append(reconstruction_error)
            volumes.append(volume)
            # convergence test
            if n_iter >= self.min_iter and n_iter % self.conv_test_freq == 0:
                relative_loss_change = (losses[-2] - loss) / conv_test_baseline
                if (loss <= losses[-2]) and (relative_loss_change <= self.tol):
                    converged = True
                else:
                    converged = False
                print('Iter=%d, Loss=%.3g, Loss_prev=%.3g, Relative_loss_change=%.3g' % (n_iter, loss, losses[-2], relative_loss_change))
            if converged and n_iter >= self.min_iter:
                break
        self.losses = np.array(losses)
        self.reconstruction_errors = np.array(reconstruction_errors)
        self.volumes = np.array(volumes)
        self.line_search_steps = np.array(line_search_steps)
        self.gammas = np.array(gammas)
        self.n_iter = n_iter
        self.W = W
        self.H = H
    
    def fit(self):
        self.initialize()
        self.solve()


if __name__ == '__main__':
    # test-1
    W = np.array([[1,2], [3,4], [3,2], [5,2]])
    H = np.array([[0, 3, 4, 0], [2, 1, 0, 2]])
    X = W @ H
    print(X)

    nmf = MVNMF(X, 2)
    nmf.fit()

    np.set_printoptions(precision = 2, floatmode = 'maxprec')
    print(nmf.W)
    print(nmf.H)
    print(nmf.W @ nmf.H)
    
    
    # test-2
    f = 'MuSiCal/examples/data/simulated_example.Skin.Melanoma.X.csv'
    import pandas as pd
    df = pd.read_csv(f, index_col = 0)
    
    nmf = MVNMF(df.to_numpy(), 15)
    nmf.fit()
    
    f = 'MuSiCal/examples/data/simulated_example.Skin.Melanoma.W_true.csv'
    Wtrue = pd.read_csv(f, index_col = 0)
    from scipy.spatial.distance import cdist
    corr = (1 - cdist(nmf.W.T, Wtrue.to_numpy().T, metric='correlation'))
    corr = pd.DataFrame(corr, index=[ 'NMF%d' % i for i in range(15) ], columns=Wtrue.columns)
    
    # plot corr
    import seaborn as sns
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize = (3.5, 3.5))  # (x, y)
    ax = fig.add_subplot(111)
    sns.clustermap(
        data = corr, row_cluster = False, col_cluster = False,
        vmin = -1, vmax = 1,
        cmap = 'RdYlBu_r',
    )
    plt.suptitle('')
    plt.savefig('plot_corr.pdf')
    plt.close()


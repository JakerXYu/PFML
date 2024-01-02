import numpy as np
import pandas as pd
from itertools import groupby
from sklearn.neighbors import KernelDensity
from sklearn.metrics import confusion_matrix

def measures_from_Yhat(Y, Z, Yhat=None, threshold=0.5):
    assert isinstance(Y, np.ndarray)
    assert isinstance(Z, np.ndarray)
    assert Yhat is not None
    assert isinstance(Yhat, np.ndarray)

    if Yhat is not None:
        Ytilde = (Yhat >= threshold).astype(np.float32)
    assert Ytilde.shape == Y.shape and Y.shape == Z.shape

    # Accuracy
    acc = (Ytilde == Y).astype(np.float32).mean()
    # DDP
    DDP = abs(np.mean(Ytilde[Z != 1]) - np.mean(Ytilde[Z == 1]))
    # DP
    DP = abs(max([np.mean(Ytilde[Z != 1]),np.mean(Ytilde[Z == 1])]) - np.mean(Ytilde))
    # EO
    Y_Z0, Y_Z1 = Y[Z != 1], Y[Z == 1]
    Y1_Z0 = Y_Z0[Y_Z0 == 1]
    Y0_Z0 = Y_Z0[Y_Z0 != 1]
    Y1_Z1 = Y_Z1[Y_Z1 == 1]
    Y0_Z1 = Y_Z1[Y_Z1 != 1]

    FPR, FNR = {}, {}
    FPR[0] = np.sum(Ytilde[np.logical_and(Z != 1, Y != 1)]) / len(Y0_Z0)
    FPR[1] = np.sum(Ytilde[np.logical_and(Z == 1, Y != 1)]) / len(Y0_Z1)

    FNR[0] = np.sum(1 - Ytilde[np.logical_and(Z != 1, Y == 1)]) / len(Y1_Z0)
    FNR[1] = np.sum(1 - Ytilde[np.logical_and(Z == 1, Y == 1)]) / len(Y1_Z1)

    TPR_diff = abs((1 - FNR[0]) - (1 - FNR[1]))
    FPR_diff = abs(FPR[0] - FPR[1])
    DEO = TPR_diff + FPR_diff

    # DD
    Yh_Z0, Yh_Z1 = Yhat[Z != 1], Yhat[Z == 1]
    Yt_Z0, Yt_Z1 = Ytilde[Z != 1], Ytilde[Z == 1]
    
    step = 0.01
    probs1 = np.zeros(100)
    probs2 = np.zeros(100)
    for k, g in groupby(sorted(Yh_Z0), key=lambda x: x // step):
        # print('{}-{}: {}'.format(k * step, (k + 1) * step, len(list(g))))
        probs1[int(k)] = len(list(g))
    prob1 = probs1 / len(Yh_Z0)
    for k, g in groupby(sorted(Yh_Z1), key=lambda x: x // step):
        # print('{}-{}: {}'.format(k * step, (k + 1) * step, len(list(g))))
        probs2[int(k)] = len(list(g))
    prob2 = probs2 / len(Yh_Z1)
    DD = sum(abs(prob1 - prob2))

    kde = KernelDensity(bandwidth=0.02, kernel='gaussian')
    x_d = np.linspace(0, 1.0, 100)
    kde.fit(Yh_Z0[:, None])
    # score_samples returns the log of the probability density
    logprob1 = kde.score_samples(x_d[:, None])
    kde.fit(Yh_Z1[:, None])
    logprob2 = kde.score_samples(x_d[:, None])
    DD2 = sum(abs(np.exp(logprob1) - np.exp(logprob2))) / 100
    RD = abs(sum(Yt_Z0) / len(Yt_Z0) - sum(Yt_Z1) / len(Yt_Z1))
    data = [float('{:.5f}'.format(acc)), float('{:.5f}'.format(DDP)),float('{:.5f}'.format(DP)), float('{:.5f}'.format(DEO)),
            float('{:.5f}'.format(DD)), float('{:.5f}'.format(DD2)), float('{:.5f}'.format(RD)), ]
    columns = ['acc', 'DDP','DP', 'DEO', 'DD', 'DD2', 'RD']
    return pd.DataFrame([data], columns=columns)

def dd_loss(Yhat, Z):
    # calculate DD
    measures = measures_from_Yhat(Y, Z, Yhat)
    return measures['DD']


def optimize_data(initial_data, Z, learning_rate=0.01, num_iterations=1000):
    data = initial_data.copy()
    for _ in range(num_iterations):
        # calculate the gradient
        gradient = compute_gradient(data, Z)
        # update the data
        data -= learning_rate * gradient
    return data

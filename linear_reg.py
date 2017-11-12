import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df_orig = pd.read_csv('winequality-white.csv', sep=';')

# y = b0 + b1 * x1 + b2 * x2
# Y = B'X
# Cost = (Y - (X * B.T)) ^ 2

#Normalise - subtract each column by mean of its column and divide by column variance


def normalise_data(df):
    norm_df = (df - df.min()) / (df.max() - df.min())
    # norm_df = (df - df.mean()) / df.var()
    return norm_df


def optimise(X, Y, B, alpha=0.1, iterations=100):
    cost_history = []

    for i in range(iterations):
        H = np.dot(X, B.T)
        loss = H - Y
        cost = np.sum(np.square(loss)) / (2 * len(Y))
        # cost = np.sum(np.square(loss)) / (10000000)
        cost_history.append(cost)
        print("Iteration %d | Cost: %f" % (i, cost))
        print("Iteration %d | sum of losses: %f" % (i, np.sum(loss)))

        # print(loss)
        #gradient
        gradient = np.dot(X.T, loss) / len(Y)

        # update
        for k in range(X.shape[1] - 1):
            # print(k)
            B[:, k] = B[:,k] - alpha * gradient[k, :]

    return B, cost_history


print(normalise_data(df_orig).head())

df = normalise_data(df_orig)
# Since last column refers to quality, it is the variable that needs to be predicted.
n_col = df.shape[1]
X = df.iloc[:, 0:n_col-1]
X = np.matrix(X.values)
X = np.append(np.ones((X.shape[0], 1), dtype=type(X[0, 0])), X, 1)

Xtrain = X[0:4000,:]
Xtest = X[4001:,:]
y = df.iloc[:, n_col-1:n_col]
y = np.matrix(y.values)
ytrain = y[0:4000,:]
ytest = y[4001:,:]
B = np.random.rand(1, n_col)

#
print(Xtrain.shape)
print(Xtest.shape)
print(ytrain.shape)
print(ytest.shape)
print(B.shape)

params, cost_hist_sc = optimise(Xtrain, ytrain, B, 0.1, 10000)
y_pred_s = np.dot(Xtest, B.T)

regr = LinearRegression()
regr.fit(Xtrain, ytrain)
y_pred_l = regr.predict(Xtest)

print('Coefficients scratch : \n', B)
print('Coefficients linreg : \n', regr.coef_)

print("Mean squared error: %.2f" % mean_squared_error(ytest, y_pred_s))
print("Mean squared error: %.2f" % mean_squared_error(ytest, y_pred_l))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(ytest, y_pred_s))
print('Variance score: %.2f' % r2_score(ytest, y_pred_l))











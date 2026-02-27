# from sklearn.linear_model import LinearRegression

# def R2_score(X, y):

#     """
#     Perform univariate linear regression and return R2 score
#     """

#     X = X.reshape(-1, 1)

#     # Linear Regression
#     LsLin = LinearRegression().fit(X, y)

#     # Coefficient of determination
#     return LsLin.score(X, y)
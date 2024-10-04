import numpy as np

def test():
    print("Hello, World!")

# Manually doing dot product
def predict(feature, parameter, intercept):
    prediction = 0
    n = feature.shape[0]
    for i in range(n):
        prediction += feature[i] * parameter[i]
    prediction += intercept
    return prediction

# Dot product using numpy
def predict_numpy(feature, parameter, intercept):
    prediction = np.dot(feature, parameter) + intercept
    return prediction

# Mean Squred Error calculation
def compute_cost(x_test, y_test, w, b):
    n = x_test.shape[0]
    cost = 0.0
    for i in range(n):
        f_wb_i = np.dot(x_test[i], y_test[i]) + b
        cost = (f_wb_i - y_test[i]) ** 2
    cost = cost / (2 * n)
    return cost

# Gradien Descent calculation
# Baiscally it adjusts the parameter and intercept to get the best results from model
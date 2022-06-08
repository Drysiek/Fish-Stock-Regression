from fish_model import get_data, func, CustomModelWrapper
from scipy.optimize import curve_fit
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error


def linear_model(x_train, x_test, y_train, y_test):
    model_lin = LinearRegression()
    model_lin.fit(x_train, y_train)
    y_predicted_lin = model_lin.predict(x_test)
    print("\nMean absolute percentage error from linearRegression:")
    print(mean_absolute_percentage_error(y_test, y_predicted_lin))
    print("Mean squared error from linearRegression:")
    print(mean_squared_error(y_test, y_predicted_lin))
    return y_predicted_lin


def SVR_model(x_train, x_test, y_train, y_test):
    model_svr = SVR()
    model_svr.fit(x_train, y_train)
    y_predicted_svr = model_svr.predict(x_test)
    print("\nMean absolute percentage error from SVR")
    print(mean_absolute_percentage_error(y_test, y_predicted_svr))
    print("Mean squared error from SVR")
    print(mean_squared_error(y_test, y_predicted_svr))
    return y_predicted_svr


def custom_model(x_train, x_test, y_train, y_test):
    parameters, _ = curve_fit(func, xdata=x_train, ydata=y_train.values.ravel(), p0=np.ones(len(x_train.columns) + 1))
    model_custom = CustomModelWrapper(func, parameters)
    y_predicted_custom = model_custom.predict(x_test)
    print("\nMean absolute percentage error from curve_fit")
    print(mean_absolute_percentage_error(y_test, y_predicted_custom))
    print("Mean squared error from curve_fit")
    print(mean_squared_error(y_test, y_predicted_custom))
    return y_predicted_custom


if __name__ == '__main__':
    x, y = get_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.20, random_state=40)

    # Linear model
    y_predicted_l = linear_model(x_train, x_test, y_train, y_test)

    # Model SVR(Support Vector Regression)
    y_predicted_s = SVR_model(x_train, x_test, y_train, y_test)

    # Custom model
    y_predicted_c = custom_model(x_train, x_test, y_train, y_test)

    reach = range(0, int(len(y_test)))

    plt.title('Regressions and actual values')
    plt.xlabel("Test number")
    plt.ylabel("Number of fish")

    plt.scatter(reach, y_test)
    plt.plot(reach, y_predicted_l, 'b', label='linear regression')
    plt.plot(reach, y_predicted_s, 'c', label='SVR model regression')
    plt.plot(reach, y_predicted_c, 'r', label='custom regression')
    plt.show()

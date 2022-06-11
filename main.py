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
    print('{}%'.format(mean_absolute_percentage_error(y_test, y_predicted_lin)))
    print("Mean squared error from linearRegression:")
    print(mean_squared_error(y_test, y_predicted_lin))
    return y_predicted_lin


def SVR_model(x_train, x_test, y_train, y_test):
    model_svr = SVR()
    model_svr.fit(x_train, y_train)
    y_predicted_svr = model_svr.predict(x_test)
    print("\nMean absolute percentage error from SVR")
    print('{}%'.format(mean_absolute_percentage_error(y_test, y_predicted_svr)))
    print("Mean squared error from SVR")
    print(mean_squared_error(y_test, y_predicted_svr))
    return y_predicted_svr


def custom_model(x_train, x_test, y_train, y_test):
    parameters, _ = curve_fit(func, xdata=x_train, ydata=y_train.values.ravel(), p0=np.ones(len(x_train.columns) + 1))
    model_custom = CustomModelWrapper(func, parameters)
    y_predicted_custom = model_custom.predict(x_test)
    print("\nMean absolute percentage error from curve_fit")
    print('{}%'.format(mean_absolute_percentage_error(y_test, y_predicted_custom)))
    print("Mean squared error from curve_fit")
    print(mean_squared_error(y_test, y_predicted_custom))
    return y_predicted_custom


if __name__ == '__main__':
    new_file = input('Do you want to draw ner rows to calculate regression?\t(y/n)')
    while new_file != 'y' and new_file != 'n':
        new_file = input('Do you want to draw ner rows to calculate regression?\t(y/n)')
    if new_file == 'y':
        new_file = True
    else:
        new_file = False

    how_many = 100
    if new_file:
        while int(how_many) < 10 or int(how_many) > 5000 or not how_many.isnumeric():
            how_many = input('How many rows do you want to draw?\nType a number between 10 and 5000')

    x, y = get_data(new_file, int(how_many))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.20, random_state=40)

    reach = range(0, int(len(y_test)))
    plt.title('Custom model regression')
    plt.xlabel("Test number")
    plt.ylabel("Number of fish")
    plt.scatter(reach, y_test, color='g')

    # Linear model
    y_predicted_l = linear_model(x_train, x_test, y_train, y_test)
    plt.scatter(reach, y_predicted_l, color='b')
    # Model SVR(Support Vector Regression)
    y_predicted_s = SVR_model(x_train, x_test, y_train, y_test)
    plt.scatter(reach, y_predicted_s, color='c', label='SVR')
    # Custom model
    try:
        y_predicted_c = custom_model(x_train, x_test, y_train, y_test)
        plt.scatter(reach, y_predicted_c, color='r')
    except TypeError:
        print('\nToo few rows for custom regression, try something more than 550 rows')

    plt.show()

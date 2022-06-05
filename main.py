from fish_model import get_data, func, CustomModelWrapper
from scipy.optimize import curve_fit
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error


if __name__ == '__main__':
    X, Y = get_data()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.20, random_state=40)

    # Model liniowy
    model_lin = LinearRegression()
    model_lin.fit(X_train, Y_train)
    Y_predicted_lin = model_lin.predict(X_test)
    print("\nMean absolute percentage error from linearRegression:")
    print(mean_absolute_percentage_error(Y_test, Y_predicted_lin))
    print("Mean squared error from linearRegression:")
    print(mean_squared_error(Y_test, Y_predicted_lin))

    # Model SVR(Support Vector Regression)
    model_svr = SVR()
    model_svr.fit(X_train, Y_train)
    Y_predicted_svr = model_svr.predict(X_test)
    print("\nBMean absolute percentage error from SVR")
    print(mean_absolute_percentage_error(Y_test, Y_predicted_svr))
    print("Mean squared error from SVR")
    print(mean_squared_error(Y_test, Y_predicted_svr))

    parameters, _ = curve_fit(func, xdata=X_train, ydata=Y_train.values.ravel())
    model_custom = CustomModelWrapper(func, parameters)
    Y_predicted_custom = model_custom.predict(X_test)
    print('\n')
    # print(parameters)
    print("Mean absolute percentage error from curvefit")
    print(mean_absolute_percentage_error(Y_test, Y_predicted_custom))
    print("Mean squared error from curvefit")
    print(mean_squared_error(Y_test, Y_predicted_custom))

    zasieg = int(len(Y_test))

    sns.scatterplot(x=range(0, zasieg), y=Y_test, color="white", edgecolor="black")
    sns.lineplot(x=range(0, zasieg), y=Y_predicted_lin, color="red")
    sns.lineplot(x=range(0, zasieg), y=Y_predicted_svr, color="green")
    sns.lineplot(x=range(0, zasieg), y=Y_predicted_custom, color="orange")
    plt.show()

    # sns.scatterplot(x=range(0, zaiseg), y=Y_test, color="white", edgecolor="black")

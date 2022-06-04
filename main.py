from fish_model import get_data1, get_data2, f_map
from scipy.optimize import curve_fit
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error


if __name__ == '__main__':

    # regresja własna
    X, Y = get_data2()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.20, random_state=40)


    # Model liniowy
    model_lin = LinearRegression()
    model_lin.fit(X_train, Y_train)
    Y_predicted_lin = model_lin.predict(X_test)
    print("\nBlad procentowy z linearRegression")
    print(mean_absolute_percentage_error(Y_test, Y_predicted_lin))
    print("Blad kwadratowy z linearRegression")
    print(mean_squared_error(Y_test, Y_predicted_lin))

    # Model SVR(Support Vector Regression)
    model_svr = SVR()
    model_svr.fit(X_train, Y_train)
    Y_predicted_svr = model_svr.predict(X_test)
    print("\nBlad procentowy z SVR")
    print(mean_absolute_percentage_error(Y_test, Y_predicted_svr))
    print("Blad kwadratowy z SVR")
    print(mean_squared_error(Y_test, Y_predicted_svr))


    parameters, _ = curve_fit(f_map, xdata=X_train, ydata=Y_train.values.ravel())

    Y_predicted = f_map(X_test, *parameters)
    print('\n')
    # print(parameters)
    print("Blad procentowy z curvefit")
    print(mean_absolute_percentage_error(Y_test, Y_predicted))
    print("Blad kwadratowy z curvefit")
    print(mean_squared_error(Y_test, Y_predicted))

    zasieg = int(len(Y_test))

    sns.scatterplot(x=range(0, zasieg), y=Y_test, color="white", edgecolor="black")
    sns.lineplot(x=range(0, zasieg), y=Y_predicted_lin, color="red")
    sns.lineplot(x=range(0, zasieg), y=Y_predicted_svr, color="green")
    sns.lineplot(x=range(0, zasieg), y=Y_predicted, color="orange")
    plt.show()

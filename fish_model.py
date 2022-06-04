import pandas as pd
import os
import numpy as np
from scipy.optimize import curve_fit


def get_data1():
    X = pd.read_csv('f4.csv', sep=',')
    Y = X['Number']
    X = X.drop(['Number'], axis=1)

    for i in range(len(X)):
        X['County'][i] = hash(X['County'][i])/1000
        X['Waterbody'][i] = hash(X['Waterbody'][i])/1000
        X['Town'][i] = hash(X['Town'][i])/1000
        X['Month'][i] = hash(X['Month'][i])/1000
        X['Species'][i] = hash(X['Species'][i])/1000

    return X, Y


def get_data2():
    if not os.path.isfile('f6.csv'):
        print('Downlaoding file from: ', 'Fish Stocking.csv')
        data_frame = pd.read_csv('Fish Stocking.csv', sep=',')
        data_frame = data_frame[data_frame.notnull()]
        data_frame.to_csv('f6.csv', index=False)
        print('Downloaded')
    else:
        print(f'File "f6.csv" already exists')

    X = pd.read_csv('f4.csv', sep=',')
    Y = X['Number']
    X = X.drop(['Number'], axis=1)


    import people_also_ask
    # wa = list(set(X['Waterbody']))
    # for i in wa:
    #     try:
    #         txt= (people_also_ask.get_simple_answer("What is surface area of "+i))
    #         wa[i] = [int(s) for s in txt.split() if s.isdigit()][0]
    #     except Exception:
    #         wa[i] = 0

    Co = list(set(X['County']))
    Wa = list(set(X['Waterbody']))
    To = list(set(X['Town']))
    Mo = list(set(X['Month']))
    Sp = list(set(X['Species']))

    X['County'] = X["County"].replace(tuple(Co), tuple(range(len(Co))))
    X["Waterbody"] = X["Waterbody"].replace(tuple(Wa), tuple(range(len(Wa))))
    X["Town"] = X["Town"].replace(tuple(To), tuple(range(len(To))))
    X["Month"] = X["Month"].replace(tuple(Mo), tuple(range(len(Mo))))
    X["Species"] = X["Species"].replace(tuple(Sp), tuple(range(len(Sp))))

    return X, Y


iterator = 0


def f_map(x, a, b, c, d, e, f, g, h):
    result = h
    p = [a, b, c, d, e, f, g]
    for i in range(7):
        result += x[x.columns[i]] * p[i]
    return result


def f_map2(x, a, b, c, d, e, f, g, h):
    global iterator
    result = h
    p = [a, b, c, d, e, f, g]
    for i in range(7):
        if iterator != i:
            result += x[x.columns[i]] * p[i]
    return result


if __name__ == '__main__':
    df = pd.read_csv('f4.csv', sep=',')

    X = df[['Year', 'County', 'Waterbody', 'Town', 'Month', 'Species', 'Size']]
    # X = my_dummie(X)
    for i in range(len(X)):
        X['County'][i]=hash(X['County'][i])
        X['Waterbody'][i] = hash(X['Waterbody'][i])
        X['Town'][i] = hash(X['Town'][i])
        X['Month'][i] = hash(X['Month'][i])
        X['Species'][i] = hash(X['Species'][i])
    print(X)
    Y = df['Number']

    parameters, _ = curve_fit(f_map, xdata=X, ydata=Y.values.ravel())

    Y_predicted = f_map(X, *parameters)

    print("Blad kwadratowy z curvefit przy wykorzystaniu wszystkich zmiennych")
    print(np.sum(np.square(Y - Y_predicted))/Y.size)

    # 4) Policz jak usunięcie jednej z kolumn wpłynie na wartość błędu średniokwadratowego dla modelu
    for i in range(10):
        iterator = i
        parameters, _ = curve_fit(f_map2, xdata=X, ydata=Y.values.ravel())

        Y_predicted = f_map2(X, *parameters)

        print("Blad kwadratowy z curvefit przy wykorzystaniu wszystkich zmiennych oprócz {}".format(X.columns[i]))
        print(np.sum(np.square(Y - Y_predicted))/Y.size)

    # Y10=list(Y.head(20))
    # Y_pred10 = list(Y_pred.head(20))
    # sns.scatterplot(x=range(0, 20), y=Y10)
    # sns.lineplot(x=range(0, 20), y=Y_pred10)
    # plt.show()

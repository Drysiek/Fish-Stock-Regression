import pandas as pd
import os


def get_data():
    if not os.path.isfile('random_fish_stocking_rows.csv'):
        print('Downlaoding data from: ', 'Fish Stocking.csv')
        data_frame = pd.read_csv('Fish Stocking.csv', sep=',')
        data_frame = data_frame[data_frame['Year'].notnull()]
        data_frame = data_frame[data_frame['County'].notnull()]
        data_frame = data_frame[data_frame['Waterbody'].notnull()]
        data_frame = data_frame[data_frame['Town'].notnull()]
        data_frame = data_frame[data_frame['Month'].notnull()]
        data_frame = data_frame[data_frame['Number'].notnull()]
        data_frame = data_frame[data_frame['Species'].notnull()]
        data_frame = data_frame[data_frame['Size'].notnull()]
        data_frame = data_frame.sample(n=1000)
        data_frame.to_csv('random_fish_stocking_rows.csv', index=False)
        print('Downloaded')
    else:
        print(f'File "random_fish_stocking_rows.csv" already exists')

    X = pd.read_csv('random_fish_stocking_rows.csv', sep=',')
    Y = X['Number']
    X = X.drop(['Number'], axis=1)

    # transforming string data to int
    co = list(set(X['County']))
    wa = list(set(X['Waterbody']))
    to = list(set(X['Town']))
    mo = list(set(X['Month']))
    sp = list(set(X['Species']))
    X['County'] = X["County"].replace(tuple(co), tuple(range(len(co))))
    X["Waterbody"] = X["Waterbody"].replace(tuple(wa), tuple(range(len(wa))))
    X["Town"] = X["Town"].replace(tuple(to), tuple(range(len(to))))
    X["Month"] = X["Month"].replace(tuple(mo), tuple(range(len(mo))))
    X["Species"] = X["Species"].replace(tuple(sp), tuple(range(len(sp))))

    return X, Y


def func(x, a, b, c, d, e, f, g, h):
    result = h
    p = [a, b, c, d, e, f, g]
    for i in range(7):
        result += x[x.columns[i]] * p[i]
    return result


class CustomModelWrapper:
    def __init__(self, predicate_fun, parameters):
        self.predicate_fun = predicate_fun
        self.parameters = parameters

    def predict(self, x):
        return self.predicate_fun(x, *self.parameters)

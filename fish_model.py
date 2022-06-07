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
        data_frame = data_frame.sample(n=2000)
        data_frame.to_csv('random_fish_stocking_rows.csv', index=False)
        print('Downloaded')
    else:
        print(f'File "random_fish_stocking_rows.csv" already exists')

    X = pd.read_csv('random_fish_stocking_rows.csv', sep=',')
    Y = X['Number']
    X = X.drop(['Number'], axis=1)
    X = X.drop(['Waterbody'], axis=1)

    X = pd.get_dummies(X)

    return X, Y


def func(x, *parameters):
    result = parameters[len(x.columns)]
    for i in range(len(x.columns)):
        result += x[x.columns[i]] * parameters[i]
    return result


class CustomModelWrapper:
    def __init__(self, predicate_fun, parameters):
        self.predicate_fun = predicate_fun
        self.parameters = parameters

    def predict(self, x):
        return self.predicate_fun(x, *self.parameters)

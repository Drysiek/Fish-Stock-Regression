import pandas as pd
import os


def get_data():
    file = 'random_fish_stocking_rows.csv'
    if not os.path.isfile(file):
        print('Downlaoding data from: ', 'Fish Stocking.csv')
        data_frame = pd.read_csv('Fish Stocking.csv', sep=',')
        data_frame = data_frame[data_frame['Year'].notnull()]
        data_frame = data_frame[data_frame['Waterbody'].notnull()]
        data_frame = data_frame[data_frame['Month'].notnull()]
        data_frame = data_frame[data_frame['Number'].notnull()]
        data_frame = data_frame[data_frame['Number'] < 10000]  # this kind of data will look better on the figure
        data_frame = data_frame[data_frame['Species'].notnull()]
        data_frame = data_frame[data_frame['Size'].notnull()]
        data_frame = data_frame.drop(['County'], axis=1)
        data_frame = data_frame.drop(['Town'], axis=1)

        bool_series = data_frame.duplicated(['Year', 'Waterbody', 'Month', 'Species', 'Size'], keep=False)
        data_frame = data_frame[~bool_series]

        data_frame = data_frame.sample(n=600)
        data_frame.to_csv(file, index=False, sep=',')
        print('Downloaded')
    else:
        print(f'File {file} already exists')

    x = pd.read_csv(file, sep=',')
    y = x['Number']
    x = x.drop(['Number'], axis=1)

    x = pd.get_dummies(x)

    return x, y


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

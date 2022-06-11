# Fish-Stock-Regression
MSID project

Simply download all the files(.py and .csv) and run main.py.
If you want to draw new rows to calculate regression, type "y" in the console and then type number of rows you want to use(between 10 and 5000).
If youwant to leave already drawn rows, type "n" in the console.
Next, just wait for the program to print values of mean squared error and mean absolute percentage error for linear model, SVR model and custom model regression.
If there were drawn less than 550 rows, it is possible that custom regression won't be calculated.
Soon after, program will show figure with real and predicted values of fish used in fish stock for calculated regression models.



Description:
After transporting random rows of data from 'Fish Stocking.csv' to 'random_fish_stocking_rows.csv', program takes data to pandas dataframe.
Next program creates linear regression model, RVS model and custom regresion model and creates figure using matplotlib library.
Finally program calculates mean squared error and mean absolute percentage error for these regression models and shows figure with real and calculated values.

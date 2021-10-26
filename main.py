# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
from sklearn.impute import SimpleImputer
def DataExploration(df):
    df = df.dropna # drop rows with all colums Nan
    if  2 > len(df.index) or 2 > len(df.columns): # limit the DataSet to Min of 2x2
        return ("Error | The DataSet is too big or too small") # Stop the Script with Error Message
    print(df.dtypes)
    # Use a breakpoint in the code line below to debug your script.
    #If all Nan
    #If row == row


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    df = pd.read_csv('test.csv')
    print(df)
    print(DataExploration(df))
# See PyCharm help at https://www.jetbrains.com/help/pycharm/

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import pandas as pd
import numpy as np
import re
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt
import seaborn as sns
import sys

                    
        
def ErrorsSolver(df):
    # Use a breakpoint in the code line below to debug your script.
    if 2 > len(df.index) or 2 > len(df.columns):  # limit the DataSet to Min of 2x2
        print("Error | The DataSet is too big or too small")  # print Error message
    
    
    # replace all Nan None to NA
    df.replace(['nan','None','Missing','nfc','none','NULL','Null','NFS'], np.nan, inplace = True )
    
    spec_chars = "[!#%&()*+-./:;<=>?@[\\]^_`{|,},~,–]"
    for column in df.columns: 
        try:
            df[column] = df[column].str.replace(spec_chars, "", regex=True)
        except:
            continue # there is no characters to change in this column
        AvrNan = np.mean(df[column].isnull()) # We get array of True/False from using isnull, so if we count how many True from all we will get the avrage of nan in all column
        if AvrNan > float(0.3):
            df = df.drop([column], axis=1)
        
    #df = df.applymap(lambda s: s.lower() if type(s) == str else s) # all df to be in small letters
    df = df.dropna(how='all')  # Drop the rows where all elements are missing.
    df = df.dropna(axis='columns', how ='all') # Drop the columns where all elements are missing
    #df = df.drop_duplicates()  # Drop all duplicate rows
    return (df)

def MissingData(df):
    counter = int(0)
    newcolumns = df.columns.tolist() # getting all the Columns name 
    for i in newcolumns: # take each column and work on it
        testingCol =df.iloc[:, counter]# Column data
        b = (df[i].apply(type).value_counts()).index[0] #getting the type of this column 
        try: 
            column1 =testingCol.astype(b, errors='raise') # converting the column to the right type
            #print("the column type is change to " + str(b))
            column1.replace(['nan','None','Missing','nfc'], np.nan, inplace = True )
        except:
            column1 = pd.to_numeric(testingCol, errors='coerce')
            print('Used to_numric()')
        try:
            columnType = b.__name__ # geting the Column type as a stirng 
        except:
            columnType = 'str'
            print('Error 2 | Had a problem with converting the column type so it was converted')
        df[i] = column1
        CulomnsAsRow =df.iloc[:, counter:counter+1].values
        try:
            if columnType == 'str': # for each column type the Null will be field with diffrent 
                imr = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
                imr = imr.fit(CulomnsAsRow)
                df[i] = imr.transform( CulomnsAsRow)
                
            elif columnType == 'int':
                imr = SimpleImputer(missing_values=np.nan, strategy='median')
                imr = imr.fit(CulomnsAsRow)
                df[i] = imr.transform( CulomnsAsRow)
                
            elif columnType == 'float':
                df[i] = df[i].fillna(df[i].mean())
            else:
                print('Error3 | there is a new type that the script dont know how to handle ')
        except:
            print('Error | could not add values to the missing data at culomn: ' + str( i))
        counter = counter +1
    return (df)

def DataExploration(df):
    print(df.info())
    print(df.head(n=3))
    for column in df.columns:
        print(column)
        try:
            df[column].value_counts().plot(kind='bar', figsize=(50,10))
            plt.show()

        except: 
            continue
        try:
            df[column].plot.kde()
        except:
            continue
        
        try:
            df[column].plot.hist()
        except:
            continue
        
        try:
            print(df[column].apply(type).value_counts())
            df[column].apply(type).value_counts()
        except:
            continue
        
def RemoveColumnsCorr(df):
    mat = df.corr().abs()
    highCorr=np.where(mat>0.8)
    highCorr=[(mat.columns[x],mat.columns[y]) for x,y in zip(*highCorr) if x!=y and x<y]
    return df

def RemoveOutlier(df):
    df1 = df.copy()
    df = df._get_numeric_data()

    low = df.quantile(0.25)
    high = df.quantile(0.75)
    dif = high - low
    lower_bound = low -(1.5 * dif) 
    upper_bound = high +(1.5 * dif)
    
    
    for col in df.columns:
        for i in range(0,len(df[col])):
            if df[col][i] < lower_bound[col]:            
                df[col][i] = lower_bound[col]
    
            if df[col][i] > upper_bound[col]:            
                df[col][i] = upper_bound[col]    
    
    for col in df.columns:
        df1[col] = df[col]

    return(df1)



def Standardization(df):
    for i in df: # take each column and work on it
        try:
            sc_x = StandardScaler()
            X_train = sc_x.fit_transform(X_train)
            X_test = sc_x.transform(X_test)
        except:
            print('Error | Function Normalization() | Column: ' + str(testingCol ))
    return df


def Normalization(df):
    for i in df: # take each column and work on it
        testingCol =df[i]
        try:
            print('ss')
            mm_x= MinMaxScaler()
            testingCol= mm_x.fit_transform(testingCol)
        except:
            print('Error | Function Normalization() | Column: ' + str(testingCol ))
    return df
    
    
def DataCleaner(FileName):
    
    locasion = str(FileName+ '.csv')
    newfileLoc = str(FileName+'_cleaned.csv')
    df = pd.read_csv(locasion)
    df = ErrorsSolver(df) #
    df =MissingData(df)
    #df = Standardization(df)
    DataExploration(df)
    df = RemoveColumnsCorr(df)
    df = ErrorsSolver(df)
    df = RemoveOutlier(df)
    df.to_csv(newfileLoc)
    DataExploration(df)
    return df

if __name__ == '__main__':
    FileName = 'tes'
    df = DataCleaner(FileName)

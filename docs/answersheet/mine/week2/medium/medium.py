
'''
#### Mini Coding Challenges
1.	Write a function to compute the mean and standard deviation of a NumPy array.
2.	Group a DataFrame by a categorical column and compute the average of a numerical column.
3.	Merge two DataFrames on a common key and display the result.
4.	Create a scatter plot using Seaborn with custom colors and labels.
5.	Write a program to fill missing values in a DataFrame column with the column mean.
'''

#1.	Write a function to compute the mean and standard deviation of a NumPy array.
import numpy as np

def getmeanandstd(myarray):
    return myarray.mean(),myarray.std()

myarray = np.array([1,2,3])
print(getmeanandstd(myarray))
    

#2.	Group a DataFrame by a categorical column and compute the average of a numerical column.
import pandas as pd

df = pd.DataFrame(
    {
        "Name" : ['asura','beera','ank','yank','asura','beera','ank','yank'],
        "Subject": ['Math','Math','Math','Math','English','English','English','English'],
        "Score": [56,45,78,89,65,56,89,98]
    }
)
print(df.groupby('Subject', as_index=False).agg(Avg=('Score','mean')))

# 3.	Merge two DataFrames on a common key and display the result.
df1 = pd.DataFrame(
    {
        "Name": ['Mar','Maoiu','Drek','Zinga'],
        "Age" : [34,23,15,45]
    }
)

df2 = pd.DataFrame(
    {
        "Name": ['Mar','Maoiu','Drek','Zinga','Bazinga'],
        "IQ" : [56,48,67,87,98]
    }
)

print('Left join method 1 --> \n',pd.merge(df1,df2,on='Name',how='left'))
print('Left join method 2 --> \n',df1.merge(df2,on='Name',how='left'))
print('Outer Join--> \n',pd.merge(df1,df2,on='Name',how='outer').fillna({'Age':20}))

#4.	Create a scatter plot using Seaborn with custom colors and labels.
import seaborn as sns
import matplotlib.pyplot as plt
# sns.scatterplot(df1,x='Name',y='Age',palette='pastel')


# 5.	Write a program to fill missing values in a DataFrame column with the column mean.
def fillna(mydf):
    correcteddf = mydf
    correcteddf = correcteddf.fillna({'Subject':'English'})
    correcteddf['Score'] = correcteddf.groupby('Subject')['Score'].transform(lambda x: x.fillna(x.mean()))
    return correcteddf

mydf = pd.DataFrame(
    {
        "Name" : ['asura','beera','ank','yank','asura','beera','ank','yank','marek','marek'],
        "Subject": ['Math','Math','Math','Math','English','English','English','English','Math',np.nan],
        "Score": [56,45,78,89,65,56,89,98,np.nan,np.nan]
    }
)
print(fillna(mydf))

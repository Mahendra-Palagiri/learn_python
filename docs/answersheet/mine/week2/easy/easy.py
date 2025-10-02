'''
#### Mini Coding Challenges
1.	Create a NumPy array of numbers from 1 to 10 and print it.
2.	Create a Pandas DataFrame with columns “Name” and “Age” and print it.
3.	Plot a bar chart of the number of students in different classes using Matplotlib.
4.	Filter a DataFrame to show only rows where the score is above 80.
5.	Use Seaborn to create a histogram of a given dataset.
'''

import numpy as np
import pandas as pd

#1.	Create a NumPy array of numbers from 1 to 10 and print it.
'''Rudimentary way'''
nparray  = np.array(range(1,11))
print(nparray)

'''Best way'''
npnatarray = np.arange(1,11)
print(npnatarray)

'''other possible way'''
npotarray = np.linspace(1,10,10,dtype=int)
print(npotarray)

#2.	Create a Pandas DataFrame with columns “Name” and “Age” and print it.
df = pd.DataFrame(
    {
        "Name": ['Sever','Sven','Smear'],
        "Age" : [34,50,32]
    }
)
print(df)


#3.	Plot a bar chart of the number of students in different classes using Matplotlib.
import matplotlib.pyplot as plt
bdf = pd.DataFrame(
    {
        "Class" : ['Math','Science','Art','Social','English'],
        "Students": [20,18,14,13,20]
    }
)
plt.bar(bdf['Class'], bdf['Students'])
plt.title("Class vs Count of Students")
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

#4.	Filter a DataFrame to show only rows where the score is above 80.
stdf = pd.DataFrame(
    {
        "Name" : ['sven','sven','giri','matt','matt','rachel'],
        "Subject": ['math','english','math','math','computers','arts'],
        "Score": [87,79,98,45,90,56]
    }
)
print(stdf[stdf['Score']>80])

'''Alternative approach'''
print(stdf.query('Score>80'))

#5.	Use Seaborn to create a histogram of a given dataset.
import seaborn as sns
#Intial attemmpt which is incorrect  --> sns.histplot(stdf,x='Name',y='Score',hue='Subject') 
sns.histplot(stdf['Score'],bins=10)
plt.show()
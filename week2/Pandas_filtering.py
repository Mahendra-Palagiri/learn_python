#🎯 Learning Goals
#	1.	Filter data using conditions
#	2.	Sort data by column(s)
#	3.	Add and modify columns
#	4.	Group and aggregate data
#	5.	Merge/join DataFrames

import pandas as pd
import numpy as np

# Sample DataFrame
data = {
    "Name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
    "Age": [25, 30, 35, 40, 28],
    "Score": [88, 92, 85, 90, 95],
    "Subject": ["Math", "Math", "Science", "Science", "Math"]
}

df = pd.DataFrame(data)
print(df)

# 1️⃣ Filtering rows
print("\nFilterning rows with score greater than 90:\n",df[df['Score']>90])

# 2️⃣ Sorting data
print(f"\n Sorting data by score:\n {df.sort_values(by=['Subject','Score'], ascending=False).loc[:,['Name','Subject','Score']]}") #Sort by Subject and Score  and select whatever columns we need

# 3️⃣ Adding/Modifying a column
df['Pass'] = df['Score'] > 75
print(f"\n Added new column to dataframe :\n {df}")

# 4️⃣ Grouping and aggregation
print(df.groupby('Subject')['Score'].mean()) #Group by 'Subject' and aggregate on 'Score'

# 5️⃣ Merging DataFrames
clubs = pd.DataFrame({
    "Name": ["Alice", "Bob", "Eve"],
    "Club": ["Robotics", "Drama", "Chess"]
})

dfm = pd.merge(df, clubs, on='Name', how='left') #Like merging to tables
odfm = df.merge(clubs, on='Name', how='left')
print(f"\nmerged data set:\n {dfm}")
print(f"\n Other way of merging data set:\n {odfm}")



#🏋️ Part 2: Exercises
#	1.	Filtering
#	    •	From sample.csv, select students with Score >= 90 and Age < 35
#	2.	Sorting
#	    •	Sort sample.csv by Age ascending, then by Score descending
#	3.	Column Addition
#	    •	Add a column Grade based on:
#	    •	Score >= 90 → "A"
#	    •	80 <= Score < 90 → "B"
#	    •	Otherwise "C"
#	4.	Grouping
#	    •	Group sample.csv by Grade and count how many students in each
#	5.	Merging
#	    •	Create a new DataFrame with Name and Hobby
#	    •	Merge it with sample.csv data

# 1️⃣ Filtering
sdf = pd.read_csv('./data/week2/sample.csv')
print(sdf)
print(f"\n Filtering students who scored more than or equal to  90 and whose age is less than 35\n. {sdf[(sdf['Score'] >= 90) & (sdf['Age'] < 35)]}") #If there is more than one filtering criteria ensure each condition is enclosed in ()

# 2️⃣ Sorting
print("\n Sorting data: \n",sdf.sort_values(by=['Age','Score'], ascending=[True,False]))

# 3️⃣ Column Addition
def grade_switch(score):
    if score >= 90:
        return 'A'
    elif score >= 80:
        return 'B'
    else:
        return 'C'

sdf['Grade'] = sdf['Score'].apply(grade_switch)
print("\n Modified Data frame with Grade : \n,",sdf)


conditions =[
    sdf['Score'] >= 90,
    (sdf['Score'] >=80) & (sdf['Score'] <90)
]
choices =['A','B']
sdf['modGrade'] = np.select(conditions,choices,default='C')
print("\n Modified Data frame with Grade : \n,",sdf) #Using Conditions and Choices is the most optimized way.

# 4️⃣ Grouping
print("\nGroup by grade and count how many for each grade: \n",sdf.groupby(by='Grade').count())

print("\n Alternate way of group by grade and count how many for each grade: \n",sdf['Grade'].value_counts()) #Just gives the column 'Grade' and the count (instead of whole table)

hdf = pd.DataFrame({
    "Name": ['Alice','Bob','Diana','Mahi'],
    "Hobby": ['Music','Surfing','Singing','Coding']
})

mrgdf = pd.merge(sdf,hdf, on='Name',how='left') #Left will give data entities only in left data frame (i.e. New Name 'Mahi' won't appear)
print("\n Merged Data frame : \n", mrgdf)

smrgdf = pd.merge(sdf,hdf, on='Name',how='outer')  #Outer will get all data in left and any new values in right data frame also. i.e. 'Mahi' will appear in output
print("\n Merged Data frame : \n", smrgdf)
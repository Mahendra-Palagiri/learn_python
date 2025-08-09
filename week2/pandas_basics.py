import pandas as pd


# 1️⃣ Series (1D labeled data)
scores = pd.Series([10,20,30,40], name="Scores")
print("\n scores : \n",scores)
print("\n Mean of Scores:\n",scores.mean())


# 2️⃣ DataFrame (2D labeled data)
data = {
    "Name": ["Alice","Bob","Rick","Diana"],
    "Age" : [25,30,35,40],
    "Score" : [88,92,85,90]
}
df = pd.DataFrame(data)
print("\n DataFrame example:\n",df)


# 3️⃣ Accessing Data
for col in df.columns:
    print(f"\n Elements of Column: '{col}' :\n {df[col]}") #Adding 'f' at start of print can be used when you want the values in string formatting to be dynamically evaluated.


#iloc. --> Accessing data elements based on integer location. 
print("\n Accessing first two rows of data frame \n",df.iloc[:2]) 
print("\n Accessing row three and onwards \n",df.iloc[2:]) 
print("\n Accessing first two rows and first two columns of each row \n",df.iloc[0:2,0:2]) 
print(f"\n First Row:\n{df.iloc[0]},\n Last Row:\n{df.iloc[-1]},\n First and Third Row:\n{df.iloc[[0,2]]}")
print(f"\n first column of first row:\n{df.iloc[0,0]},\n Second column of first row:\n{df.iloc[0,1]},\n third column of fourth row:\n{df.iloc[3,2]}") #First element is Row and second element is column


#loc --> Access data elements by label values. (rather than index values)
print("\n Accessing the score of Row 0\n",df.loc[0,"Score"])


# 4️⃣ Summary statistics
print("\n Summary Statistics:\n",df.describe())


# 5️⃣ Load from CSV
csvdata = pd.read_csv("./data/week2/sample.csv")
print("\n csv header data:\n",csvdata.head())


#🏋️ : Exercises
#1.	Series Practice
#   •	Create a Series of [5, 10, 15, 20]
#   •	Find its mean and standard deviation
#2.	DataFrame Practice
#   •	Create a DataFrame with columns: Product, Price, Quantity (4 rows)
#   •	Calculate Total Price = Price × Quantity
#3.	CSV Loading & Exploration
#   •	Load data/sample.csv
#   •	Show:
#       •	First 3 rows
#       •	Average Score
#       •	All rows where Score > 90
#4.	Subsetting Challenge
#   •	Print Name and Score for top 3 scoring students

# 💡 Solution:
# 1️⃣ Series Practice
series  = pd.Series([5,10,15,20],name="Num")
print(f"\n Mean of series:\n{series.mean()} \nStandard Deviation of Series:\n{series.std()}")

# 2️⃣ DataFrame Practice
edata = {
    "Product" : ["Pen","Pencil","Eraser","Sharpner"],
    "Price" : [1.2,0.5,0.25,0.36],
    "Quantity" : [3,8,5,9]
}
edf = pd.DataFrame(edata)
for index, row in edf.iterrows():
    print(f"\n Product:'{row['Product']}'--> Quantity:{row['Quantity']} -->Price:{row['Price']} ==> Total Price: {row['Quantity']*row['Price']:.2f}")

#probably efficient soluiont
edf["TotalPrice"] = edf["Quantity"] * edf["Price"]
print("\n extended data frame with total prices\n",edf)
   

# 3️⃣ CSV Loading & Exploration
ecsv = pd.read_csv("./data/week2/sample.csv")
print(ecsv)
print(f"\n First 3 Rows:\n{ecsv.iloc[:3]} \nAlternative way\n: {ecsv.head(3)}")
print(f"\nAverage Score\n: {ecsv['Score'].mean()} ")
print(f"\n Rows where Score > 90 \n: {ecsv[ecsv['Score']>90]} ") #Apply masking function(fitler) for a dataframework similar to numpy

# 4️⃣ Subsetting Challenge
print("\n Top 3 scoring students:\n",ecsv.sort_values(by="Score",ascending=False).iloc[:3])
print("\n Alternative way of displyaing top 3 scoring students:\n",ecsv.sort_values(by='Score',ascending=False).loc[:,['Name','Score']].head(3))

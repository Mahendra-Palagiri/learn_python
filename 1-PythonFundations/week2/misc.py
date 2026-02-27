# 1.	Write a function is_even(n) that returns True if a number is even.
def is_even(n):
    return n%2==0

print(is_even(4))


# 2.	Take a list of numbers and create:
# 	•	A list of their squares.
# 	•	A dictionary mapping each number → square.
arr = [2,3,4,5]
sqrarr =[]
sdict ={}
for val in arr:
    sqrarr.append(val*val)
    sdict[val] = val*val

print(sqrarr)
print(sdict)

#Efficient solution
osqrarr = [num**2 for num in arr]
osdict = {num:num**2 for num in arr}
print(osqrarr)
print(osdict)


# 3.	Create a NumPy array of 20 random integers (1–100).
# •	Find the mean and standard deviation.
# •	Select all numbers greater than the mean.
import numpy as np
np.random.seed(42)
randarr = np.random.randint(1,100,size=20)
print(randarr)
print(randarr.mean())
print(randarr.std())
print(randarr[randarr > randarr.mean()])


# 4.	Create a DataFrame with columns: Name, Math, Science.
# •	Add a new column Average as the mean of Math and Science.
# •	Filter rows where Average > 80.
import pandas as pd
df = pd.DataFrame(
    {
        "Name": ['Alice','Bob','Rand'],
        "Math": [90,75,88],
        "Science": [85,95,70]
    }
)
df["Avg"] = df[['Math','Science']].mean(axis=1)
print(df)
print(df[df['Avg']>80])
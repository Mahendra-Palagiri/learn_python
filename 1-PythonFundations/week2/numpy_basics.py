import numpy as np

#1D Array
array_1d = np.array([1,2,3,4])
print(array_1d)

#Array Matrix
arr_mat = np.array([[1,2],[3,4]])
print()
print(arr_mat)

#Special Arrays
arr_z = np.zeros((2,3)) #First element can be thought about as row and second element as columns
arr_o = np.ones(5) # A one dimensional arrary of 1s
arr_r = np.arange(0,10,2) #Provide numbers between 0 and 10 with an increments of 2
print()
print(arr_z)
print(arr_o)
print(arr_r)

#Performing math operations on arrays
a = np.array([10,20,30])
b = np.array([1,2,3])

print()
print("Addition :", a+b)
print("Substraction :", a-b)
print("Multiplication :", a*b)
print("Division :", a/b)
print("Mean of a :", np.mean(a)) #average
print("Sum of b :", np.sum(b))
print("Median of b :", np.median(b)) #Sort and find the middle number if odd , if even numbers take the middle two and divide by 2 to get median
print("Standard deviation of a:",np.std(a))

#Indexing and slicing
print()
print("Fisrt element of a:", a[0])
print("Last row of Matrix array:",arr_mat[-1])
print("Slice array_1d[1:3]:",array_1d[1:3]) # original arrary elements [1 2 3 4] with slice you are getting part of this data  (Starting at second index and less than 3rd index) i.e. 2, 3

#Reshape and Transpose
reshaped = np.reshape(array_1d,(2,2))
print()
print("Reshaped array:",reshaped)# original arrary elements [1 2 3 4] reshape of 2,2 will split the arrary in matrix of 2 rows and 2 columns each i.e. [1,2] & [3,4]


transposed = arr_mat.T
print()
print("Transposed array:",transposed)

print()

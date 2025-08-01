import numpy as np
#------------------------------------------------------Broadcasting--------------------------------------------------------#
a = np.array([2,6,9])
b = 2
print("\nBroadcast 'b' to 'a' elements in the form of addition:\n",a+b) #Adds 2 (the value of b) to each element of array a
        #Expected output. : Broadcast 'b' to 'a' elements in the form of addition: [ 4  8 11]


#Broadcasting for compatible 2D and 1D array
arr_2d = np.array([[12,19,23],[45,76,34]])
arr_1d = np.array([3,4,5])
print("\nBroadcast 'arr_1d' to 'arr_2d' elements in the form of addition:\n",arr_2d+arr_1d) 
        #Expected output. : Broadcast 'arr_1d' to 'arr_2d' elements in the form of addition: [[15 23 28][48 80 39]]


#-------------------------------------------------------Boolean maskign and filtering-------------------------------------#
arr_a = np.array([5,10,45,36,98,26])
mask = arr_a > 27
print(mask) #Expected output [False, False, True, True, True, False]
print("\nFiltered content:\n",arr_a[mask]) #Expected Output  [45,36,98]


#-----------------------------------------------Stacking arrays (Horizontal and Vertical)----------------------------------#
x = np.array([1,2,3])
y = np.array([4,5,6])
print("\nVertical Stacking of 'x' and 'y':\n",np.vstack((x,y))) #Combined two 1 d arrays into one two dimensional array
        #Expected output -->  Vertical Stacking of 'x' and 'y': [[1 2 3] [4 5 6]]
print("\nHorizontal Stacking of 'x' and 'y':\n",np.hstack((x,y))) #Combined two 1 d arrays into one one dimensional array
        #Expected output -->  Horizontal Stacking of 'x' and 'y': [1 2 3 4 5 6]


#-------------------------------------------------Split array into smaller parts-----------------------------------------#
aspt  = np.array([1,2,3,4,5,6]) 
ospt = np.array([1,2,3,4,5,6,7,8])
#split function can only be used if the array is divisible by the number specified
#In the following example aspt has 6 elements and is completely divisble by 3
print("\nSplitting array into 3 equal parts:\n",np.split(aspt,3)) #Split array in 3 equal parts .  [1,2] [3,4] [5,6]

#If the array is not completely divisibl by the number of parts we want, we have to use array_split function
#In the following example ospt has 8 elements and is NOT completely divisble by 3
print("\nSplitting array into 3 parts:\n",np.array_split(ospt,3)) #Split array in 3 parts .  [1,2,3] [4,5,6] [7,8]


#----------------------------------------------Random nubmers-----------------------------------------#
#Setting seed will ensure you get same result for all subsequent rand functions used  (Seed value can be any integer need not be 42)
#If seed is not set , every time random function is run the value might be different
np.random.seed(42)  
rand_arr = np.random.rand(3,3) # 3x3 Random floats. (3 Random values between 0,1 with a 3x3 matrix)
        #Sample output. --> Random float array :
        #[[0.37454012 0.95071431 0.73199394]
        #[0.59865848 0.15601864 0.15599452]
        #[0.05808361 0.86617615 0.60111501]]

rand_int = np.random.randint(1,10, size=(2,5))
print("\nRandom float array :\n",rand_arr)
print("\nRandom int array :\n",rand_int)
        #Sample output. --> Random int array :
        #[[8 3 6 5 2]
        #[8 6 2 5 1]]


############################### Excercise ####################
	#1.	Broadcasting Practice
	#   •	Create an array [5, 10, 15, 20]
	#   •	Multiply all elements by 3 using broadcasting
	#2.	Filtering
	#   •	From an array of [10, 25, 30, 45, 50], extract all elements greater than 30
	#3.	Stack & Split
	#   •	Stack two arrays [1,2,3] and [4,5,6] vertically
	#   •	Then split the result back into two separate rows
	#4.	Random Data Challenge
	#   •	Generate a 5×5 random integer matrix between 1–100
	#   •	Filter out all even numbers

ea  = np.array([5,10,15,20])
m = 3
print("\nBroadcasting multiplicaiton of array 'ea' with 'm':\n",ea*m)

fa =np.array([10, 25, 30, 45, 50])
filt = fa > 30
print("\nElements greater than 30:\n",fa[filt])

ax = np.array([1,2,3])
ay = np.array([4,5,6])
print("\n Vertical Stacking of arrays 'ax' and 'ay':\n",np.vstack((ax,ay)))
print("\n Horizontal Stacking of arrays 'ax' and 'ay':\n",np.hstack((ax,ay)))
print("\n Split the horizaontal stack back to two arrays\n",np.array_split(np.hstack((ax,ay)),2))

np.random.seed(21)
ri = np.random.randint(1,100,size=(5,5))
print("\n 5X5 matrix of random integers between 1 and 100:\n",ri)

evennum_mask = ri%2==0
print("\n Even numbers in 5X5 matrix of random integers between 1 and 100:\n",ri[evennum_mask])




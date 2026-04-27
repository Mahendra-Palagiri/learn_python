import torch


# =============================================================================
# 1. Creating tensors
# =============================================================================
vector = torch.tensor([10,20,30,40])
zeros = torch.zeros(2,3)
ones = torch.ones(2,3)
range_tensor = torch.arange(0,12)
rand_tensor = torch.rand(2,3)

print(f"\n\n -- 1. Creating Tensors --\n vector : {vector}  \n zeros : {zeros} \n ones: {ones}\n range tensor: {range_tensor}\n random tensor: {rand_tensor}")

''' -- OUTPUT--
 -- 1. Creating Tensors --
 vector : tensor([10, 20, 30, 40])  
 zeros : tensor([[0., 0., 0.],
        [0., 0., 0.]]) 
 ones: tensor([[1., 1., 1.],
        [1., 1., 1.]])
 range tensor: tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
 random tensor: tensor([[0.7319, 0.6526, 0.3321],
        [0.3039, 0.8209, 0.2604]])
'''


# =============================================================================
# 2. Indexing and Slicing
# =============================================================================
x = torch.tensor([10,20,30,40,50])

print("\nIndexing and Slicing")
print("\nx[0]:", x[0]) #Indexing --> First element --> 10
print("\nx[2]:", x[2]) #Indexing -->  Third element --> 30
print("\nx[-1]:", x[-1])#Indexing -->  Last element --> 50
print("\nx[-12:", x[-2])#Indexing -->  Last but one element --> 40
print("\nx[1:4]:", x[1:4])#Slicing (Stop and Start Indexes) Stop Index element is not considered  --> Values from 1st Index to 4th Index --> 20,30,40 
print("\nx[:3]:", x[:3])#Slicing --> Values starting with Index 0 upto Index 3 --> 10,20,30 
print("\nx[2:]:", x[2:])#Slicing --> Values starting with Index 2 upto end--> 30,40,50
print("\nx[::2]:", x[::2])#Slicing --> Every second value --> 10,30,50

''' --OUTPUT --
Indexing and Slicing
x[0]: tensor(10)
x[2]: tensor(30)
x[-1]: tensor(50)
x[-12: tensor(40)
x[1:4]: tensor([20, 30, 40])
x[:3]: tensor([10, 20, 30])
x[2:]: tensor([30, 40, 50])
x[::2]: tensor([10, 30, 50])
'''

# =============================================================================
# 3. 2D Indexing and Slicing
# =============================================================================
x = torch.tensor([
    [1,2,3],
    [4,5,6],
    [7,8,9]
])

print("\n--OUTPUT--\n Matrix(2D)  Indexing and Slicing")
print(f"\n Shape:{x.shape}")
print(f"\n x[0,0]:{x[0,0]}") # Indexing ---> Select first row first column value --> "1"
print(f"\n x[1,2]:{x[1,2]}") # Indexing ---> Select second row third column value --> "6"
print(f"\n x[1]:{x[1]}") # Slicing ---> Select Second row  values --> "[4,5,6]"
print(f"\n x[:,1]:{x[:,1]}") # Slicing ---> Select second columns values --> "2,5,8"
print(f"\n Top left block: {x[:2,:2]}") # Slicing ---> Select first two rows and first two columns --> "[[1,2],[4,5]]"

'''--OUTPUT--
 Matrix(2D)  Indexing and Slicing

Shape:torch.Size([3, 3])
x[0,0]:1
x[1,2]:6
x[1]:tensor([4, 5, 6])
x[:,1]:tensor([2, 5, 8])
Top left block: tensor([[1, 2],
        [4, 5]])
'''


# =============================================================================
# 4. Reshaping
# =============================================================================
x = torch.arange(0,12)
print("\n--OUTPUT--\n Reshaping")
print(f"\n X:{x}")
print(f"\n Reshape 4x3 :{x.reshape(4,3)}") 
# A matrix tensor with 4 rows and 3 columns
#The original entity has 12 elements so (4x3=12) is a valid shape for tensor 
#An invalid entity would be something like 5x3 (which results in 15 elements)
print(f"\n Reshape 2x6 :{x.reshape(2,6)}") #A matrix tensor with 2 rows and 6 columns
print(f"\n Reshape 3x4 :{x.reshape(3,-1)}") #A matrix tensor with 3 rows (and columns infered(calculated) by tensor)
print(f"\n Reshape 6x2 :{x.reshape(-1,2)}") #A matrix tensor with 2 columns (and rows infered(calculated) by tensor)
#NOTE : we cannot provide -1, -1 when reshaping (i.e. atleast one dimension has to be clearly defined and the other one infered)

''' --OUTPUT--
Reshaping

 X:tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])

 Reshape 4x3 :tensor([[ 0,  1,  2],
        [ 3,  4,  5],
        [ 6,  7,  8],
        [ 9, 10, 11]])

 Reshape 2x6 :tensor([[ 0,  1,  2,  3,  4,  5],
        [ 6,  7,  8,  9, 10, 11]])

 Reshape 3x4 :tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])

 Reshape 6x2 :tensor([[ 0,  1],
        [ 2,  3],
        [ 4,  5],
        [ 6,  7],
        [ 8,  9],
        [10, 11]])
'''


# =============================================================================
# 5. Unsqueeze and Squeeze (Adding and removing dimensions to tensor)
# =============================================================================
x = torch.tensor([8.0,2.0])
print("\n--OUTPUT--\n Unsqueeze and Squeeze")
print(f"\n X:{x} \nx.shape = {x.shape}") #A one dimensional tensor with 2 values

# Unsqueeze is used to add dimension 
# Neural networks typically expects input shape like [batch_size, number_of_features]
x_unsqz_x = x.unsqueeze(0) #Add one sample (and use the exisiting values as columns/features)
print(f"\n unsqueeze X = {x_unsqz_x} \n dimensions = {x_unsqz_x.shape}") # We changed the vector to matrix [1,2] i.e.  with 1 sample 2 features 

x_unsqz_y = x.unsqueeze(1) #Add one sample for each existing values
print(f"\n unsqueeze X = {x_unsqz_y} \n dimensions = {x_unsqz_y.shape}") # We changed the vector to matrix [2,1] i.e. two samples with one feature each


#Squeeze is to remove a dimension
x_sqz_x = x_unsqz_x.squeeze(0)
print(f"\n squeeze X = {x_sqz_x} \n dimensions = {x_sqz_x.shape}")
x_sqz_y = x_unsqz_y.squeeze(1)
print(f"\n squeeze X = {x_sqz_y} \n dimensions = {x_sqz_y.shape}")

'''--OUTPUT--
 Unsqueeze and Squeeze

 X:tensor([8., 2.]) 
x.shape = torch.Size([2])

 unsqueeze X = tensor([[8., 2.]]) 
 dimensions = torch.Size([1, 2])

 unsqueeze X = tensor([[8.],
        [2.]]) 
 dimensions = torch.Size([2, 1])

 squeeze X = tensor([8., 2.]) 
 dimensions = torch.Size([2])

 squeeze X = tensor([8., 2.]) 
 dimensions = torch.Size([2])
'''

# =============================================================================
# 6. Math and reductions (aggregrations on the tensor) and broadcasting
# =============================================================================
a = torch.tensor([1.0,2.0,3.0])
b = torch.tensor([9.0,18.0,27.0])
print("\n--OUTPUT--\n Math ,  Reductions and broadcasting")
print(f"\n Math operations \n addition = {a+b}\n substraction = {b-a}\n multiplication = {a*b}\n division = {b/a}")
print(f"\n Reductions : \n sum={a.sum()}, \n mean={a.mean()}")

m = torch.tensor([
    [1.0,2.0,3.0],
    [4.0,5.0,6.0]
])
print(f"\n m = {m}\n mean = {m.mean()}")
print(f"\n By Columns. -->  sum (dim=0) = {m.sum(dim=0)} \n mean (dim=0) = {m.mean(dim=0)}") #Sum and Mean  --> Columns
print(f"\n By Rows. --> sum (dim=1) = {m.sum(dim=1)} \n mean (dim=1) = {m.mean(dim=1)}") #Sum and Mean --> Rows

# Broadcasting
bias = torch.tensor([10.0,20.0,30.0])
print(f"\n Broadcast m+bias = {m+bias}")

'''--OUTPUT--
 Math ,  Reductions and broadcasting

 Math operations 
 addition = tensor([10., 20., 30.])
 substraction = tensor([ 8., 16., 24.])
 multiplication = tensor([ 9., 36., 81.])
 division = tensor([9., 9., 9.])

 Reductions : 
 sum=6.0, 
 mean=2.0

 m = tensor([[1., 2., 3.],
        [4., 5., 6.]])
 mean = 3.5

 By Columns. -->  sum (dim=0) = tensor([5., 7., 9.]) 
 mean (dim=0) = tensor([2.5000, 3.5000, 4.5000])

 By Rows. --> sum (dim=1) = tensor([ 6., 15.]) 
 mean (dim=1) = tensor([2., 5.])

 Broadcast m+bias = tensor([[11., 22., 33.],
        [14., 25., 36.]])
'''
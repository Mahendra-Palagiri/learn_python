import torch


# =============================================================================
# 1. Scalar Tensor (Single value)
# =============================================================================
scalar = torch.tensor(7)
print(f"\n\n -- 1. Scalar Tensor --\n {scalar}  \n Shape : {scalar.shape} \n Dtype: {scalar.dtype}")

# =============================================================================
# 2. Vector Tensor (One dimensional array of values)
# =============================================================================
vector = torch.tensor([10,20,30])
print(f"\n\n -- 2. Vector Tensor --\n {vector}  \n Shape : {vector.shape} \n Dtype: {vector.dtype}")

# =============================================================================
# 3. Matrix Tensor (two dimensional array of values)
# =============================================================================
matrix = torch.tensor([
    [1,2,3],
    [4,5,6]
]) # 2 Samples with 3 features each
print(f"\n\n -- 3. Matrix Tensor --\n {matrix}  \n Shape : {matrix.shape} \n Dtype: {matrix.dtype}")


# =============================================================================
# 4. Floating Point/Feature Tensor (typically used in neural networks)
# =============================================================================
x = torch.tensor([
    [8.0,2.0],
    [4.0,7.0],
    [6.0,1.0],
    [3.0,9.0]
])
print(f"\n\n -- 4. Floting point aka Feature Tensor --\n {x}  \n Shape : {x.shape} \n Dtype: {x.dtype}")

# =============================================================================
# 5. Integer Tensor (typically used for classfication lables)
# =============================================================================
y = torch.tensor([1,0,1,0])
print(f"\n\n -- 5. Label --\n {y}  \n Shape : {y.shape} \n Dtype: {y.dtype}")

# =============================================================================
# 6. Applying layering on features
# =============================================================================
layer = torch.nn.Linear(2,3)
output = layer(x) #Our input X has 2 features (matching the first dimension of the layer)
print(f"\n\n -- 6. Layered output --\n {output}  \n Shape : {output.shape} \n Dtype: {output.dtype}")

'''. --- OUTPUT ----

-- 1. Scalar Tensor --
 7  
 Shape : torch.Size([]) 
 Dtype: torch.int64


 -- 2. Vector Tensor --
 tensor([10, 20, 30])  
 Shape : torch.Size([3]) 
 Dtype: torch.int64


 -- 3. Matrix Tensor --
 tensor([[1, 2, 3],
        [4, 5, 6]])  
 Shape : torch.Size([2, 3]) 
 Dtype: torch.int64


 -- 4. Floting point aka Feature Tensor --
 tensor([[8., 2.],
        [4., 7.],
        [6., 1.],
        [3., 9.]])  
 Shape : torch.Size([4, 2])  # 4 Samples with 2 features each
 Dtype: torch.float32


 -- 5. Label --
 tensor([1, 0, 1, 0])  
 Shape : torch.Size([4]) # 4 -Labels
 Dtype: torch.int64


 -- 6. Layered output --
 tensor([[-1.2903,  2.7107,  0.0942],
        [-1.0527, -0.8895, -4.1548],
        [-0.9488,  2.2356,  0.2694],
        [-1.0411, -2.0735, -5.7099]], grad_fn=<AddmmBackward0>)  
 Shape : torch.Size([4, 3])  # Turned a 4 sample 2 features input to 4 Sample 3 feature output
 Dtype: torch.float32
'''
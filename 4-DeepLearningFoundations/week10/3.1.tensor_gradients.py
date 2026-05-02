import torch


# =============================================================================
# 1. Basic required_grad example
# =============================================================================
x = torch.tensor(3.0,requires_grad=True) #We are explictly asking tensor to track gradients

y = x**2
print(f"\n Example 1 : y = x ^ 2")
print(f"\n X:{x} \n Y (x**2):{y}")

#compute dy/dx (Relative change of y wrt to X)
y.backward()
print(f"\n X.grad:{x.grad}")


'''------ OUTPUT --------
 Example 1 : y = x ^ 2

 X:3.0 
 Y:9.0

 X.grad:6.0
'''

# =============================================================================
# 2. Longer chain explained
# =============================================================================
x= torch.tensor(2.0,requires_grad=True)
y = x*3
z = y+4
loss = z**2

print("\n----- OUTPUT -----")
print(f"\n Example 2 : Longer Computation Graph")
print(f"\n X:{x} \n Y(x*3):{y}\n Z(y+4):{z} \n loss(z**2):{loss}")
loss.backward()
print(f"\n X.grad:{x.grad}")

'''----- OUTPUT -----
 Example 2 : Longer Computation Graph

 X:2.0 
 Y(x*3):6.0
 Z(y+4):10.0 
 loss(z**2):100.0

 X.grad:60.0
'''

# =============================================================================
# 3. Gradient accumulation
# =============================================================================
x = torch.tensor(3.0,requires_grad=True)
y = x**2
y.backward()

print("\n----- OUTPUT -----")
print(f"\n Gradient accumulation")
print(f"\n After First pass X.grade:{x.grad}")

z = x**2
z.backward()
print(f"\n After Second pass X.grade:{x.grad}")

'''----- OUTPUT -----

 Gradient accumulation (Gradients accumulate if not cleared)

 After First pass X.grade:6.0

 After Second pass X.grade:12.0
'''



# =============================================================================
# 4. Clearing gradients manually
# =============================================================================
x = torch.tensor(3.0,requires_grad=True)
y = x**2
y.backward()

print("\n----- OUTPUT -----")
print(f"\n Gradient accumulation")
print(f"\n After First pass X.grade:{x.grad}")

x.grad=None #Manually clearing the gradients

z = x**2
z.backward()
print(f"\n After Second pass X.grade:{x.grad}")

'''----- OUTPUT -----

 Gradient accumulation (Gradients has been cleared after first pass)

 After First pass X.grade:6.0

 After Second pass X.grade:6.0
'''

# =============================================================================
# 5. No grade xample
# =============================================================================
x = torch.tensor(3.0,requires_grad=True)


with torch.no_grad():
    y = x**2

print("Example 5: torch.no_grad()")

print("y:", y)

print("Does y require grad?", y.requires_grad)

print(f"\n\n")


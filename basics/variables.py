print ("Hello there")

# Variables and Data Types
name = "Mahi"
age = 40
print("My name is", name, "and I am", age, "years old.")

# Conditional Statements
temperature = 72
if temperature > 75:
    print("It's a hot day.")
elif temperature > 60:
    print("Awesome weather! Let's go for a walk.")
else:
    print("It's little chilly today.") 

# Loops
for i in range(3):
    print("This is iteration number", i)

n=0
while n < 5:
    print("while loop iteration number", n)
    n += 1      


# Condition based on user input
x =0
while x < 5:
    user_input = int(input("Enter a number: "))
    if user_input %2 == 0:
        print("You entered an even number.")
    else:
        print("You entered an odd number.")
    x += 1

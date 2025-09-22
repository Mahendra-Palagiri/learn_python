'''1.	Write a program that prints “Hello, World!”.
2.	Ask the user for their name and age, and print a message back.
3.	Write a program to find the largest of two numbers.
4.	Swap two numbers without using a third variable.
5.	Write a program that takes a number and prints whether it is even or odd.'''

#1.--------------------------------------------------------------------------------------
print("Hello, World!")


#2.--------------------------------------------------------------------------------------
name = input("Please provide your name\n")
age = 0

def get_age():
    try:
        return int(input("Please provide your age\n"))
    except:
        print("Please provide proper age")
        return 0

#Ensure user provided proper age        
while age  <= 0:
    age = get_age()

print(f"\nHello '{name}'! nice to meet you and good to know that you are younger at '{age}' ")

#3.--------------------------------------------------------------------------------------
def get_largenum(x,y):
    return x if x>y else y

x  = 2
y = 3
print(f"\nlarger number among '{x}', '{y}' --> {get_largenum(x,y)}")

#4.--------------------------------------------------------------------------------------
mx = 3
my = 4

print (f"\n Current values of 'mx' and 'my' are {mx},{my}")

mx,my = my,mx
print (f"\n Values of 'mx' and 'my' after reversing are {mx},{my}")

#5.--------------------------------------------------------------------------------------
def get_number():
    try:
        return int(input("Please provide a number\n"))
    except:
        print("Please provide proper number")
        return None

num = None

#Ensure user provided proper number        
while num is None:
    num = get_number()

print (f"\nThe number {num} you provided is --> {'Even' if num%2==0 else 'Odd'}")

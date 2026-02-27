# --Functions 
def greet(name):
    """Function to greet a person."""
    return f"Hello, {name}!"

def power(base, exponent=2):
    """Function to calculate power of a number."""
    return base ** exponent

def get_min_max_avg(numbers):
    """Function to get minimum, maximum, and average of a list of numbers."""
    if not numbers:
        return None, None, None
    minimum = min(numbers)
    maximum = max(numbers)
    average = sum(numbers) / len(numbers)
    return minimum, maximum, average


#--Lists
fruits = [ "banana","apple", "cherry"]
print("Fruits Raw List:", fruits)
fruits.append("orange")  # Add an item
print("Fruits After Addition:", fruits)
fruits.remove("cherry")  # Remove an item
print("Fruits after removal:", fruits)
fruits.sort()  # Sort the list  
print("Fruits after sorting:", fruits)

#--Tuples
point = (3,5)
print("----------------------------------------")
print("x:", point[0], "y:", point[1])

#--Dictionaries
person = {
    "name": "Mahi",
    "age": 40,
    "city": "New York"
}

print("----------------------------------------")
print ("Name: ", person["name"])
print ("Age: ", person["age"])
person["ProgrammingLanguage"] = "python"
print("Updated Person: ", person)

#Using Functions

print("----------------------------------------")
print(greet("Mahendra"))


num = 2
print("----------------------------------------")
print("square of", num, "is", power(num))

numbers = [3, 10, 1, 6, 2, 9]
print("----------------------------------------")
print("Min-Max-Avg of nunmbers:", get_min_max_avg(numbers))
min_val, max_val, avg_val= get_min_max_avg(numbers)
print("Minimum:", min_val)
print("Maximum:", max_val)
print("Average:", avg_val)

people = { "Mahi" : 40, "Chaitu": 37, "Tanisha": 12, "Tanvika": 7}
print("----------------------------------------")
print("People Dictionary:", people)

def get_max_age(people_dict):
    """Function to get the person with the maximum age."""
    if not people_dict:
        return None
    max_person = max(people_dict, key=people_dict.get)
    return max_person, people_dict[max_person]


max_person, max_age = get_max_age(people)
print("Person with maximum age:", max_person, "Age:", max_age)



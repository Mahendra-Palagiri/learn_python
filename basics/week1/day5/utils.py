def average(numbers):
    return sum(numbers)/len(numbers)

def grade(score):
    if score >=90:
        return "A"
    elif score >80:
        return "B"
    elif score > 70:
        return "C"
    else:
        return "We can always try to be better"
    
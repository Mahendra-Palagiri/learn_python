#--------------------------------------------------------------------
# Following program is related to questions
#--------------------------------------------------------------------
# Q.1
def isprime(num):
    divisble = False
    for i in range(2,num):
        if num % i ==0 :
            divisble = True
            break

    return 'Non-Prime' if divisble else 'Prime'

print(isprime(int(input("Enter a number\n"))))

# Q.11
def reverse_list(nums):
    revlist  = []
    iterable = len(nums)
    while iterable > 0:
        revlist.append(nums[iterable-1])
        iterable = iterable-1

    return revlist

nums  = [3,4,5,6,7]
revnum = nums[::-1] # Simpler way of getting reverse
print(reverse_list(nums))

#Q. 12

mlist  = ['ra','ga','sa','ri']
mlist.remove('ra')
mlist.remove(mlist[1]) #mlist.pop(1) also does the same
print(mlist)
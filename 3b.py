# Niteesh Kulhari
# CS 515 D
# This code first prompts the user to enter a number or he can press enter to quit and it will keep asking user for input until he hits enter.
# After the input a while loop runs until the user hits enter and keep adding the numbers.

# Final wage = (Wage*normal hours) + (Overtime hours * (1.5 * Normal wage)

# Variable sum used to add the numbers and count is to keep track of the number of inputs.
sum = 0
count = 0

# Logic to calculate the sum
while True:
    num = input("Enter a number or hit enter to quit: ")
    if num == "":
        break
    sum += float(num)
    count += 1


# Printing the sum
print("The sum of the numbers is: " , sum)

# If there is no input from the user in that100 case average will not be calculated.
if count>0:
    average = sum/count
    print("The average of the number is: " , end="")
    print("%.2f"% average)
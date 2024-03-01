#!/usr/bin/env python
# coding: utf-8

# # Part -1

# Q1) Create a function to add two 3x3 matrices. Each matrix is represented as a list of lists. The function takes two parameters and returns the computed matrix. Do not use numpy for computation.

# In[59]:


def add_matrixes(matrix_a, matrix_b):

    result = [[0, 0, 0],
              [0, 0, 0],
              [0, 0, 0]]

    for i in range(len(matrix_a)):
        for j in range(len(matrix_a[0])):
            result[i][j] = matrix_a[i][j] + matrix_b[i][j]
    return result
    
        #print (matrix_a[i][j], matrix_b[i][j])


X = [[12, 7, 3],
     [4, 5, 6],
     [7, 8, 9]]

Y = [[5, 8, 1],
     [6, 7, 3],
     [4, 5, 9]]

result = add_matrixes(X, Y)
for r in result:
        print(r)


# In[ ]:


Q2) Now solve the above question using numpy 


# In[60]:


import numpy as np

X = [[12, 7, 3],
     [4, 5, 6],
     [7, 8, 9]]

Y = [[5, 8, 1],
     [6, 7, 3],
     [4, 5, 9]]

output_array = np.add(X, Y)
print(output_array)


# In[ ]:


Q3)  Create a function to multiply two 3x3 matrices. Each matrix is represented as a list of lists. The function takes 
two parameters and returns the computed matrix. Do not use numpy for computation.


# In[61]:


def multiply_matrixes(matrix_a, matrix_b):

    result = [[0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0]]

    for i in range(len(matrix_a)):
        for j in range(len(matrix_b[0])):
            for k in range(len(matrix_b)):
                result[i][j] += matrix_a[i][k] * matrix_b[k][j]     

    return result

X = [[12,7,3],
    [4 ,5,6],
    [7 ,8,9]]

Y = [[5,8,1,2],
    [6,7,3,0],
    [4,5,9,1]]

result = multiply_matrixes(X, Y)
for r in result:
        print(r)


#  Q4) Now solve the above question using numpy

# In[62]:


X = [[12,7,3],
    [4 ,5,6],
    [7 ,8,9]]

Y = [[5,8,1,2],
    [6,7,3,0],
    [4,5,9,1]]

output_array = np.dot(X,Y)
print(output_array)


#  Q5) Create a function to find transpose of a matrices. Matrix is represented as a list of lists. The function takes one parameters and returns the computed matrix. Do not use numpy for computation.

# In[63]:


def transpose_matrix(matrix_a):
    
    result = [[0, 0, 0],
              [0, 0, 0]]
    for i in range(len(matrix_a)):
        for j in range(len(matrix_a[0])):
            result[j][i] = matrix_a[i][j]
    
    return result

X = [[12,7],
    [4 ,5],
    [3 ,8]]
#print(len(X))

result = transpose_matrix(X)
for r in result:
        print(r)


# Q7) Create a function to find inverse of a matrix. Matrix is represented as a list of lists. The function takes one parameters and returns the computed matrix. You can use numpy. 

# In[64]:


def inverse_matrix(matrix_a):
    return np.linalg.inv(matrix_a)

X= [[3, 7],
    [2, 5]]

result = inverse_matrix(X)

for r in result:
    print(r)


# # Part 2

# In[65]:


import numpy as np
import matplotlib.pyplot as mp
import pandas as pa


# In[66]:


df = pa.read_csv("salary_data.csv")


# In[67]:


df.head()


# In[68]:


df.plot.scatter(x = 'YearsExperience', y = 'Salary' )


# In[69]:


# Collecting X and Y
X = df['YearsExperience'].values
Y = df['Salary'].values
# Calculating theta0, theta1

# Mean X and Y
X_mean = np.mean(X)
Y_mean = np.mean(Y)
 

numer = 0
denom = 0

for i in range(len(X)):
  numer += (X[i] - X_mean) * (Y[i] - Y_mean)
  denom += (X[i] - X_mean) ** 2
  θ_1 = numer / denom
  θ_0 = Y_mean - (θ_1 * X_mean)


# In[70]:


X = df['YearsExperience'].values
Y = df['Salary'].values
df.plot.scatter(x = 'YearsExperience', y = 'Salary' )
mp.plot(X, θ_1*X + θ_0, color="red")


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import pandas as pd


# In[57]:


column_list_ratings = ["UserID", "MovieID", "Ratings", "Timestamp"]
ratings_data = pd.read_csv('ratings.dat', sep='::', names = column_list_ratings, engine='python')
column_list_movies = ["MovieID", "Title", "Genre"]
movies_data = pd.read_csv("movies.dat", sep='::', names = column_list_movies, engine='python', encoding='latin')
column_list_users = ["UserID", "gender", "Age", "Occupation", "Zixp-code"]
user_data = pd.read_csv("users.dat", sep='::', names = column_list_users, engine='python')

ratings_data


# In[21]:


data = pd.merge(pd.merge(ratings_data,user_data),movies_data)
data


# In[15]:


mean_ratings = data.pivot_table('Ratings','Title', aggfunc='mean')
mean_ratings


# In[23]:


mean_ratings = data.pivot_table('Ratings', index=['Title'],aggfunc='mean')
top_15_mean_ratings = mean_ratings.sort_values(by='Ratings', ascending = False).head(15)
top_15_mean_ratings


# In[25]:


mean_ratings=data.pivot_table('Ratings', index=['Title'], columns=['gender'], aggfunc='mean')
mean_ratings


# In[29]:


data = pd.merge(pd.merge(ratings_data,user_data),movies_data)

mean_ratings = data.pivot_table('Ratings', index=['Title'], columns=['gender'], aggfunc='mean')
top_femal_ratings = mean_ratings.sort_values(by='F', ascending=False)
print(top_femal_ratings.head(15))

top_male_ratings = mean_ratings.sort_values(by='M', ascending=False)
print(top_male_ratings.head(15))


# In[30]:


mean_ratings['diff'] = mean_ratings['M'] - mean_ratings['F']
sorted_by_diff = mean_ratings.sort_values(by='diff')
sorted_by_diff[:10]


# In[33]:


ratings_by_title = data.groupby('Title').size()
ratings_by_title.sort_values(ascending=False).head(10)


# In[49]:


ratings_by_title = data.groupby('Title').size()
popular_titles = ratings_by_title[ratings_by_title>=2500]
popular_titles.sort_values(ascending=False).head(15)


# Question 1
# Create a ratings matrix using Numpy. This matrix allows us to see the ratings for a given movie and user ID. The element at location [𝑖,𝑗]
#  is a rating given by user 𝑖
#  for movie 𝑗
# . Print the shape of the matrix produced.
# 
# Additionally, choose 3 users that have rated the movie with MovieID "1377" (Batman Returns). Print these ratings, they will be used later for comparison.
# 
# Notes:
# 
# Do not use pivot_table.
# A ratings matrix is not the same as ratings_data from above.
# The ratings of movie with MovieID 𝑖
#  are stored in the (𝑖
# -1)th column (index starts from 0)
# Not every user has rated every movie. Missing entries should be set to 0 for now.
# If you're stuck, you might want to look into np.zeros and how to use it to create a matrix of the desired shape.
# Every review lies between 1 and 5, and thus fits within a uint8 datatype, which you can specify to numpy.

# In[159]:


user_ids = ratings_data['UserID'].unique()
movie_ids = ratings_data['MovieID'].unique()

ratings_matrix = np.zeros((len(user_ids), len(movie_ids)), dtype = np.uint8)

for index, row in ratings_data.iterrows():
    user_id = row['UserID']
    movie_id = row['MovieID']
    rating = row['Ratings']
    user_idx = np.where(user_ids == user_id)[0][0]
    movie_idx = np.where(movie_ids == movie_id)[0][0]
    ratings_matrix[user_idx, movie_idx] = rating


print(ratings_matrix.shape)
print(ratings_matrix)
print(user_ids)


# In[128]:


movie_id = 1377
user_ids_1377 = ratings_data[ratings_data['MovieID'] == movie_id]['UserID'].head(3).values
user_ids_1377
movie_idx = np.where(movie_ids == movie_id)[0][0]
ratings_1377 = ratings_matrix[user_ids_1377-1, movie_idx]
print(ratings_1377)
print(user_ids_1377)


# ###Question 2
# 
# Normalize the ratings matrix (created in Question 1) using Z-score normalization. While we can't use sklearn's StandardScaler for this step, we can do the statistical calculations ourselves to normalize the data.
# 
# Before you start:
# 
# Your first step should be to get the average of every column of the ratings matrix (we want an average by title, not by user!).
# 
# Make sure that the mean is calculated considering only non-zero elements. If there is a movie which is rated only by 10 users, we get its mean rating using (sum of the 10 ratings)/10 and NOT (sum of 10 ratings)/(total number of users)
# 
# All of the missing values in the dataset should be replaced with the average rating for the given movie. This is a complex topic, but for our case replacing empty values with the mean will make it so that the absence of a rating doesn't affect the overall average, and it provides an "expected value" which is useful for computing correlations and recommendations in later steps.
# 
# In our matrix, 0 represents a missing rating.
# 
# Next, we want to subtract the average from the original ratings thus allowing us to get a mean of 0 in every column. It may be very close but not exactly zero because of the limited precision floats allow.
# 
# Lastly, divide this by the standard deviation of the column.
# 
# Not every MovieID is used, leading to zero columns. This will cause a divide by zero error when normalizing the matrix. Simply replace any NaN values in your normalized matrix with 0.

# In[60]:


col_mean = np.nanmean(ratings_matrix, axis=0)
col_mean


# In[61]:


col_mean = np.mean(ratings_matrix, axis=0)
col_mean


# In[63]:


column_mean = np.ma.masked_equal(ratings_matrix, 0).mean(axis=0)
col_mean


# In[91]:


#rating_matrix_1 = np.where(ratings_matrix==0,np.nan,ratings_matrix)
#rating_matrix_1
num_users = ratings_data['UserID'].nunique()
num_movies = ratings_data['MovieID'].nunique()


















mean_ratings = np.zeros((num_movies))
for j in range(num_movies):
    idx = np.where(ratings_matrix[:, j] != 0)
    mean_ratings[j] = np.mean(ratings_matrix[idx, j])
    
# Replace missing values with the mean rating for the given movie
for i in range(num_users):
    for j in range(num_movies):
        if ratings_matrix[i, j] == 0:
            ratings_matrix[i, j] = mean_ratings[j]
print(mean_ratings.shape)            
mean_ratings 


# In[92]:


ratings_matrix


# In[96]:


std_ratings = np.zeros((num_users, num_movies))
for j in range(num_movies):
    id = np.where(ratings_matrix[:,j]!=0)
    std_ratings[id,j] = (ratings_matrix[id,j] - mean_ratings[j]) / np.std(ratings_matrix[id,j])
    
std_ratings = np.nan_to_num(std_ratings, nan=0)
std_ratings


# Question 3
# 
# We're now going to perform Singular Value Decomposition (SVD) on the normalized ratings matrix from the previous question. Perform the process using numpy, and along the way print the shapes of the 𝑈, 𝑆, and 𝑉 matrices you calculated.

# In[129]:


u, s, vT = np.linalg.svd(std_ratings)
print("Shape of U matrix:", u.shape)
print("Shape of S matrix:", s.shape)
print("Shape of V matrix:", vT.shape)


# Question 4
# 
# Reconstruct four rank-k rating matrix 𝑅𝑘
# , where 𝑅𝑘=𝑈𝑘𝑆𝑘𝑉𝑇𝑘
#  for k = [100, 1000, 2000, 3000]. Using each of 𝑅𝑘
#  make predictions for the 3 users selected in Question 1, for the movie with ID 1377 (Batman Returns). Compare the original ratings with the predicted ratings.

# In[212]:


import numpy as np

# Reconstruct four rank-k rating matrices and make predictions for the 3 selected users for movie 1377

k_values = [100, 1000, 2000, 3000]

for k in k_values:
    Rk = np.dot(np.dot(u[:, :k], np.diag(s[:k])), vT[:k, :])
    Rk[Rk < 0] = 0  # replace negative values with 0
    
    print(f"\nReconstructed rating matrix R{k} shape: {Rk.shape}")
    predicted_ratings = []
    
    for user in users:
        user_idx = np.where(user_ids == user)[0][0]
        movie_idx = np.where(movie_ids == 1377)[0][0]
        original_rating = ratings[user_idx, movie_idx]
        predicted_rating = Rk[user_idx, movie_idx]
        predicted_ratings.append(predicted_rating)
        
        print(f"User {user}, movie 1377: Original rating = {original_rating:.2f}, Predicted rating = {predicted_rating:.2f}")
    
    mse = np.mean((np.array(predicted_ratings) - np.array(original_ratings)) ** 2)
    print(f"\nMean squared error for R{k} = {mse:.2f}")


# In[182]:


from sklearn.metrics.pairwise import cosine_similarity
def top_movie_similarity(data, movie_id, top_n=5):
    # Get the movie vector
    movie_vector = data[:, movie_id-1]
    # Compute the cosine similarity between the movie vector and all other movie vectors
    similarities = np.nan_to_num(cosine_similarity(data.T, movie_vector.reshape(1, -1)).flatten())
    # Sort the similarities in descending order
    similar_indices = np.argsort(similarities)[::-1][1:top_n+1]
    # Return the top n similar movies
    return similar_indices

def print_similar_movies(movie_titles, top_indices):
    print('Most similar movies: ')
    for i, index in enumerate(top_indices):
        print(f'{i+1}: {movie_titles[index]}')

# Print the top 5 movies for Batman Returns
movie_id = 1377
top_n = 5
movie_titles = movies_data["Title"].tolist()

similar_movies = top_movie_similarity(Rk_1000, movie_id, top_n)
print_similar_movies(movie_titles, similar_movies)


# In[ ]:





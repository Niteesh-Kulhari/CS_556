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


# # Question 1
# 
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

# In[356]:


# Extracting all the unique iser_ids and movie_ids from the data
user_ids = ratings_data['UserID'].unique()
movie_ids = ratings_data['MovieID'].unique()

#Rating matrix filled with zero's with dimension of user_ids, movie_ids
ratings_matrix = np.zeros((len(user_ids), len(movie_ids)), dtype = np.uint8)

# Populating the rating matrix with the rating given by a user to a movie
for index, row in ratings_data.iterrows():
    user_id = row['UserID']
    movie_id = row['MovieID']
    rating = row['Ratings']
    user_idx = np.where(user_ids == user_id)[0][0]
    movie_idx = np.where(movie_ids == movie_id)[0][0]
    ratings_matrix[user_idx, movie_idx] = rating

# Printing the shape of ratings_matrix
print(ratings_matrix.shape)

# Printing the ratings_matrix
print(ratings_matrix)


# In[250]:


# Movie id as given in question
movie_id = 1377

# Extracting user_id for the given movie_id
user_ids_1377 = ratings_data[ratings_data['MovieID'] == movie_id]['UserID'].head(3).values

# Extracting the column number which corresponds to movie_id ->1377
movie_idx = np.where(movie_ids == movie_id)[0][0]

# Printing the rating given by every user to the movie
for user_1 in user_ids_1377:
    ratings_1377 = ratings_matrix[user_1-1, movie_idx]
    print("User",user_1, "Rated the movie: ",ratings_1377 )


# # Question 2
# 
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

# In[374]:


#rating_matrix_1 = np.where(ratings_matrix==0,np.nan,ratings_matrix)
#rating_matrix_1

# number of users and movie id's
num_users = ratings_data['UserID'].nunique()
num_movies = ratings_data['MovieID'].nunique()

# Calculating mean rating of every column
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


# # normalized

# #####  In this first i have replaced every 0 to nan value and then calculated mean of evrey column and then replaced all the nan with the mean.
# #####  Then subtracted mean from original matrix.
# #####  Finally to calculate normalized rating divided normalized rating from std rating and replaced any nan to 0.

# In[437]:


# Replacing every 0 to NAN 
ratings_nan = np.where(ratings_matrix==0,np.nan,ratings_matrix)
ratings_nan

# Calculating mean of the column ignoring the NAN
ratings_mean = np.nanmean(ratings_nan, axis=0)
ratings_mean

# Replacing NAN with mean of every column with the mean
ratings_nan = np.where(np.isnan(ratings_nan),ratings_mean,ratings_nan)
ratings_nan

# Subtracting mean from every column of ratings
normalized_rating = ratings_nan - ratings_mean
normalized_rating

# Standard deviation
std_ratings = np.nanstd(ratings_nan, axis=0)

# Calculating normalized rating by dividing normalized ratings to std ratings
normalized_rating = np.divide(normalized_rating,std_ratings,
                             out = np.zeros_like(normalized_rating), where=std_ratings!=0)

# Replacing every NAN after division to 0
normalized_rating = np.nan_to_num(normalized_rating)


# # Question 3
# 
# We're now going to perform Singular Value Decomposition (SVD) on the normalized ratings matrix from the previous question. Perform the process using numpy, and along the way print the shapes of the 𝑈, 𝑆, and 𝑉 matrices you calculated.

# In[438]:


#u, s, vT = np.linalg.svd(std_ratings)
#print("Shape of U matrix:", u.shape)
#print("Shape of S matrix:", s.shape)
#print("Shape of V matrix:", vT.shape)

# Calculating SVD Matrix 
u,s,vT = np.linalg.svd(normalized_rating)
print("Shape of U matrix:", u.shape)
print("Shape of S matrix:", s.shape)
print("Shape of V matrix:", vT.shape)


# # Question 4
# 
# Reconstruct four rank-k rating matrix 𝑅𝑘
# , where 𝑅𝑘=𝑈𝑘𝑆𝑘𝑉𝑇𝑘
#  for k = [100, 1000, 2000, 3000]. Using each of 𝑅𝑘
#  make predictions for the 3 users selected in Question 1, for the movie with ID 1377 (Batman Returns). Compare the original ratings with the predicted ratings.

# ### Prediction using svd matrix obtained by normalized rating in the above question

# In[424]:


# Rank values as given in question
k_values = [100, 1000, 2000, 3000]

# Loop over the k values
for k in k_values:
    # Reconstruct the rank-k matrix
    R_k = u[:, :k] @ np.diag(s[:k]) @ vh[:k, :]
    # Storing Rank 1000 matrix to use in later questions
    if k == 1000:
        Rk_1000 = R_k
        
    # Make predictions for the selected users and movie
    print(f"\nReconstructed rating matrix R{k} shape: {R_k.shape}")
    for user_id in [10,13,18]:
        user_idx = user_id-1
        movie_idx = 1376  # index starts from 0
        

        # Get the original rating
        rating_original = ratings_data[(ratings_data["UserID"] == user_id) & 
                                        (ratings_data["MovieID"] == 1377)]["Ratings"].values[0]
        
        # Get the predicted rating
        rating_predicted = R_k[user_idx, movie_idx]
        
        # Print the original and predicted ratings
        print("For k = {}, User {}'s original rating: {}, predicted rating: {:.2f}".format(k, user_id, rating_original, rating_predicted))


# # ------------------------------------------------------------------------

# ###  As seen above when i make ranked matrix suing the svd matrix that was created using the normailed_rating in question 2 my predicted rating comes out -ve
# 
# ### But when i do the prediction using the original rating_matrix which is done below my predicted ratings are +ve i was not able to rectify this part

# In[441]:


from numpy.linalg import eig

A=ratings_matrix
print("The shape of ratings matrix:  ",A.shape)


U,S,VT=np.linalg.svd(A)
print('U matrix shape:  ',U.shape)
print('S matrix shape:  ',S.shape)
print('VT matrix shape:  ',VT.shape)


k=[100,1000,2000,3000]

#Constructing lower rank prediction matrices
for i in range(len(k)):
    Rk=U[:,:k[i]]@np.diag(S[:k[i]])@VT[:k[i],:]
    if (i==0):
        R_100=Rk
    elif i==1:
        R_1000=Rk
    elif i==2:
        R_2000=Rk
    elif i==3:
        R_3000=Rk

    for user_id in user_ids_1377:
        user_idx = user_id-1
        movie_idx = 1376  # index starts from 0
        
        rating_original = ratings_data[(ratings_data["UserID"] == user_id) & 
                                        (ratings_data["MovieID"] == 1377)]["Ratings"].values[0]
        predicted_rating = Rk[user_idx][movie_idx]
        
        print("For k = {}, User {}'s original rating: {}, predicted rating: {}".format(k[i], user_id, rating_original, predicted_rating))
        


# # ----------------------------------------------------------------------------------

# In[323]:


print(Rk_1000.shape)


# # Question 5

# In[451]:


def top_movie_similarity(data, movie_id, top_n=5):
    # Get the column index for the given movie_id
    movie_idx = movie_id - 1
    
    # Get the column vector for the given movie
    vector_movie = data[:, movie_idx]
    
    # Calculate the cosine similarity between the given movie and all other movies
    simulated_scores = np.nan_to_num(np.dot(data.T, vector_movie) / (np.linalg.norm(data, axis=0) * np.linalg.norm(vector_movie)))
    
    # Sort the movies based on their similarity with the given movie
    top_indices = np.argsort(-simulated_scores)[:top_n]
    top_scores = simulated_scores[top_indices]
    
    return top_indices, top_scores

# function to print the similar movies
def print_similar_movies(movie_titles, top_indices):
    print('Most Similar movies: ')
    for i, idx in enumerate(top_indices):
        print(f'{i+1}. {movie_titles[idx]}')
        
# Find the top 5 similar movies
movie_id = 1377
top_n = 5
top_indices, top_scores = top_movie_similarity(Rk_1000, movie_id, top_n)

movie_titles = movies_data["Title"].tolist()
#movie_titles

# Print the top 5 similar movies
print_similar_movies(movie_titles, top_indices)


# # Question 6

# In[454]:


# Calculate cosine similarity between user 5954 and all other users
def top_user_similarity(data, user_id):
    # Get row of user 5954
    user_row = data[user_id-1]
    # Calculate cosine similarity between user 5954 and all other users
    sim_scores = np.dot(data, user_row)/(np.linalg.norm(data, axis=1)*np.linalg.norm(user_row))
    # Sort similarity scores in descending order
    sorted_scores = np.argsort(sim_scores)[::-1]
    # Return the most similar user
    return sorted_scores[1]

# Find top movie recommendations for user 5954
def get_movie_recommendations(data, user_id, top_n=5):
    # Find most similar user
    similar_user = top_user_similarity(data, user_id)
    # Get rows of user 5954 and most similar user
    print(user)
    user_row = data[user_id-1]
    similar_user_row = data[similar_user]
    # Find movies that similar user rated highly but user 5954 has not seen
    unseen_movies = np.where(user_row == 0)[0]
    similar_user_ratings = similar_user_row[unseen_movies]
    # Sort unseen movies by rating from similar user in descending order
    sorted_movies = np.argsort(similar_user_ratings)[::-1]
    # Return top n movie recommendations
    return unseen_movies[sorted_movies][:top_n]

# Get movie recommendations for user 5954
user_id = 5954
#user = top_user_similarity(Rk_1000, user_id)
recommendations = get_movie_recommendations(Rk_1000, user_id)
#print(user)
for movie in recommendations:
    movie_id = movie+1
    print(movies_data[movies_data['MovieID']==movie_id]['Title'].values[0])
        


# In[ ]:





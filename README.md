# Netflix Movie Recommendation system

# 1. Business Problem
## 1.1 Problem Description

Netflix is all about connecting people to the movies they love. To help customers find those movies, they developed world-class movie recommendation system: CinematchSM. Its job is to predict whether someone will enjoy a movie based on how much they liked or disliked other movies. Netflix use those predictions to make personal movie recommendations based on each customer’s unique tastes. And while Cinematch is doing pretty well, it can always be made better.

Now there are a lot of interesting alternative approaches to how Cinematch works that netflix haven’t tried. Some are described in the literature, some aren’t. We’re curious whether any of these can beat Cinematch by making better predictions. Because, frankly, if there is a much better approach it could make a big difference to our customers and our business.

Credits: https://www.netflixprize.com/rules.html
## 1.2 Problem Statement

Netflix provided a lot of anonymous rating data, and a prediction accuracy bar that is 10% better than what Cinematch can do on the same training data set. (Accuracy is a measurement of how closely predicted ratings of movies match subsequent actual ratings.)
## 1.3 Sources

    https://www.netflixprize.com/rules.html
    https://www.kaggle.com/netflix-inc/netflix-prize-data
    Netflix blog: https://medium.com/netflix-techblog/netflix-recommendations-beyond-the-5-stars-part-1-55838468f429 (very nice blog)
    surprise library: http://surpriselib.com/ (we use many models from this library)
    surprise library doc: http://surprise.readthedocs.io/en/stable/getting_started.html (we use many models from this library)
    installing surprise: https://github.com/NicolasHug/Surprise#installation
    Research paper: http://courses.ischool.berkeley.edu/i290-dm/s11/SECURE/a1-koren.pdf (most of our work was inspired by this paper)
    SVD Decomposition : https://www.youtube.com/watch?v=P5mlg91as1c

## 1.4 Real world/Business Objectives and constraints

Objectives:

    Predict the rating that a user would give to a movie that he ahs not yet rated.
    Minimize the difference between predicted and actual rating (RMSE and MAPE)

Constraints:

    Some form of interpretability.

# 2. Machine Learning Problem
## 2.1 Data
### 2.1.1 Data Overview

Get the data from : https://www.kaggle.com/netflix-inc/netflix-prize-data/data

Data files :

    combined_data_1.txt

    combined_data_2.txt

    combined_data_3.txt

    combined_data_4.txt

    movie_titles.csv
    </ul> 

  
The first line of each file [combined_data_1.txt, combined_data_2.txt, combined_data_3.txt, combined_data_4.txt] contains the movie id followed by a colon. Each subsequent line in the file corresponds to a rating from a customer and its date in the following format:

CustomerID,Rating,Date

MovieIDs range from 1 to 17770 sequentially.
CustomerIDs range from 1 to 2649429, with gaps. There are 480189 users.
Ratings are on a five star (integral) scale from 1 to 5.
Dates have the format YYYY-MM-DD.


## 2.2 Mapping the real world problem to a Machine Learning Problem
### 2.2.1 Type of Machine Learning Problem

For a given movie and user we need to predict the rating would be given by him/her to the movie. 
The given problem is a Recommendation problem 
It can also seen as a Regression problem 

### 2.2.2 Performance metric

    1.Mean Absolute Percentage Error: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error
    2.Root Mean Square Error: https://en.wikipedia.org/wiki/Root-mean-square_deviation

### 2.2.3 Machine Learning Objective and Constraints

    1.Minimize RMSE.
    2.Try to provide some interpretability.

# 3. Exploratory Data Analysis
## 3.1 Preprocessing
## 3.2 Spliting data into Train and Test(80:20) 
## 3.3 Exploratory Data Analysis on Train data 
### 3.3.1 Distribution of ratings 
### 3.3.2 Number of Ratings per a month
### 3.3.3 Analysis on the Ratings given by user
### 3.3.4 Analysis of ratings of a movie given by a user 

### 3.3.5 Number of ratings on each day of the week
### 3.3.6 Creating sparse matrix from data frame 
### 3.3.7 Finding Global average of all movie ratings, Average rating per user, and Average rating per movie
### 3.3.8 Cold Start problem
## 3.4 Computing Similarity matrices
### 3.4.1 Computing User-User Similarity matrix

    Calculating User User Similarity_Matrix is not very easy(unless you have huge Computing Power and lots of time) because of number of. usersbeing lare.
        You can try if you want to. Your system could crash or the program stops with Memory Error

### 3.4.1.1 Trying with all dimensions (17k dimensions per user)

### 3.4.1.2 Trying with reduced dimensions (Using TruncatedSVD for dimensionality reduction of user vector)

   
### 3.4.2 Computing Movie-Movie Similarity matrix


    Even though we have similarity measure of each movie, with all other movies, We generally don't care much about least similar movies.

    Most of the times, only top_xxx similar items matters. It may be 10 or 100.

    We take only those top similar movie ratings and store them in a saperate dictionary.


### 3.4.3 Finding most similar movies using similarity matrix

Does Similarity really works as the way we expected...?
Let's pick some random movie and check for its similar movies.
# 4. Machine Learning Models 

## 4.1 Sampling Data
### 4.1.1 Build sample train data from the train data
### 4.1.2 Build sample test data from the test data4.2 Finding Global Average of all movie ratings, Average rating per User, and Average rating per Movie (from sampled train)
## 4.2 Finding Global Average of all movie ratings, Average rating per User, and Average rating per Movie (from sampled train)
## 4.3 Featurizing data 

### 4.3.1 Featurizing data for regression problem
### 4.3.1.1 Featurizing train data

    GAvg : Average rating of all the ratings

    Similar users rating of this movie:
        sur1, sur2, sur3, sur4, sur5 ( top 5 similar users who rated that movie.. )

    Similar movies rated by this user:
        smr1, smr2, smr3, smr4, smr5 ( top 5 similar movies rated by this movie.. )

    UAvg : User's Average rating

    MAvg : Average rating of this movie

    rating : Rating of this movie by this user.
### 4.3.1.2 Featurizing test data 

    GAvg : Average rating of all the ratings

    Similar users rating of this movie:
        sur1, sur2, sur3, sur4, sur5 ( top 5 simiular users who rated that movie.. )

    Similar movies rated by this user:
        smr1, smr2, smr3, smr4, smr5 ( top 5 simiular movies rated by this movie.. )

    UAvg : User AVerage rating

    MAvg : Average rating of this movie

    rating : Rating of this movie by this user.
### 4.3.2 Transforming data for Surprise models

### 4.3.2.1 Transforming train data

    We can't give raw data (movie, user, rating) to train the model in Surprise library.

    They have a saperate format for TRAIN and TEST data, which will be useful for training the models like SVD, KNNBaseLineOnly....etc..,in Surprise.

    We can form the trainset from a file, or from a Pandas DataFrame. http://surprise.readthedocs.io/en/stable/getting_started.html#load-dom-dataframe-py


## 4.3.2.2 Transforming test data

    Testset is just a list of (user, movie, rating) tuples. (Order in the tuple is impotant)


## 4.4 Applying Machine Learning models

    Global dictionary that stores rmse and mape for all the models....

        It stores the metrics in a dictionary of dictionaries

            keys : model names(string)

            value: dict(key : metric, value : value )

### 4.4.1 XGBoost with initial 13 features
TEST DATA
------------------------------
RMSE :  1.0890322448240302
MAPE :  35.13968692492444
### 4.4.2 Suprise BaselineModel 



Test Data
---------------
RMSE : 1.0865215481719563

MAPE : 34.9957270093008


### 4.4.3 XGBoost with initial 13 features + Surprise Baseline predictor 
TEST DATA
------------------------------
RMSE :  1.0891181427027241
MAPE :  35.13135164276489



### 4.4.4.1 Surprise KNNBaseline with user user similarities
---------------
Test Data
---------------
RMSE : 1.0865005562678032

MAPE : 35.02325234274119

### 4.4.4.2 Surprise KNNBaseline with movie movie similarities
Test Data
---------------
RMSE : 1.0868914468761874

MAPE : 35.02725521759712


### 4.4.5 XGBoost with initial 13 features + Surprise Baseline predictor + KNNBaseline predictor

            First we will run XGBoost with predictions from both KNN's ( that uses User_User and Item_Item similarities along with our previous features.

            Then we will run XGBoost with just predictions form both knn models and preditions from our baseline model.

TEST DATA
------------------------------
RMSE :  1.088749005744821
MAPE :  35.188974153659295


### 4.4.6 Matrix Factorization Techniques
### 4.4.6.1 SVD Matrix Factorization User Movie intractions
Test Data
---------------
RMSE : 1.0860031195730506

MAPE : 34.94819349312387



### 4.4.6.2 SVD Matrix Factorization with implicit feedback from user ( user rated movies ) 



    
Test Data
---------------
RMSE : 1.0862780572420558

MAPE : 34.909882014758175

### 4.4.7 XgBoost with 13 features + Surprise Baseline + Surprise KNNbaseline + MF Techniques
TEST DATA
------------------------------
RMSE :  1.0891599523508655
MAPE :  35.12646240961147

### 4.4.8 XgBoost with Surprise Baseline + Surprise KNNbaseline + MF Techniques 

TEST DATA
------------------------------
RMSE :  1.095123189648495
MAPE :  35.54329712868095

## 4.5 Comparision between all models 
 MODEL          |  RMSE
 ---------------|----------
svd             |     1.0860031195730506
svdpp           |    1.0862780572420558
bsl_algo        |   1.0868914468761874
knn_bsl_u       |   1.0865005562678032
knn_bsl_m       |   1.0868914468761874
xgb_knn_bsl     |   1.088749005744821
xgb_final       |   1.0891599523508655
xgb_bsl         |    1.0891599523508655
first_algo      |   1.0890322448240302
xgb_all_models  |   1.095123189648495


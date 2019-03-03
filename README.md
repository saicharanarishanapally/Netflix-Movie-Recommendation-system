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

### 2.1.2 Example Data point

1:
1488844,3,2005-09-06
822109,5,2005-05-13
885013,4,2005-10-19
30878,4,2005-12-26
823519,3,2004-05-03
893988,3,2005-11-17
124105,4,2004-08-05
1248029,3,2004-04-22
1842128,4,2004-05-09
2238063,3,2005-05-11
1503895,4,2005-05-19
2207774,5,2005-06-06
2590061,3,2004-08-12
2442,3,2004-04-14
543865,4,2004-05-28
1209119,4,2004-03-23
804919,4,2004-06-10
1086807,3,2004-12-28
1711859,4,2005-05-08
372233,5,2005-11-23
1080361,3,2005-03-28
1245640,3,2005-12-19
558634,4,2004-12-14
2165002,4,2004-04-06
1181550,3,2004-02-01
1227322,4,2004-02-06
427928,4,2004-02-26
814701,5,2005-09-29
808731,4,2005-10-31
662870,5,2005-08-24
337541,5,2005-03-23
786312,3,2004-11-16
1133214,4,2004-03-07
1537427,4,2004-03-29
1209954,5,2005-05-09
2381599,3,2005-09-12
525356,2,2004-07-11
1910569,4,2004-04-12
2263586,4,2004-08-20
2421815,2,2004-02-26
1009622,1,2005-01-19
1481961,2,2005-05-24
401047,4,2005-06-03
2179073,3,2004-08-29
1434636,3,2004-05-01
93986,5,2005-10-06
1308744,5,2005-10-29
2647871,4,2005-12-30
1905581,5,2005-08-16
2508819,3,2004-05-18
1578279,1,2005-05-19
1159695,4,2005-02-15
2588432,3,2005-03-31
2423091,3,2005-09-12
470232,4,2004-04-08
2148699,2,2004-06-05
1342007,3,2004-07-16
466135,4,2004-07-13
2472440,3,2005-08-13
1283744,3,2004-04-17
1927580,4,2004-11-08
716874,5,2005-05-06
4326,4,2005-10-29

## 2.2 Mapping the real world problem to a Machine Learning Problem
### ### 2.2.1 Type of Machine Learning Problem

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


    It is very skewed.. just like nunmber of ratings given per user.

- There are some movies (which are very popular) which are rated by huge number of users.

- But most of the movies(like 90%) got some hundereds of ratings.

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

    We have 405,041 users in out training set and computing similarities between them..( 17K dimensional vector..) is time consuming..

    From above plot, It took roughly 8.88 sec for computing simlilar users for one user

    We have 405,041 users with us in training set.

    ${ 405041 \times 8.88 = 3596764.08 \sec } = 59946.068 \min = 999.101133333 \text{ hours} = 41.629213889 \text{ days}...$
        Even if we run on 4 cores parallelly (a typical system now a days), It will still take almost 10 and 1/2 days.

    IDEA: Instead, we will try to reduce the dimentsions using SVD, so that it might speed up the process...

Here,

    $\sum \longleftarrow$ (netflix_svd.singular_values_ )

    $\bigvee^T \longleftarrow$ (netflix_svd.components_)

    $\bigcup$ is not returned. instead Projection_of_X onto the new vectorspace is returned.

    It uses randomized svd internally, which returns All 3 of them saperately. Use that instead.
    

    I think 500 dimensions is good enough

    By just taking (20 to 30) latent factors, explained variance that we could get is 20 %.

    To take it to 60%, we have to take almost 400 latent factors. It is not fare.

    It basically is the gain of variance explained, if we add one additional latent factor to it.

    By adding one by one latent factore too it, the _gain in expained variance with that addition is decreasing. (Obviously, because they are sorted that way).
    LHS Graph:
        x --- ( No of latent factos ),
        y --- ( The variance explained by taking x latent factors)

    More decrease in the line (RHS graph) :
        We are getting more expained variance than before.
    Less decrease in that line (RHS graph) :
        We are not getting benifitted from adding latent factor furthur. This is what is shown in the plots.

    RHS Graph:
        x --- ( No of latent factors ),
        y --- ( Gain n Expl_Var by taking one additional latent factor)



: This is taking more time for each user than Original one.

    from above plot, It took almost 12.18 for computing simlilar users for one user

    We have 405041 users with us in training set.

    ${ 405041 \times 12.18 ==== 4933399.38 \sec } ==== 82223.323 \min ==== 1370.388716667 \text{ hours} ==== 57.099529861 \text{ days}...$
        Even we run on 4 cores parallelly (a typical system now a days), It will still take almost (14 - 15) days.

    Why did this happen...??

- Just think about it. It's not that difficult.

---------------------------------( sparse & dense..................get it ?? )-----------------------------------

Is there any other way to compute user user similarity..??

-An alternative is to compute similar users for a particular user, whenenver required (ie., Run time)

- We maintain a binary Vector for users, which tells us whether we already computed or not..
- ***If not*** : 
    - Compute top (let's just say, 1000) most similar users for this given user, and add this to our datastructure, so that we can just access it(similar users) without recomputing it again.
    - 
- ***If It is already Computed***:
    - Just get it directly from our datastructure, which has that information.
    - In production time, We might have to recompute similarities, if it is computed a long time ago. Because user preferences changes over time. If we could maintain some kind of Timer, which when expires, we have to update it ( recompute it ). 
    - 
- ***Which datastructure to use:***
    - It is purely implementation dependant. 
    - One simple method is to maintain a **Dictionary Of Dictionaries**.
        - 
        - **key    :** _userid_ 
        - __value__: _Again a dictionary_
            - __key__  : _Similar User_
            - __value__: _Similarity Value_

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
4.4.2 Suprise BaselineModel 


Predicted_rating : ( baseline prediction )

-  http://surprise.readthedocs.io/en/stable/basic_algorithms.html#surprise.prediction_algorithms.baseline_only.BaselineOnly 

    $ \large {\hat{r}_{ui} = b_{ui} =\mu + b_u + b_i} $

    $\pmb \mu $ : Average of all trainings in training data.
    $\pmb b_u$ : User bias
    $\pmb b_i$ : Item bias (movie biases)

Optimization function ( Least Squares Problem )

- http://surprise.readthedocs.io/en/stable/prediction_algorithms.html#baselines-estimates-configuration 

    $ \large \sum_{r_{ui} \in R_{train}} \left(r_{ui} - (\mu + b_u + b_i)\right)^2 + \lambda \left(b_u^2 + b_i^2 \right).\text { [mimimize } {b_u, b_i]}$

Test Data
---------------
RMSE : 1.0865215481719563

MAPE : 34.9957270093008


### 4.4.3 XGBoost with initial 13 features + Surprise Baseline predictor 
TEST DATA
------------------------------
RMSE :  1.0891181427027241
MAPE :  35.13135164276489


### 4.4.4 Surprise KNNBaseline predictor


    KNN BASELINE
        http://surprise.readthedocs.io/en/stable/knn_inspired.html#surprise.prediction_algorithms.knns.KNNBaseline

    PEARSON_BASELINE SIMILARITY
        http://surprise.readthedocs.io/en/stable/similarities.html#surprise.similarities.pearson_baseline

    SHRINKAGE
        2.2 Neighborhood Models in http://courses.ischool.berkeley.edu/i290-dm/s11/SECURE/a1-koren.pdf

    predicted Rating : ( based on User-User similarity )

$$\begin{align} \hat{r}_{ui} = b_{ui} + \frac{ \sum\limits_{v \in N^k_i(u)} \text{sim}(u, v) \cdot (r_{vi} - b_{vi})} {\sum\limits_{v \in N^k_i(u)} \text{sim}(u, v)} \end{align}$$

    $\pmb{b_{ui}}$ - Baseline prediction of (user,movie) rating

    $ \pmb {N_i^k (u)}$ - Set of K similar users (neighbours) of user (u) who rated movie(i)

    sim (u, v) - Similarity between users u and v
        Generally, it will be cosine similarity or Pearson correlation coefficient.
        But we use shrunk Pearson-baseline correlation coefficient, which is based on the pearsonBaseline similarity ( we take base line predictions instead of mean rating of user/item)

    Predicted rating ( based on Item Item similarity ): $$\begin{align} \hat{r}<em>{ui} = b</em>{ui} + \frac{ \sum\limits_{j \in N^k<em>u(i)}\text{sim}(i, j) \cdot (r</em>{uj} - b<em>{uj})} {\sum\limits</em>{j \in N^k_u(j)} \text{sim}(i, j)} \end{align}$$
        Notations follows same as above (user user based predicted rating )

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


### ### 4.4.6 Matrix Factorization Techniques
### 4.4.6.1 SVD Matrix Factorization User Movie intractions
 Predicted Rating :

- $ \large  \hat r_{ui} = \mu + b_u + b_i + q_i^Tp_u $

    - $\pmb q_i$ - Representation of item(movie) in latent factor space

    - $\pmb p_u$ - Representation of user in new latent factor space

    A BASIC MATRIX FACTORIZATION MODEL in https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf

- Optimization problem with user item interactions and regularization (to avoid overfitting)

- $\large \sum_{r_{ui} \in R_{train}} \left(r_{ui} - \hat{r}_{ui} \right)^2 +

\lambda\left(b_i^2 + b_u^2 + ||q_i||^2 + ||p_u||^2\right) $
Test Data
---------------
RMSE : 1.0860031195730506

MAPE : 34.94819349312387



### 4.4.6.2 SVD Matrix Factorization with implicit feedback from user ( user rated movies ) 



    -----> 2.5 Implicit Feedback in http://courses.ischool.berkeley.edu/i290-dm/s11/SECURE/a1-koren.pdf

- Predicted Rating :

- $ \large \hat{r}_{ui} = \mu + b_u + b_i + q_i^T\left(p_u +
|I_u|^{-\frac{1}{2}} \sum_{j \in I_u}y_j\right) $ 

    $ \pmb{I_u}$ --- the set of all items rated by user u

    $\pmb{y_j}$ --- Our new set of item factors that capture implicit ratings.

- Optimization problem with user item interactions and regularization (to avoid overfitting)

- $ \large \sum_{r_{ui} \in R_{train}} \left(r_{ui} - \hat{r}_{ui} \right)^2 +

\lambda\left(b_i^2 + b_u^2 + ||q_i||^2 + ||p_u||^2 + ||y_j||^2\right) $
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
svd               1.0860031195730506

svdpp             1.0862780572420558

bsl_algo          1.0868914468761874

knn_bsl_u         1.0865005562678032

knn_bsl_m         1.0868914468761874

xgb_knn_bsl        1.088749005744821

xgb_final         1.0891599523508655

xgb_bsl            1.0891599523508655

first_algo        1.0890322448240302

xgb_all_models     1.095123189648495


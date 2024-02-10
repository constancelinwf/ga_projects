# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Project 3: Web APIs & NLP

## Introduction
For project 3, our goal is two-fold:
1. Collecting posts from two subreddits of our choosing. We used Reddit's API for the subreddits of tea and for coffee.  
2. We then use NLP to train a classifier on which subreddit a given post came from. This is a binary classification problem.

**The Data Science Process**
- Problem Statement
- Data Scraping & Cleaning
- Data Cleaning & EDA
- Preprocessing & Modeling
- Evaluation and Conceptual Understanding
- Conclusion and Recommendations

## Problem Statement
We are helping social media platforms to better their targeted ads based on user comments for their F&B clients, by creating a tea/coffee drinker classification model. 

## Datasets
The following data sets are in data folder:
1. raw_orig_coffee_comments
2. raw_orig_coffee_threads
3. raw_orig_tea_comments
4. raw_orig_tea_threads
5. coffee_comments_clean_merged
6. tea_comments_clean_merged
7. coffee_comments_clean_merged_filtered
8. tea_comments_clean_merged_filtered

 ### Data dictionary:

thread_id: unique identifier of the thread being commented on

comment_id: unique identifier of comment

comment_text: text body of comment

comment_score: upvote score given by fellow redditors to that comment

author_name: user who comment

id: unique identifier of each thread posted
title: title of thread

score: upvote score given by fellow redditors to that thread

num_comments: total number of comments to thread

post_hint: to indicate if thread contain an image (image) or not (NaN)

self_text: text body of thread

author_name_thread: user who posted thread

url: url of thread

url_is_media: BOOLEAN - if url contain images (1) or words (0)

## Data Scraping & Cleaning (Notebook 1)

The structure of Notebook 1 is as follows:

- Part 1: Data Collection - Scrapping Subreddit Data
- Part 2. EDA : To Uncover Potential Data Issues Requiring Cleaning
- Part 3. Data Cleaning : Removing Dataframe Rows that are Not Useful/Errorneous
- Part 4. Merge Comments & Threads Dataframes
- Part 5. Save Cleaned Merged Dataframes as CSV Files 

We scrape 1,000 "hottest" threads & accompanying comments from (a) r/coffee and (b) r/tea subreddits, cleaned the scraped data and save them as separate "coffee" and "tea" datasets.

## Filtering Top 4 and Bottom 4 Comments (by comment score) across All threads (Notebook 2)

The structure of Notebook 2 is as follows:

- Part 1: EDA to Illustrate Decision Making Approach towards Establishing Filtering Criteria
- Part 2: Filtering Code Implementation
- Part 3. Save Cleaned-Filtered Dataframes as CSV Files

There are potential drawbacks of running analyses on "full" datasets, we made a decision to filter a subset of comments which is the top 4 and bottom 4 comments (by comment score) across all threads. 

## Sentiment Analysis using VADER (Notebook 3)

In this notebook, we made use of the SentimentIntensityAnalyzer for sentiment analysis of our 2 selected subreddits.

The rationale is that we want to preserve the original sentiments of the comments and if preprocessing were to be done on them i.e. stemming the words, the original sentiments will be lost. 

Based on background research, the VADER lexicon is robust enough to interpret text "as-is", hence we will put in the comments directly.

## Classification Analyses and Modelling (Notebook 4)

The structure of Notebook 4 is as follows:

- Part 1: Preprocessing the Corpus
    - Adding labels to respective subreddit comments
    - Lemmatization and Removal of Unnecessary Characters
    - Word Vectorization (includes removal of stop words)
- Part 2: Modelling
    - Naive Bayes with Count Vectorizer
    - Naive Bayes with TD-IDF Vectorizer
    - Random Forest with Count Vectorizer
    - Random Forest with TD-IDF Vectorizer
    - Gradient Boosting with Count Vectorizer
- Part 3: Choosing our Model & Deployment onto Streamlit
    - Save chosen model into a .pkl file
    - Save chosen vectorizer into a .pkl file

In part 1, we will perform removal of unnecessary characters, stop words, and lemmatization on our corpus.

In part 2, we will be using the 4 modelling algorithms for classification analyses:
1. Naive Bayes with Count Vectorizer
2. Naive Bayes with TD-IDF Vectorizer
3. Random Forest with Count Vectorizer
4. Random Forest with TD-IDF Vectorizer
5. Gradient Boosting

For each modelling algorithm, we will run the following matrices and a classification report to see how well it did for classification:
- Accuracy 
- Specificity
- Recall
- Precision
- F1-score

## Demo (Streamlit)

A demo using the chosen model is available for this project. Two additional libraries are required:

- `joblib`
- `streamlit`

The following files are available in the `demo` folder:

- `coffee.gif` and `tea.gif`: images to be displayed after the model completes a prediction
- `naive_bayes_cvec.pkl`: the file containing the fitted naive bayes model
- `cvec.pkl`: the file containing the count vectoriser with parameters used for the model

#### Running the Demo
1) Use a terminal such as command prompt or Anaconda prompt depending on the installation path of Python, change the directory to point the the `demo` folder
2) Run the line on the terminal `streamlit run demo.py`
3) When the streamlit demo is successfully running, the terminal will display the Local URL, which shows the localhost port used on the machine, and the Network URL

To allow other machines to connect to the Network URL, go to your machine's Firewall settings and allow `python.exe` to access networks. The other machines should be connected to the same network as the machine you are using to host the streamlit demo.

## Conclusion & Recommendations
### Key Takeaways:
1. Our tea / coffee drinker classification model worked reasonably well.
2. Coffee drinkers are more verbose and like to discuss equipment and the process of coffee brewing.
3. Tea drinkers are less verbose but more emotive in expression. They lean towards the brewing process focusing on the types of leaves.   
4. We would suggest for future work to train classification model on a wider range of text-based platforms
5. We would also like to incorporate analysis of images in future classification modelling.



# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Project 4: Social Impact

## <b>1. Introduction</b>
The Myers-Briggs Type Indicator (MBTI) is a personality type measure that focuses on 4 different categories i.e. Energy (Extrovert/ Introvert), Perceiving (Sensing/ Intuitive) , Judging (Thinking/ Feeling)  and Orientation (Judgement/ Perceiving). 

1. Energy is the scale of extraversion to introversion and how they direct their attention and how they derive energy i.e. from their sourroundings or from solitude. 
2. Perceiving is the preference of intaking information such as using all 5 senses or based on intuition. 
3. Judging is to categorize how one makes decision, either basing it on logic and facts vs the method of solving an issue such as through harmonious team dynamics. 
4. Lastly, Orientation is about a preference of orderly and decisive lifestyle or a more flexible type of lifestyle.

Source: [Myers Brigg](https://www.ncbi.nlm.nih.gov/books/NBK554596/#:~:text=Definition%2FIntroduction,health%20care%20professionals%2C%20particularly%20nurses.)

For project 4, our goal is three-fold:
1. Collecting text data with MBTI labels from an MBTI personality cafe and using NLP to train a classifier model to derive the MBTI personality traits (3 traits -  S/N, T/F, J/P). This is a binary classification problem.  
2. Collect text data with associated job roles from various Linkedin profiles to input in the trained classifier model to predict MBTI labels.This is get a MBTI label matched to the associated text data and job roles.
3. Using this data, we will create a matching system where a potential job seeker can input text data and an MBTI label will be derived along with a job role recommendation.

<br>

## <b>2. Problem Statement</b>
In targeting to reduce unemployment in Singapore, mid-career job-seekers form a main proportion of unemployment numbers (63% of job seekers over the age of 45 are unemployed for over a year, compared to only 36% of job seekers aged 18 to 24). 

With the increasing challenge to increase employee retention rates, ensuring that employees continue to stay in their new job roles is also critical for efficient utilisation of resources across all stakeholders (job seekers, employers, Government). Hence, there is a need to consider job satisfaction in this equation. This could be improved from better understanding job characteristics and personality traits, playing a vital role in successful job-fits.

To maintain/ further reduce unemployment rates in Singapore; we aim to use MBTI personality traits to ensure that mid-career job seekers are quickly employed with suitable roles in the tech field, and continue to stay in/ remain satisfied their new roles.

<br>

## <b>3. Datasets</b>
The following data sets are in data folder:
1. Raw data
    - mbti_8k (obtained from [Kaggle](https://www.kaggle.com/datasets/datasnaek/mbti-type))
    - presampled_100_linkedin_data
    - jobs_posts_600
    - linkedin_data_240 
2. Cleaned data
    - mbti_cleaned
    - final_balanced_cleaned_mbti
    - linkedin_cleaned
3. Optional data - sdv_testing
(refer to Notebook 1 for further elaboration)
    - presampled_100_linkedin_data
    - resulting csv files
4. iframe_figures
    - html figures produced as a result of running code from SDV library

### Data dictionary:

MBTI dataset
- type: MBTI label to a given text post
- posts: last 50 posts written by a user 
- processed_posts: posts that have undergone EDA

Linkedin dataset
- job_title: job role of a currently employed Linkedin user/ synthetically created job title
- posts: written by user or synthetically created post reflective of job title
- processed_posts: posts that have undergone EDA

<br>

## <b>4. Notebooks</b>
- Notebook 1: Data gathering & Data upsampling
- Notebook 2: Data cleaning
- Notebook 3: Data balancing for MBTI dataset
- Notebook 4: Text processing & Modelling on Balanced MBTI dataset
- Notebook 5: Using our chosen model to predict MBTI for Linkedin dataset and Recommender system

### Notebook 1: Data gathering & Data upsampling

The structure of Notebook 1 is as follows:

* Part 1: Data Collection - Importing dataset from Kaggle
* Part 2: Increasing sample size of Linkedin dataset
* Part 3: Exploring the Linkedin Job Role Distribution
* Part 4: Saving the upsampled Linkedin dataset into a new CSV file

### Notebook 2: Data cleaning

The structure of Notebook 2 is as follows:

* Part 1. EDA - To Uncover Potential Data Issues
* Part 2. Data Cleaning - Removing Dataframe Rows that are Not Useful/Errorneous
* Part 3. Save Cleaned Dataframes as new CSV Files 
* Part 4. Obtaining Insights 

### Notebook 3: Data balancing for MBTI dataset

The structure of Notebook 3 is as follows:

* Part 1: Exploring the MBTI class distribution
* Part 2: Data Oversampling & Undersampling for Imbalanced MBTI Classes
* Part 3: Saving the Balanced MBTI dataset into a new a CSV file

### Notebook 4: Text processing & Modelling on Balanced MBTI dataset

The structure of Notebook 4 is as follows:

- Part 1: Preprocessing the Corpus
    - Import cleaned MBTI dataset
    - Splitting the MBTI traits into new columns & converting them to Boolean 
- Part 2: Tokenization & Stemming
    - Creating a function to tokenize and stem the corpus
- Part 3: Using TF-IDF to vectorize the corpus
    - Train Test Split the data
    - Vectorize the corpus
        * S-N trait
        * T-f trait
        * J-P Trait
- Part 4: Modelling
    * S-N Trait Modelling Scores
    * T-F Trait Modelling Scores
    * J-P Trait Modelling Scores
    * S-N Trait Traning
    * T-F Trait Training
    * J-P Trait Training
- Part 5: Model Evaluation
    - Save chosen model into a .pkl file
    - Save chosen vectorizer into a .pkl file


In part 4, we will be using the 6 modelling algorithms for classification analyses:
1. Logistic Regression
2. Bernoulli Naive Bayes
3. Multinomial Naive Bayes
4. SVC
5. Random Forest 
6. Forward Feed Neural Network

For each modelling algorithm, we will run the following matrices and a classification report to see how well it did for classification:
- Accuracy 
- Recall
- Precision
- F1-score

Accuracy will be our main metric as our goal is to classify the MBTI of the user correctly with equal emhasis on all classes. Other metrics such as recall/ precision are not prioritised but since they are available on the classification report code.


### Notebook 5: Using our chosen model to predict MBTI for Linkedin dataset and Recommender system

The structure of Notebook 5 is as follows:

* Part 1: Importing the .pkl files for chosen model & vectorizer
* Part 2: Generating predictions/ MBTI Labels for LinkedIn dataset
* Part 3: Generating Recommender Systems: Profile-Based Recommendations


**Conclusion:**

Recap of binary class mapping:
* S = 1, N = 0
* T = 1, F = 0
* J = 1, P = 0

Based on the above results from testing different modelling algorithms, the models show us how accurately we can predict for "1". 
We will be using `accuracy` as our evaluation metric and the threshold is 0.5 i.e. below 0.5, it will be classified as "0"; above 0.5, it will be classified as "1". 

We used Multinomial Naive Bayes as our baseline because it is a model that is known to be good with classification on text data. 

We have decided to use 2 different model algorithms to predict the 3 traits. We base our decision on (a) highest possible R2 train and R2 test scores and (b) the smallest difference between the R2 train and R2 test scores (c) Difference between the R2 train and cross validation score
- S-N trait: Logistic Regression
- T-F trait: Logistic Regression
- J-P trait: Multinomial Naive Bayes

To highlight, for the T/F trait, even though the difference between R2 train and R2 test is smaller with the Multinomial Naive Bayes model, the Logistic Regression difference is just 0.004 more and the overall scores of the R2 train and test is higher so we decided to use Logistic Regression to predict for this.

For the J-P trait, although Neural Network accuracy score is the best out of all, in consideration of ease of modelling and computational power as well as the relatively high test loss, we decide to still stick to Multinomial Naive Bayes.


<br>

## <b>5. Cost Benefit Analysis</b>
(a) Cost
1. Cost of Burnout per Employee
* Direct Costs
    * According to our research, the cost of burnout could be as much as half an employee's annual salary* 
    * For an employee earning $50,000, this would be $25,000

* Productivity Loss
    * Disengaged employees, often a result of burnout, cost their employer an average of 34% of their annual salary due to lost productivity 
    * For a $50,000 salary, this amounts to $17,000
    * Burnout resulted in a productivity loss of 4.2 hours per week in Singapore

* Healthcare Costs
    * Burnout can lead to increased healthcare costs
    * More spending on employees with chronic conditions, which can be exacerbated by burnout*
    * In Singapore, cost  is estimated to be around US$2.3 billion

2. Total Cost of Burnout for the Company
* Lost Productivity
    * The U.S. lost $1.8 trillion in productivity due to corporate burnout
    * 38% of resignations in the tech industry* in Singapore are attributed to stress and burnout

* Turnover Costs
    * Replacing an employee can cost from 1.5 to 2x the employee's salary
    * For a tech company with a high incidence of burnout, these costs can be substantial
    * In Singapore, it is estimated to cost Singapore almost $12 billion*

(b) Benefits
1. Savings from Reduced Turnover
* Effective job fit assessments can reduce turnover-related expenses significantly
    *Crucial 
        * turnover costs can equal up to one-third of the employee's annual salary
        * job fit assessments reduce turnover by 29% to 59%*


2. Value from Increased Productivity
* Productivity Increase: 
    * Engaged employees are more productive
    * Conservative 20% increase in productivity for an employee earning $50,000 would equate to an additional $10,000 in value per employee.*
    * In Singapore, the actual cost and savings will vary depending on effectiveness of measures to curb burnout

<br>

## <b>6. Limitations in Data Collection</b>
1. LinkedIn Dataset consists of synthetic data as LinkedIn is highly stringent and against web scraping their user’s profiles
    * MBTI and job titles in Recommender System might not be representative of the actual MBTI population/ proportion in job titles

2. Training MBTI Classification model using posts from online cafe might be limited in labelling other text forms e.g. LinkedIn posts as language used could be different

<br>

## <b>7. Potential Benefits & Future Recommendations</b>
### Predicted Compounding effects 
1. Improve job satisfaction of job seekers
    - Mid-career job seekers are more quickly employed to continue supporting their families
    - Resources (both time and money) saved from taking unnecessary upskill courses and training
    - Improved job satisfaction and allow greater meaning in work for job seeker


2. Minimising unemployment burden
    - With suitable job roles quickly identified, unnecessary mismatched training could be avoided, reducing subsidies spent by WoG
    - Improve in job satisfaction allows improve in mental health of seekers in further improving productivity of economy


3. Increasing productivity in companies
    - Reduction in downtime of ‘handholding’ new staff
    - Enhancing innovation as mid-career professionals combine skills/ fresh perspectives from past vocations to present roles
    - Higher retention rate - avoids inefficient use of resources to re-train new staff constantly
    - Mid-career seekers could be more adaptive to change, a critical trait in tech industry
    - Company’s productivity and performance increases
 

4. Lifelong learning
    - Companies can tailor employee's training plans to their personality
        - Employee can learn with the method that suits them best
    - Ensure that companies are always head to the most relevant skills as skills allocation can be diversified to fit different employee's personality type


### Future work:
1. We would suggest for future work to train classification model on employees' social media posts together with linkedin posts to show their personality hence more accurate predictions
2. We would also like to collect more data for the extrovert MBTI types to train the classification model so that it can be used for more generalised cases. 
3. Explore model training on other personality tests such as Enneagram, DISC personality tests to see if there are similar effects. This is so that the model can be used in companies that do not utilise the MBTI tests
4. Explore models such as CNN/ RNN in improving MBTI classification model


<br>

Sources:

- [What is the Average Cost of Training a New Employee?](https://www.auston.edu.sg/advice/subsidies-for-skillsfuture-courses-in-singapore/)
- [SkillsFuture Mid Career Subsidy](https://www.skillsfuture.gov.sg/initiatives/mid-career/enhancedsubsidy#:~:text=What%20is%20it%3F,adapt%20to%20changing%20job%20requirements.)
- [What is the Actual Cost of Training Employees](https://hrshelf.com/cost-of-training-employees/)
- [The Real Cost Of Training A New Hire](https://elmlearning.com/blog/how-much-does-employee-training-really-cost/)
- [How to Scrape LinkedIn and 8 Best LinkedIn Scrapers in 2024](https://research.aimultiple.com/linkedin-scrapers/)
- [What is the Cost to Deploy and Maintain a Machine Learning Model?](https://medium.com/@yomna/im-99-extroverted-and-i-struggle-with-social-media-f627c40982f1)
- [The Cost of Burnout In US Companies For 2023](https://www.3treetech.com/cost-of-burnout-in-us-companies-in-2023-burnout-cost/)
- [The True Cost of Employee Burnout](https://www.whoopunite.com/blog/business/articles/cost-of-employee-burnout/)
- [Corporate Burnout Is Coming For Investor Profits](https://www.forbes.com/sites/qai/2023/01/30/corporate-burnout-is-coming-for-investor-profits/?sh=602d322f7008)
- [The True Costs of Employee Turnover](https://builtin.com/recruiting/cost-of-turnover)
- [8 in 10 IT professionals experience burnout: Survey](https://theindependent.sg/8-in-10-it-professionals-experience-burnout-survey/)
- [Cybersecurity burnout hits APAC firms, with lack of resources the key challenge](https://www.zdnet.com/google-amp/article/cybersecurity-burnout-hits-apac-firms-with-lack-of-resources-the-key-challenge/)
- [How work-related stress affects employee health and productivity](https://www.cigna.com.sg/health-content-hub/mental-health/how-work-related-stress-affects-employee)
- [CNBC: 4 in 5 employees in Asia have moderate to high mental health risk, study shows](https://www.cnbc.com/2023/09/20/4-in-5-employees-in-asia-at-risk-developing-mental-health-issues-study.html)
- [Graebel’s innovative solution lowers costs, enhances employee productivity in Singapore](https://www.graebel.com/blog/graebels-innovative-solution-lowers-costs-enhances-employee-productivity-in-singapore/)

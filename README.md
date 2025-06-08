# PsyberNet: Personality Prediction Using Machine Learning

This repository contains the code and documentation for "PsyberNet," a machine learning project aimed at predicting personality types using text data from social media posts. The project evaluates the validity of the Myers-Briggs Type Indicator (MBTI) in predicting language style and online behavior.

## Table of Contents

1.  [Introduction](#introduction)
2.  [Objectives](#objectives)
3.  [Dataset Overview](#dataset-overview)
4.  [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
5.  [Feature Engineering](#feature-engineering)
6.  [Data Cleaning and Preprocessing](#data-cleaning-and-preprocessing)
7.  [Modeling Approach](#modeling-approach)
8.  [Model Implementations and Results](#model-implementations-and-results)
    * [Logistic Regression](#logistic-regression)
    * [Linear Support Vector Classifier (SVC)](#linear-support-vector-classifier-svc)
    * [Multinomial Naive Bayes](#multinomial-naive-bayes)
    * [Decision Tree Classifier](#decision-tree-classifier)
    * [Random Forest Classifier](#random-forest-classifier)
    * [XGBoost Classifier](#xgboost-classifier)
    * [CatBoost Classifier](#catboost-classifier)
9.  [Future Work](#future-work)
10. [References](#references)
11. [Contribution](#contribution)

---

## 1. Introduction

The Myers-Briggs Type Indicator (MBTI) is a widely recognized personality assessment tool that categorizes individuals into 16 personality types based on four dichotomies: Introversion (I) vs. Extroversion (E), Intuition (N) vs. Sensing (S), Thinking (T) vs. Feeling (F), and Judging (J) vs. Perceiving (P). Each personality type is a unique combination of these preferences. This project leverages a Kaggle dataset of social media posts, each labeled with an MBTI type, to explore the connections between personality and language. [cite_start]The primary goal is to predict communication patterns and analyze the MBTI's validity in predicting online language style and behavior. 

## 2. Objectives

The main objectives of this project are:
* [cite_start]To investigate whether MBTI can accurately predict individual language styles and online behaviors. 
* [cite_start]To assess the discriminative power of different personality types using various machine learning techniques. 

## 3. Dataset Overview

[cite_start]The dataset used in this analysis is sourced from [Kaggle MBTI Personality Types Dataset](https://www.kaggle.com/datasets/datasnaek/mbti-type)[cite: 349].
* [cite_start]**Structure**: The dataset comprises user posts labeled with one of the 16 MBTI personality types.  [cite_start]Each entry contains text data reflecting a user's language style and online behavior. 
* **Dimensions**: The dataset contains 8675 entries across two columns (type and posts), providing a substantial number of samples for analysis. [cite_start]Data types were verified, and no missing values were found, ensuring data completeness. 

## 4. Exploratory Data Analysis (EDA)

[cite_start]Exploratory Data Analysis was performed to understand the dataset's structure, identify noise, and gain initial insights into language patterns. 

* [cite_start]**Pre-Cleaning Analysis**: Initial EDA involved analyzing the distribution of personality types, calculating average post length, and identifying frequent terms.  [cite_start]This helped in understanding common issues like URLs, special characters, and non-informative language, guiding the data cleaning approach. 
* **Data Exploration & Initial Insights**:
    * [cite_start]**Visualization**: Interactive visualizations were created, including histograms of MBTI type distribution (revealing class imbalance)  [cite_start]and box plots of text lengths by MBTI type (showing how verbosity varies). 
    * [cite_start]**Word Frequency Analysis**: Common language patterns were examined by identifying the 20 most frequently occurring words.  [cite_start]This provided an overview of commonly used language elements and general themes across personality types. 
    * [cite_start]**TF-IDF Analysis**: Term Frequency-Inverse Document Frequency (TF-IDF) was used to convert text data into a numerical format, highlighting distinctive words associated with each MBTI type.  [cite_start]Higher TF-IDF scores indicate more relevant and distinctive terms within a document. 
    * [cite_start]**Sentiment Analysis**: Conducted using the TextBlob library, sentiment analysis calculated a polarity score (-1 to +1) for each post.  [cite_start]Box plots visualized sentiment distribution across MBTI types, revealing general neutral sentiment but also extreme scores for some individuals. 
* [cite_start]**Key Challenges**: High-dimensional text data with significant variation across posts and noise from irrelevant terms (e.g., URLs, filler words) were identified as key challenges. 

## 5. Feature Engineering

[cite_start]To prepare the data for machine learning, relevant numerical features were derived: 
* [cite_start]**Encoding MBTI Types**: MBTI types were converted into numerical codes using `LabelEncoder` from `scikit-learn`, a necessary step for machine learning algorithms. 
* [cite_start]**Text Length**: The character length of each post was calculated to provide insights into verbosity and language style. 
* [cite_start]**Sentence Count**: The number of sentences in each post (determined by counting periods) was added as another dimension of textual analysis. 
* [cite_start]**Sentiment Score**: The TextBlob-derived sentiment polarity score was included to reflect the emotional tone of posts. 
* [cite_start]**Pairwise Relationships Analysis**: Pairwise scatter plots were generated using `seaborn.pairplot` to visualize relationships between text length, sentence count, and sentiment score, colored by MBTI type, revealing patterns and clusters.  [cite_start]Text length and sentence count showed a strong positive correlation, while sentiment was largely independent of text volume. 
* [cite_start]**N-Grams Analysis**: N-grams (trigrams and 4-grams) were used to identify frequently co-occurring phrases, reflecting common themes or ideas associated with specific personality traits.  [cite_start]Frequent YouTube links and informal language were observed, suggesting a conversational and discussion-based dataset. 

## 6. Data Cleaning and Preprocessing

[cite_start]Data cleaning and preprocessing were crucial steps to prepare the dataset for modeling. 
* **Text Normalization**: Text data was standardized by:
    * [cite_start]Lowercasing to reduce redundancy. 
    * [cite_start]Removing URLs to eliminate noise. 
    * [cite_start]Removing non-alphanumeric characters to improve focus on meaningful content. 
* [cite_start]**Stratified Train-Test Split**: The dataset was split into training and test sets, preserving the distribution of MBTI personality types to maintain class balance and prevent bias. 
* [cite_start]**Lemmatization**: Words were converted to their base forms (e.g., "running" to "run") to enhance feature quality. 
* [cite_start]**Feature Extraction using TF-IDF**: The TF-IDF vectorizer converted text data into numerical arrays, limiting features to 5,000 and excluding common stop words to focus on relevant terms, improving model efficiency and accuracy.  [cite_start]Word clouds were also generated to visualize prominent words across MBTI types. 

## 7. Modeling Approach

Various classification models were trained and compared to identify the best performer. [cite_start]Model performance was comprehensively evaluated using Classification Reports (Precision, Recall, F1-score, overall accuracy), Confusion Matrices, and ROC curves. 

* [cite_start]**Classification Report**: Provides precision, recall, and F1-score for each class.  [cite_start]Certain classes (e.g., INFJ, INFP) showed higher precision and recall, while others (e.g., ESFP, ESTJ) performed worse, likely due to data imbalance. 
* [cite_start]**ROC Curves and AUC Scores**: Display the model's ability to distinguish between classes, with AUC indicating effectiveness.  [cite_start]Higher AUC scores (closer to 1.0) indicate better discrimination.  [cite_start]Models performed well for types like INTJ, INFJ, and ISTP, but others showed lower AUCs, indicating potential misclassification. 
* **Confusion Matrix**: Shows a breakdown of predictions for each class, with correct classifications on the diagonal. [cite_start]Off-diagonal values highlight commonly confused classes.  INFJ and INFP had high correct predictions, while ESFP and ESTJ were often misclassified, possibly due to overlapping linguistic patterns.

## 8. Model Implementations and Results

[cite_start]The following models were implemented and evaluated:

### Logistic Regression

* [cite_start]**Implementation**: The Logistic Regression model was trained with a maximum of 3000 iterations and a regularization parameter (C) set to 0.5.  [cite_start]This model, often used for binary classification, was extended for multi-class scenarios (e.g., one-vs-rest). 
* [cite_start]**Accuracy (Training)**: `0.67` 
* [cite_start]**Accuracy (Test)**: `0.63` 

| Metric      | [cite_start]Precision (Train)  | [cite_start]Recall (Train)  | [cite_start]F1-Score (Train)  | [cite_start]Support (Train)  | [cite_start]Precision (Test)  | [cite_start]Recall (Test)  | [cite_start]F1-Score (Test)  | [cite_start]Support (Test)  |
|-------------|-------------------|----------------|------------------|-----------------|------------------|---------------|-----------------|----------------|
| ENFJ        | 0.86              | 0.16           | 0.27             | 152             | 0.08             | 0.02          | 0.03            | 38             |
| ENFP        | 0.80              | 0.65           | 0.72             | 540             | 0.67             | 0.55          | 0.61            | 135            |
| ENTJ        | 0.93              | 0.29           | 0.44             | 185             | 0.29             | 0.12          | 0.17            | 46             |
| ENTP        | 0.82              | 0.66           | 0.73             | 548             | 0.63             | 0.42          | 0.50            | 137            |
| ESFJ        | 0.00              | 0.00           | 0.00             | 33              | 0.00             | 0.00          | 0.00            | 8              |
| ESFP        | 0.00              | 0.00           | 0.00             | 38              | 0.00             | 0.00          | 0.00            | 10             |
| ESTJ        | 0.00              | 0.04           | 0.08             | 31              | 0.00             | 0.00          | 0.00            | 8              |
| ESTP        | 1.00              | 0.08           | 0.14             | 71              | 0.00             | 0.00          | 0.00            | 18             |
| INFJ        | 0.73              | 0.83           | 0.78             | 1176            | 0.72             | 0.81          | 0.76            | 294            |
| INFP        | 0.66              | 0.93           | 0.77             | 1466            | 0.62             | 0.91          | 0.74            | 367            |
| INTJ        | 0.73              | 0.80           | 0.77             | 873             | 0.69             | 0.74          | 0.71            | 218            |
| INTP        | 0.69              | 0.88           | 0.77             | 1043            | 0.66             | 0.84          | 0.74            | 261            |
| ISFJ        | 0.89              | 0.38           | 0.54             | 133             | 0.25             | 0.12          | 0.16            | 33             |
| ISFP        | 0.86              | 0.39           | 0.54             | 217             | 0.22             | 0.09          | 0.13            | 54             |
| ISTJ        | 0.86              | 0.51           | 0.64             | 164             | 0.50             | 0.26          | 0.34            | 41             |
| ISTP        | 0.86              | 0.45           | 0.59             | 270             | 0.31             | 0.14          | 0.19            | 68             |

* **Confusion Matrix (Test)**: The confusion matrix for Logistic Regression (test set) shows the actual vs. predicted personality types, with diagonal values representing correct classifications. [cite_start]Misclassifications are visible in off-diagonal cells. 

* **ROC Curves**: ROC curves for Logistic Regression showed varying performance across classes. [cite_start]AUC scores for classes like ENFJ (0.88), ENFP (0.93), ENTJ (0.95), ENTP (0.92), ESFJ (0.89), ESFP (0.79), ESTJ (0.93), ESTP (0.97), INFJ (0.93), INFP (0.93), INTJ (0.91), INTP (0.95), ISFJ (0.97), ISFP (0.93), ISTJ (0.95), and ISTP (0.94) indicate the model's discriminative ability. 

### Linear Support Vector Classifier (SVC)

* [cite_start]**Implementation**: The Linear SVC was implemented with a regularization parameter (C) of 0.1.  [cite_start]SVC is well-suited for high-dimensional text classification problems. 
* [cite_start]**Test Accuracy**: 0.628818 

### Multinomial Naive Bayes

* [cite_start]**Implementation**: The Multinomial Naive Bayes classifier is particularly suited for text classification.  [cite_start]It operates under the assumption that features (words) are conditionally independent given the class label. 
* [cite_start]**Test Accuracy**: 0.378098 

### Decision Tree Classifier

* [cite_start]**Implementation**: The Decision Tree Classifier was implemented with a maximum depth of 14 to prevent overfitting and maintain interpretability. 
* [cite_start]**Test Accuracy**: 0.506628 

### Random Forest Classifier

* **Implementation**: The Random Forest Classifier is an ensemble learning method that builds multiple decision trees and merges their predictions to improve accuracy and control overfitting.
* [cite_start]**Test Accuracy**: 0.438040 

### XGBoost Classifier

* **Implementation**: The XGBoost Classifier utilizes gradient boosting and is known for its performance and speed. [cite_start]Configured to use GPU acceleration, it includes hyperparameters such as maximum depth and learning rate to fine-tune the model's learning process. 
* [cite_start]**Test Accuracy**: 0.665130 

### CatBoost Classifier

* **Implementation**: The CatBoost Classifier is another gradient boosting library that handles categorical features efficiently and is known for its high performance.
* [cite_start]**Test Accuracy**: 0.672622 
* [cite_start]**Key Insights**: Both XGBoost and CatBoost excelled in accuracy and AUC scores due to their ensemble boosting, ability to handle non-linear relationships, and effective feature handling.

---

## 9. Future Work

* [cite_start]Testing alternative personality frameworks, like the Big Five, to compare their effectiveness in personality prediction. 
* [cite_start]Integrating more sophisticated language models or ensembling different models to capture personality nuances more effectively. 

## 10. References

* [cite_start]Dataset: [Kaggle MBTI Personality Types Dataset](https://www.kaggle.com/datasets/datasnaek/mbti-type) 
* [cite_start]For basic ideas: [Myers-Briggs Type Indicator (MBTI) Assumptions, Dichotomies, and Application](https://www.geeksforgeeks.org/myers-briggs-type-indicator-mbti-assumptions-dichotomies-and-application/) 
* [cite_start]For Streamlit: [Streamlit](https://streamlit.io) 

## 11. Contribution

* [cite_start]**Mahi Upadhyay**: EDA, Data Cleaning, Tokenization, Report Generation. 
* [cite_start]**Raditya Saraf**: Model Selection, Accuracy Comparison, Streamlit Development, PPT Creation. 
* [cite_start]**Akshay Verma**: Initial Research, GitHub Readme, Code Designing, Report Assistance. 
* [cite_start]**Polaki Avinash**: Helped with PPT & Research.
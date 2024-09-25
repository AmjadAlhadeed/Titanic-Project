# Titanic Survival Prediction

## Project Overview

This project involves predicting the survival of passengers on the Titanic using machine learning techniques. The dataset provides information on the passengers, such as age, sex, class, and whether they survived the disaster. By analyzing these factors, the model aims to predict the likelihood of survival for a given passenger.

## Dataset Information

The Titanic dataset contains the following key features:

- **PassengerId**: Unique identifier for each passenger.
- **Pclass**: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd).
- **Name**: Name of the passenger.
- **Sex**: Gender of the passenger.
- **Age**: Age of the passenger.
- **SibSp**: Number of siblings or spouses aboard the Titanic.
- **Parch**: Number of parents or children aboard the Titanic.
- **Ticket**: Ticket number.
- **Fare**: Amount paid for the ticket.
- **Cabin**: Cabin number.
- **Embarked**: Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).
- **Survived**: Target variable indicating survival (0 = No, 1 = Yes).

## Key Features of the Project

- **Data Cleaning & Preprocessing**:
  - Handle missing values (especially in `Age`, `Cabin`, and `Embarked` columns).
  - Convert categorical variables (e.g., `Sex`, `Embarked`) to numerical format using techniques such as one-hot encoding.
  
- **Exploratory Data Analysis (EDA)**:
  - Visualize survival rates by different features such as `Pclass`, `Sex`, and `Age`.
  - Analyze correlations between different features and the likelihood of survival.

- **Model Building**:
  - Train machine learning models (e.g., Logistic Regression, Random Forest, Decision Trees) to predict survival.
  
- **Model Evaluation**:
  - Use metrics such as accuracy, precision, recall, and F1-score to assess the performance of the models.

## Installation & Requirements

To run this project, you will need Python and the following libraries:

- **Pandas**: For data manipulation.
- **Matplotlib** and **Seaborn**: For data visualization.
- **Scikit-learn**: For machine learning model building and evaluation.

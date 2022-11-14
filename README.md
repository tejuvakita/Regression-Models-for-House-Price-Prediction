
# Regression Models for House Price Prediction

## Table of Content
  * [Business Objective](#Business-Objective)
  * [Data Description](#Data-Description)
  * [Aim](#Aim)
  * [Tech stack](#Tech-stack)
  * [Approach](#Approach)
  * [Project Takeaways](#Project-Takeaways)

## Business Objective

The price of a house is based on several characteristics such as location, total area, number of rooms, various amenities available, etc.
In this project, we will perform house price prediction for 200 apartments in Pune city.
Different regression models such as Linear, Random Forest, XGBoost, etc., will be implemented. Also, multi-layer perceptron (MLP) models will be implemented using
scikit-learn and TensorFlow.
This house price prediction project will help you predict the price of houses based on various features and house properties.


## Data Description

We are given a real estate dataset with around 200 rows and 17 different variables that play an important role in predicting our target variable, i.e., price.

## Aim

The goal is to predict sale prices for homes in Pune city.

## Tech stack

**Language** - Python<br />
**Libraries** - sklearn, pandas, NumPy, matplotlib, seaborn, xgboost<br />

## Approach

1. Data Cleaning<br />
● Importing the required libraries and reading the dataset.<br />
● Preliminary exploration<br />
● Check for the outliers and remove outliers.<br />
● Dropping of redundant feature columns<br />
● Missing value handling<br />
● Regularizing the categorical columns<br />
● Save the cleaned data<br />

2. Data Analysis<br />
● Import the required libraries and read the cleaned dataset.<br />
● Converting binary columns to dummy variables<br />
● Feature Engineering<br />
● Univariate and Bivariate analysis<br />
● Check for correlation<br />
● Feature selection<br />
● Data Scaling<br />
● Saving the final updated dataset<br />

3. Model Building<br />
● Data preparation<br />
● Performing train test split<br />
● Linear Regression<br />
● Ridge Regression<br />
● Lasso Regressor<br />
● Elastic Net<br />
● Random Forest Regressor<br />
● XGBoost Regressor<br />
● K-Nearest Neighbours Regressor<br />
● Support Vector Regressor<br />
4. Model Validation<br />
● Mean Squared Error<br />
● R2 score<br />
● Plot for residuals<br />
5. Performs the grid search and cross-validation for the given regressor<br />
6. Fitting the model and making predictions on the test data< /br>
7. Checking for Feature Importance<br />
8. Model comparisons.<br />
9. MLP (Multi-Layer Perceptron) Models<br />
● MLP Regression with scikit-learn<br />
● Regression with TensorFlow<br />
## Screenshots

![App Screenshot](flow.png)

## Project Takeaways

1. Understanding the business problem.<br />
2. Importing the dataset and required libraries.<br />
3. Performing basic Exploratory Data Analysis (EDA).<br />
4. Data cleaning and missing data handling if required, using appropriate methods.<br />
5. Checking for outliers<br />
6. Using Python libraries such as matplotlib and seaborn for data interpretation and advanced visualizations.<br />
7. Splitting dataset into train and test data<br />
8. Performing Feature Engineering on data for better performance.<br />
9. Training a model using Regression techniques like Linear Regression, Random Forest Regressor, XGBoost Regressor, etc.<br />
10.Training multiple models using different Machine Learning Algorithms suitable for the scenario and checking for best performance.<br />
11. Performing grid search and cross-validation for the given regressor<br />
12.Making predictions using the trained model.<br />
13.Gaining confidence in the model using metrics such as MSE, R2 squared <br />
14.Plot the residual plots for train and test data<br />
15.Find those features that are most helpful for prediction using Feature Importance.<br />
16.Model comparison<br />
17.Learn how to build a Multi-Layer Perceptron model using the Scikit-learn library<br />
18.Learn how to build a Multi-Layer Perceptron model using TensorFlow

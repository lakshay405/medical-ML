# medical-ML
Insurance Cost Prediction using Linear Regression
This project focuses on predicting insurance costs based on various attributes such as age, sex, BMI (body mass index), number of children, smoker status, and region using Linear Regression.

Dataset
The dataset (insurance.csv) used in this project contains information about insurance beneficiaries, including their demographic details and medical charges. Here's a summary of the dataset exploration and preprocessing steps:

Data Exploration: The dataset is explored to understand its structure and characteristics. This includes checking the shape, information, presence of missing values, and descriptive statistics.

Data Visualization: Various aspects of the dataset are visualized to gain insights:

Distribution of age, BMI, and charges.
Counts and distributions of categorical variables like sex, children, smoker status, and region.
Data Preprocessing: Categorical variables (sex, smoker, region) are encoded into numerical format suitable for machine learning algorithms. For example, sex is encoded as 0 for male and 1 for female, smoker is encoded as 0 for yes and 1 for no, and region is encoded based on geographical regions.

Workflow
Feature Selection: The dataset is split into features (X) and the target variable (Y) which is the insurance charges.

Data Splitting: The dataset is divided into training and test sets using train_test_split from sklearn.model_selection.

Model Training: A Linear Regression model is chosen and trained using the training data (X_train and Y_train).

Model Evaluation: The trained model's performance is evaluated using R-squared score, which measures how well the model predicts the variation in the dependent variable (charges). Both training and test datasets are used for evaluation to ensure the model's generalization capability.

Prediction: The trained model is used to predict insurance costs for new input data. An example input data consisting of age, sex, BMI, number of children, smoker status, and region is provided to demonstrate the prediction capability of the model.

Technologies Used
Python
Pandas
Matplotlib
Seaborn
Scikit-learn (sklearn)
Usage
Ensure Python and necessary libraries (pandas, matplotlib, seaborn, scikit-learn) are installed.
Clone this repository and navigate to the project directory.
Place your dataset (insurance.csv) in the project directory.
Run the script insurance_cost_prediction.py to train the model and perform predictions.
Example Prediction
An example input data is provided to showcase how the model predicts insurance costs based on the provided features.

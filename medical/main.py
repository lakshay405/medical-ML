import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Loading the insurance dataset from a CSV file into a Pandas DataFrame
raw_insurance_data = pd.read_csv('/content/insurance.csv')

# Displaying the first 5 rows of the dataset
raw_insurance_data.head()

# Checking the number of rows and columns in the dataset
raw_insurance_data.shape

# Getting information about the dataset
raw_insurance_data.info()

# Checking for missing values in the dataset
raw_insurance_data.isnull().sum()

# Descriptive statistics of the dataset
raw_insurance_data.describe()

# Visualizing the distribution of age values
sns.set()
plt.figure(figsize=(6, 6))
sns.distplot(raw_insurance_data['age'])
plt.title('Age Distribution')
plt.show()

# Visualizing the distribution of sex values
plt.figure(figsize=(6, 6))
sns.countplot(x='sex', data=raw_insurance_data)
plt.title('Sex Distribution')
plt.show()

# Counting the occurrences of each sex value
raw_insurance_data['sex'].value_counts()

# Visualizing the distribution of BMI (bmi) values
plt.figure(figsize=(6, 6))
sns.distplot(raw_insurance_data['bmi'])
plt.title('BMI Distribution')
plt.show()

# Visualizing the distribution of children values
plt.figure(figsize=(6, 6))
sns.countplot(x='children', data=raw_insurance_data)
plt.title('Children Distribution')
plt.show()

# Counting the occurrences of each children value
raw_insurance_data['children'].value_counts()

# Visualizing the distribution of smoker values
plt.figure(figsize=(6, 6))
sns.countplot(x='smoker', data=raw_insurance_data)
plt.title('Smoker Distribution')
plt.show()

# Counting the occurrences of each smoker value
raw_insurance_data['smoker'].value_counts()

# Visualizing the distribution of region values
plt.figure(figsize=(6, 6))
sns.countplot(x='region', data=raw_insurance_data)
plt.title('Region Distribution')
plt.show()

# Counting the occurrences of each region value
raw_insurance_data['region'].value_counts()

# Visualizing the distribution of charges values
plt.figure(figsize=(6, 6))
sns.distplot(raw_insurance_data['charges'])
plt.title('Charges Distribution')
plt.show()

# Encoding the 'sex' column
raw_insurance_data.replace({'sex': {'male': 0, 'female': 1}}, inplace=True)

# Encoding the 'smoker' column
raw_insurance_data.replace({'smoker': {'yes': 0, 'no': 1}}, inplace=True)

# Encoding the 'region' column
raw_insurance_data.replace({'region': {'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3}}, inplace=True)

# Separating the data into features (X) and target (Y)
X = raw_insurance_data.drop(columns='charges', axis=1)
Y = raw_insurance_data['charges']

# Splitting the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

# Loading the Linear Regression model
linear_regressor = LinearRegression()

# Training the Linear Regression model
linear_regressor.fit(X_train, Y_train)

# Making predictions on the training data
training_data_predictions = linear_regressor.predict(X_train)

# Calculating the R-squared value for training data
r2_train = metrics.r2_score(Y_train, training_data_predictions)
print('R-squared value (Training): ', r2_train)

# Making predictions on the test data
test_data_predictions = linear_regressor.predict(X_test)

# Calculating the R-squared value for test data
r2_test = metrics.r2_score(Y_test, test_data_predictions)
print('R-squared value (Test): ', r2_test)

# Building a predictive system
input_data = (31, 1, 25.74, 0, 1, 0)

# Converting input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshaping the array
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Making prediction using the trained model
prediction = linear_regressor.predict(input_data_reshaped)
print('Predicted insurance cost: USD ', prediction[0])

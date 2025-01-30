from sklearn.ensemble import GradientBoostingClassifier  # Import Gradient Boosting Classifier
from sklearn.model_selection import train_test_split
import pandas as pd

from imblearn.over_sampling import SMOTE
import pickle

# Load and prepare the dataset
df = pd.read_csv("data.csv")  # Ensure the correct path to your CSV file
df.drop('id', axis=1, inplace=True)

# Convert categorical variables into dummy/indicator variables
gender = pd.get_dummies(df['Gender'], drop_first=True)
vehicle_age = pd.get_dummies(df['Vehicle_Age'], drop_first=True)
vehicle_damage = pd.get_dummies(df['Vehicle_Damage'], drop_first=True)

# Concatenate the dummy variables to the original dataframe
df = pd.concat([df, gender, vehicle_age, vehicle_damage], axis=1)
df.drop(['Gender', 'Vehicle_Age', 'Vehicle_Damage'], axis=1, inplace=True)

# Define features and target variable
X = df.drop('Response', axis=1)
Y = df['Response']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

# Apply SMOTE for handling class imbalance
sm = SMOTE(random_state=2)
X_train_res, Y_train_res = sm.fit_resample(X_train, Y_train.ravel())  # Updated method name to fit_resample

# Initialize and train the Gradient Boosting Classifier
classifier = GradientBoostingClassifier(random_state=42)
classifier.fit(X_train_res, Y_train_res)

# Saving model to disk
pickle.dump(classifier, open('model.pkl', 'wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl', 'rb'))
result = model.score(X_test, Y_test)

# Simplified result reporting
print("Result:", "Pass" if result > 0.5 else "Fail")

import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score

# Load the dataset
file_path = "C:/Users/potnu/Desktop/minor project/datasets/Software_Professional_Salaries.updated.csv"
if os.path.exists(file_path):
    print("File exists")
    data = pd.read_csv(file_path)
    print("First few rows of the dataset:")
    print(data.head())  # Print the first few rows of the DataFrame
    print("Column names in the dataset:")
    print(data.columns)  # Print the column names
else:
    print("File does not exist")
    exit()

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Replace 'Job Title' with the correct column name found in your dataset ('Job_Title')
data['job_title_encoded'] = label_encoder.fit_transform(data['Job_Title'])

# Features and target variable
X = data[['Salary']]
y = data['job_title_encoded']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
error_rate = 100 - (accuracy * 100)
print(f'Accuracy: {error_rate:.2f}%')

print(f'Error Rate: {accuracy * 100:.2f}%')
#print(f'Accuracy: {error_rate:.2f}%')

# Check number of unique classes
unique_classes_in_y_test = len(set(y_test))
unique_classes_in_label_encoder = len(label_encoder.classes_)
print(f"Unique classes in y_test: {unique_classes_in_y_test}")
print(f"Unique classes in label_encoder: {unique_classes_in_label_encoder}")

# Convert target_names to a list
target_names = list(label_encoder.classes_)

# Ensure the number of unique classes matches
if unique_classes_in_y_test == unique_classes_in_label_encoder:
    print(classification_report(y_test, y_pred, target_names=target_names))
else:
    print("Number of unique classes in y_test does not match the number of classes in label_encoder")

# Save the model and label encoder
joblib.dump(model, 'job_salary_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

print("Model and label encoder saved successfully!")

# Explore the dataset
print(data.describe())
print(data.info())
'''
# Visualize the distribution of salaries
sns.histplot(data['Salary'], kde=True)
plt.title('Salary Distribution')
plt.show()

# Visualize the relationship between salary and job title
sns.boxplot(x='Job_Title', y='Salary', data=data)
plt.title('Salary by Job Title')
plt.xticks(rotation=90)
plt.show()'''

# Normalize the Salary feature
scaler = StandardScaler()
X = scaler.fit_transform(data[['Salary']])

# Define a parameter grid for Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Initialize the model
model = RandomForestClassifier(random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Best parameters
print(f'Best parameters found: {grid_search.best_params_}')

# Evaluate the model Error Rate Accuracy
y_pred = grid_search.best_estimator_.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
error_rate = 100 - (accuracy * 100)
print(f':Accuracy {accuracy * 100:.2f}%')
print(f':Error Rate {error_rate:.2f}%')
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Evaluate the model using cross-validation
cv_scores = cross_val_score(model, X, y, cv=10)
print(f'Cross-validation scores: {cv_scores}')
print(f'Mean cross-validation score: {cv_scores.mean() * 100:.2f}%')

# Split the data into different sizes
train_sizes = [0.1, 0.3, 0.5, 0.7, 0.9]
accuracies = []
error_rates = []

for train_size in train_sizes:
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=42)
    
    # Initialize and train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    error_rate = 100 - (accuracy * 100)
    accuracies.append(accuracy)
    error_rates.append(error_rate)
    print(f'Train size: {train_size * 100:.0f}%, Accuracy: {accuracy * 100:.2f}%, Error Rate: {error_rate:.2f}%')
# Ensure the column names are correct
print(data.columns)

# Calculate the average salary for each job title
average_salaries = data.groupby('Job_Title')['Salary'].mean().reset_index()

# Rename the columns for better readability
average_salaries.columns = ['Job_Title', 'Average_Salary']

# Print the result
print(average_salaries)

#"C:\Users\potnu\Desktop\minor project\code and output\code1.py"

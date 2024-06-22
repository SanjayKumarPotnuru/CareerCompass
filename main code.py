import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Load the dataset
#file_path = r'C:/Users/potnu/Desktop/datasets/Software_Professional_Salaries.csv'
file_path="C:/Users/potnu/Desktop/minor project/datasets/Software_Professional_Salaries.updated.csv"
if os.path.exists(file_path):
    #print("File exists")
    data = pd.read_csv(file_path)
    #print("First few rows of the dataset:")
    #print(data.head())  # Print the first few rows of the DataFrame
    #print("Column names in the dataset:")
    #print(data.columns)  # Print the column names
else:
    print("File does not exist")
    exit()

# Handle missing values if any (optional)
data.dropna(subset=['Job_Title', 'Salary'], inplace=True)

# Calculate the average salary for each job title
average_salaries = data.groupby('Job_Title')['Salary'].mean().reset_index()

# Rename the columns for better readability
average_salaries.columns = ['Job_Title', 'Average_Salary']

# Print the average salaries
print("Average salaries for each job title:")
print(average_salaries)

# Take the expected salary as input from the user
expected_salary = float(input("Enter your expected salary: "))

# Find the job titles with average salaries closest to the user-provided expected salary
# Calculate the absolute difference between the expected salary and the average salary
average_salaries['Difference'] = abs(average_salaries['Average_Salary'] - expected_salary)

# Sort the DataFrame by the difference
average_salaries = average_salaries.sort_values(by='Difference')

# Get the job title with the closest average salary
closest_job_title = average_salaries.iloc[0]['Job_Title']
closest_average_salary = average_salaries.iloc[0]['Average_Salary']

# Print the result
print(f"The job title closest to your expected salary is: {closest_job_title}")
print(f"The average salary for this job title is: {closest_average_salary:.2f}")

# Optionally, you can show the top N closest job titles
top_n = 5
print(f"\nTop {top_n} job titles closest to your expected salary:")
print(average_salaries.head(top_n))





# Handle missing values if any (optional)
data.dropna(subset=['Job_Title', 'Salary'], inplace=True)

# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Calculate the average salary for each job title in the training set
average_salaries = train_data.groupby('Job_Title')['Salary'].mean().reset_index()
average_salaries.columns = ['Job_Title', 'Average_Salary']

# Function to find the closest job title based on average salary
def find_closest_job_title(salary, avg_salaries):
    avg_salaries['Difference'] = abs(avg_salaries['Average_Salary'] - salary)
    closest_job_title = avg_salaries.loc[avg_salaries['Difference'].idxmin()]['Job_Title']
    return closest_job_title

# Predict job titles for the test set based on the closest average salary
test_data['Predicted_Job_Title'] = test_data['Salary'].apply(lambda x: find_closest_job_title(x, average_salaries))




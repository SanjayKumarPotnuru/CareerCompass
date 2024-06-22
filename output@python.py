Python 3.11.4 (tags/v3.11.4:d2340ef, Jun  7 2023, 05:45:37) [MSC v.1934 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.

= RESTART: C:/Users/potnu/Desktop/minor project/code and output/code1.py
File exists
First few rows of the dataset:
           Job_Title   Salary
0  Android Developer   400000
1  Android Developer   400000
2  Android Developer  1000000
3  Android Developer   300000
4  Android Developer   600000
Column names in the dataset:
Index(['Job_Title', 'Salary'], dtype='object')
Accuracy: 88.21%
Error Rate: 11.79%
Unique classes in y_test: 449
Unique classes in label_encoder: 1084
Number of unique classes in y_test does not match the number of classes in label_encoder
Model and label encoder saved successfully!
             Salary  job_title_encoded
count  2.277400e+04       22774.000000
mean   6.953606e+05         600.522921
std    8.843263e+05         349.440995
min    2.112000e+03           0.000000
25%    3.000000e+05         240.000000
50%    5.000000e+05         755.500000
75%    9.000000e+05         854.000000
max    9.000000e+07        1083.000000
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 22774 entries, 0 to 22773
Data columns (total 3 columns):
 #   Column             Non-Null Count  Dtype 
---  ------             --------------  ----- 
 0   Job_Title          22774 non-null  object
 1   Salary             22774 non-null  int64 
 2   job_title_encoded  22774 non-null  int32 
dtypes: int32(1), int64(1), object(1)
memory usage: 444.9+ KB
None

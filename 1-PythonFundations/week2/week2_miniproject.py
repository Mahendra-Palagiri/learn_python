import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#🎯 Goal
#Take a dataset of student names, ages, scores, and subjects — then:
#	1.	Load & clean the data
#	2.	Analyze & summarize
#	3.	Visualize trends
#	4.	Write conclusions (like you would in an AI project report)


# 1️⃣ Load Data
sd  = pd.read_csv("./data/week2/studentdata.csv")
print("\n Sutdent Data : \n",sd) #6 Rows loaded from CSV should showup 

# --- Output ---
# Sutdent Data : 
#      Name   Age  Score  Subject
#0   Alice  25.0     88     Math
#1     Bob  30.0     92  Science
#2    Rick  35.0     85     Math
#3   Diana  40.0     90  English
#4   Jacob  40.0     50  Science
#5  Mathew   NaN     76  English


# 2️⃣ Clean Data (handle missing values)
sd = sd.dropna()
#sd = sd.dropna(subset=["Age", "Score"])  # explicit columns
print("\n Data frame afer dropping NA values \n",sd) #should show 5 rows as the missing value row has been eliminated

# --- Output ---
# Sutdent Data : 
#      Name   Age  Score  Subject
#0   Alice  25.0     88     Math
#1     Bob  30.0     92  Science
#2    Rick  35.0     85     Math
#3   Diana  40.0     90  English
#4   Jacob  40.0     50  Science



# 3️⃣ Add Grade Column
def grade_student(score):
    if score >=90:
        return 'A'
    elif score >=80:
        return 'B'
    else:
        return 'C'
    
sd['Grade'] = sd['Score'].apply(grade_student) #even though apply works, the most efficient solution would be use conditions and choices
print("\n Data frame afer adding Grade \n",sd)
# ----- Output
#  Data frame afer adding Grade 
#      Name   Age  Score  Subject Grade
# 0  Alice  25.0     88     Math     B
# 1    Bob  30.0     92  Science     A
# 2   Rick  35.0     85     Math     B
# 3  Diana  40.0     90  English     A
# 4  Jacob  40.0     50  Science     C

#--Optimal Solution --
# conditions =[
#     sd['Score'] >= 90,
#     (sd['Score'] >= 80) & (sd['Score'] < 90)
# ]

# chocies=['A','B']
# sd['Grade'] = np.select(conditions,chocies,default='C')  --> Most optimal way of adding a column with a conditional requirements


# 4️⃣ Summary Stats
print("\n Summary Stats for Student \n",sd.describe())
#------ Output ------
# Summary Stats for Student 
#               Age      Score
# count   5.000000   5.000000
# mean   34.000000  81.000000
# std     6.519202  17.521415
# min    25.000000  50.000000
# 25%    30.000000  85.000000
# 50%    35.000000  88.000000
# 75%    40.000000  90.000000
# max    40.000000  92.000000


# 5️⃣ Average Score by Subject
print("\n Average Score by Subject \n",sd.groupby('Subject')['Score'].mean())

# 6️⃣ Visualization: Score Distribution
sns.histplot(sd['Score'],bins=5,kde=True)
plt.title('StudentScore')
plt.show()

# 7️⃣ Visualization: Grade Counts
sns.countplot(x="Grade", data=sd, palette="viridis")
plt.title("Number of Students per Grade")
plt.show()

# 8️⃣ Visualization: Age vs Score
sns.scatterplot(x="Age", y="Score", hue="Grade", data=sd, palette="coolwarm")
plt.title("Age vs Score (Colored by Grade)")
plt.show()

# ------------------------------------------ 🏋️ Tasks.  -----------------------------------------
# 	1.	Add a Pass/Fail column (Pass if Score ≥ 75)
# 	2.	Find top 3 students by Score and print their names
# 	3.	Group by Grade and calculate the average Age per Grade
# 	4.	Create one more visualization of your choice (e.g., bar chart of Subject vs Avg Score)

# 1️⃣ Add a Pass/Fail column
sd['Result'] = np.where(sd['Score']>=75, 'Pass', 'Fail')
print("\n Data frame afer adding Pass/Fail \n",sd)
# ----- Output ----
#  Data frame afer adding Pass/Fail 
#      Name   Age  Score  Subject Grade Result
# 0  Alice  25.0     88     Math     B   Pass
# 1    Bob  30.0     92  Science     A   Pass
# 2   Rick  35.0     85     Math     B   Pass
# 3  Diana  40.0     90  English     A   Pass
# 4  Jacob  40.0     50  Science     C   Fail


# 2️⃣ Find top 3 students by Score and print their names
print("\n Top 3 students \n",sd.sort_values(by='Score',ascending=False).loc[:,'Name'].head(3))
#print("\n Top 3 students \n",sd.sort_values(by='Score',ascending=False).iloc[:3,0]) --> Alternative way
# ---- Output --------
# Top 3 students 
# 1      Bob
# 3    Diana
# 0    Alice


#Efficient and simple solution
#sd.nlargest(3, 'Score')['Name']

# 3️⃣ Group by Grade and calculate the average Age per Grade
print("\n Average Age by Grade \n",sd.groupby('Grade')['Age'].mean())
# ---- Output --------
#  Average Age by Grade 
#  Grade
# A    35.0
# B    30.0
# C    40.0

# If you want to calculate agregate of more than one columns like Age and Score
# sd.groupby('Grade', as_index=False).agg(
#     AvgAge=('Age', 'mean'),
#     AvgScore=('Score', 'mean')
# )

# 4️⃣ Create one more visualization of your choice (e.g., bar chart of Subject vs Avg Score)
avg_scores = sd.groupby('Subject')['Score'].mean()
avg_scores = avg_scores.sort_values(ascending=False)
ax = plt.bar(avg_scores.index,avg_scores.values,color='maroon')
for p in ax:
    plt.text(p.get_x() + p.get_width()/2, p.get_height(), f"{p.get_height():.1f}",
             ha='center', va='bottom')
plt.title('AverageScore by Subject')
plt.xlabel('Subject')
plt.ylabel('AvgScore')
plt.show()

# Seaborn version of Subject vs Avg Score
# sns.barplot(
#     x='Subject',
#     y='Score',
#     data=sd,
#     estimator='mean',   # default is mean, but we can specify explicitly
#     palette='viridis'
# )
# plt.title('Average Score by Subject')
# plt.xlabel('Subject')
# plt.ylabel('Average Score')
# plt.show()
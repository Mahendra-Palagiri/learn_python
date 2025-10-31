#🎯 Learning Goals
#	1.	Create basic plots (line, bar, scatter) with Matplotlib
#	2.	Make statistical visualizations with Seaborn
#	3.	Customize plots (titles, labels, colors, styles)
#	4.	Visualize real-world datasets for insights

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Sample dataset
data = {
    "Name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
    "Age": [25, 30, 35, 40, 28],
    "Score": [88, 92, 85, 90, 95]
}
df = pd.DataFrame(data)

# 1️⃣ Line Plot
plt.plot(df['Name'], df['Score'], marker='x')
plt.title('StudentScore')
plt.xlabel('Name')
plt.ylabel('Score')
plt.show()

# 2️⃣ Bar Chart
plt.bar(df['Name'], df['Score'], color='orange')
plt.title('StudentScoreBarChart')
plt.show()

# 3️⃣ Scatter Plot
plt.scatter(df['Age'], df['Score'], color='green')
plt.title('Score vs Age')
plt.xlabel('Age')
plt.ylabel('Score')
plt.show()

# 4️⃣ Seaborn Histogram
sns.histplot(df['Score'], bins=5, kde=True)
plt.title('Score Distribution')
plt.show()

# 5️⃣ Seaborn Boxplot
sns.boxenplot(x=df['Score'])
plt.title('Score Spread')
plt.show()


#🏋️ Part 2: Exercises
#	1.	Bar Chart Practice
#   	•	Load sample.csv
#	    •	Plot Name vs Score in a bar chart
#	2.	Histogram Practice
#	    •	Plot a histogram of Age from sample.csv
#	3.	Scatter Plot Challenge
#	    •	Plot Age vs Score
#	    •	Highlight students with Score > 90 in a different color
#	4.	Seaborn Bonus
#	    •	Create a boxplot of Score grouped by a new column Grade (A, B, C)

sdf = pd.read_csv("./data/week2/sample.csv")
print(sdf)

# 1️⃣ Bar Chart Practice
plt.bar(sdf['Name'], sdf['Score'], color='purple')
plt.title("NametoScoreBoard")
plt.show()

# 2️⃣ Histogram Practice
sns.histplot(sdf['Age'], bins=5, kde=True)
plt.title('HistogramPractise')
plt.show()

# 3️⃣ Scatter Plot Challenge
#Filter the data 
highs = sdf[sdf['Score']> 90]
lows = sdf[sdf['Score'] <=90]
plt.scatter(highs['Age'], highs['Score'], color='red')
plt.scatter(lows['Age'], lows['Score'], color='magenta')
plt.title('ScatterPlot')
plt.xlabel('Age')
plt.ylabel('Score')
plt.legend()
plt.show()

# 4️⃣ Seaborn Bonus
conditions = [
    sdf['Score'] >=90,
    (sdf['Score'] >=80) & (sdf['Score'] <90)
]
choices =['A','B']
sdf['Grade'] = np.select(conditions, choices, default='C')
print(sdf)
sns.boxenplot(x=sdf['Grade'])
plt.title('GradeBoxPlot')
plt.show()

'''
“What plot do I use?” (mini decision tree)
	•	Single numeric: sns.histplot or sns.kdeplot
	•	Numeric vs numeric: sns.scatterplot (+ hue= a category)
	•	Numeric by category: sns.boxplot, sns.violinplot, or sns.boxenplot
	•	Category counts: sns.countplot
'''
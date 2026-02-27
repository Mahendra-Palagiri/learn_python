"""
Week 2 Day 7 — Quiz (Pandas + Visualization + NumPy)
Run: python week2_day7_quiz.py
"""

score = 0
total = 10

print("\nQ1) In pandas, which selects rows by integer position?")
print("A) .loc    B) .iloc    C) .ix    D) .where")
ans = input("Your answer: ").strip().upper()
score += (ans == "B")

print("\nQ2) What does df.describe() return by default?")
print("A) Column dtypes    B) Summary stats for numeric cols")
print("C) Full dataset     D) Unique values per column")
ans = input("Your answer: ").strip().upper()
score += (ans == "B")

print("\nQ3) Which merges like SQL LEFT JOIN?")
print("A) pd.concat    B) pd.merge(..., how='left')    C) df.join(..., how='inner')    D) pd.append")
ans = input("Your answer: ").strip().upper()
score += (ans == "B")

print("\nQ4) In NumPy, broadcasting allows:")
print("A) Arrays of any shape to combine without rules")
print("B) Combining compatible shapes without explicit loops")
print("C) Only same-shaped arrays to combine")
print("D) Automatic type casting of strings to numbers")
ans = input("Your answer: ").strip().upper()
score += (ans == "B")

print("\nQ5) Which creates a histogram in matplotlib?")
print("A) plt.scatter   B) plt.plot   C) plt.hist   D) plt.bar")
ans = input("Your answer: ").strip().upper()
score += (ans == "C")

print("\nQ6) Given df['Score']>=90, which returns only those rows?")
print("A) df.filter('Score>=90')   B) df.query('Score>=90')")
print("C) df[df['Score']>=90]      D) Both B and C")
ans = input("Your answer: ").strip().upper()
score += (ans == "D")

print("\nQ7) Which is the fastest way to create a conditional column?")
print("A) for-loop row by row")
print("B) df.apply(row_function, axis=1)")
print("C) np.select with vectorized conditions")
print("D) map over list(df.itertuples())")
ans = input("Your answer: ").strip().upper()
score += (ans == "C")

print("\nQ8) Which returns top 3 rows by 'Score'?")
print("A) df.head(3)   B) df.sort_values('Score').tail(3)")
print("C) df.nlargest(3, 'Score')   D) df.sample(3)")
ans = input("Your answer: ").strip().upper()
score += (ans == "C")

print("\nQ9) Which line transposes a 2D NumPy array 'A'?")
print("A) A.reshape(-1)   B) A.T   C) A.swapaxes(0,0)   D) A[::-1]")
ans = input("Your answer: ").strip().upper()
score += (ans == "B")

print("\nQ10) Which seaborn function counts categories on x?")
print("A) sns.lineplot   B) sns.countplot   C) sns.boxplot   D) sns.kdeplot")
ans = input("Your answer: ").strip().upper()
score += (ans == "B")

print(f"\nYour score: {score}/{total}")
"""
Week 2 Day 7 — Challenge: Sales Mini-Analysis
"""
import pandas as pd
import matplotlib.pyplot as plt


# 1) Load dataset
# TODO: Read 'sales.csv', parse 'Date' column as datetime
sd  = pd.read_csv("./data/week2/sales.csv", parse_dates=['Date'])
print("\n\n",sd.head()) #Print first 5 rows 
print("\n Data description\n",sd.info())

# 2) Create new columns
# TODO: Extract Month from Date
# TODO: Verify revenue = Units * UnitPrice
sd['Month'] = sd['Date'].dt.to_period('M').astype(str)
filt  = (sd['Units']*sd['UnitPrice']).round(2) == sd['Revenue']
sd['RevEval'] = filt
print("\n New column to indicate if the values match \n",sd)


# 3) Analysis
# TODO: Top 3 products by total Revenue
# TODO: Revenue by Region
# TODO: Monthly revenue trend (pivot table by Product)

print("\n Revenue by region \n",sd.groupby('Region')['Revenue'].sum())

top3 = (sd.groupby('Product',as_index=False)['Revenue']
          .sum()
          .sort_values('Revenue',ascending=False)
          .head(3)
           )
print("\n Top 3 products by revenue\n",top3)


rev_region = (sd.groupby('Region',as_index=False)['Revenue']
              .sum()
              .sort_values('Revenue',ascending=False)
              )
print("\n Revenue by region \n",rev_region)

trend = sd.pivot_table(index='Month', columns='Product', values='Revenue', aggfunc='sum').fillna(0)
print("\n trend analysis \n",trend)

# 4) Plots
# TODO: Histogram of Units
# TODO: Bar chart: Revenue by Region
# TODO: Line chart: Monthly total revenue
plt.hist(sd['Units'], bins=10)
plt.title('Units freq')
plt.xlabel("Units")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

rev_region.plot(x="Region", y="Revenue", kind="bar", legend=False)
plt.title("Revenue by Region")
plt.tight_layout()
plt.show()

mon_revenue = (sd.groupby('Month',as_index=False)['Revenue']
               .sum()
               .sort_values('Revenue', ascending=False)
               )
mon_revenue.plot(x='Month',y='Revenue',kind='line',legend=False)
plt.title("Revenue by Month")
plt.show()

# 5) Stretch goal
# TODO: Find best Region–Product combo (highest avg revenue per order)
combo = (sd.groupby(['Region','Product'],as_index=False)
         .agg(avg_ret_per_order=('Revenue','mean'),orders=('Revenue','count'))
         .sort_values('avg_ret_per_order', ascending=False)
         )
print(sd)
print("\nBest Region-Product combo:\n", combo.head(5))


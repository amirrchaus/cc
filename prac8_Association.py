import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.impute import SimpleImputer

data = pd.read_csv("groceries - groceries.csv")
print(data.head())
print(data.columns)
transactions = data.values.astype(str).tolist()
transactions = [[item for item in row if item != 'nan'] for row in transactions]
print(transactions[:10])

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)
print(df.head(5))
print(df.shape)

# Generate frequent itemsets
frequent_itemsets = apriori(df, min_support=0.01, use_colnames=True)
frequent_itemsets.count()['itemsets']

# Plotting
# Plotting the barplot without the deprecation warning
plt.figure(figsize=(12, 6))
plt.xticks(rotation=90)
colors = sns.color_palette("Set2", n_colors=15)  # Use a larger palette
sns.barplot(x='itemsets', y='support', data=frequent_itemsets.nlargest(n=15, columns='support'),
            palette=colors, hue='itemsets', legend=False)  # Set hue to itemsets
plt.show()


# Association Rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1, num_itemsets=None)  # Add num_itemsets or use None
rules.sort_values(by=['support'], ascending=False)
print("RULES 1:\n", rules)

rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
rules["consequent_len"] = rules["consequents"].apply(lambda x: len(x))
print("RULES 2:\n", rules)

# Filter rules based on antecedent_len
print("RULES 3:\n", rules[rules['antecedent_len'] >= 2])

# Apply additional filters for rule selection
filtered_rules = rules[(rules['antecedent_len'] >= 2) &
                        (rules['confidence'] > 0.3) &
                        (rules['lift'] > 1)].sort_values(by=['lift', 'support'], ascending=False)
print("RULES 4:\n", filtered_rules)

# Filter based on consequent_len
filtered_rules = rules[(rules['consequent_len'] >= 2) &
                        (rules['lift'] > 1)].sort_values(by=['lift', 'confidence'], ascending=False)
print("RULES 5:\n", filtered_rules)

# Calculate Lift manually (if needed)
rules['lift'] = rules['support'] / (rules['antecedent_len'] * rules['consequent_len'])
print(rules)

# Accuracy Metrics for different metrics
print("---------------------ACCURACY METRICS-------------------------------")
rules_lift = association_rules(frequent_itemsets, metric="lift", min_threshold=1, num_itemsets=None)
print(rules_lift)
rules_confidence = association_rules(frequent_itemsets, metric="confidence", min_threshold=1, num_itemsets=None)
print(rules_confidence)
rules_leverage = association_rules(frequent_itemsets, metric="leverage", min_threshold=1, num_itemsets=None)
print(rules_leverage)
rules_conviction = association_rules(frequent_itemsets, metric="conviction", min_threshold=1, num_itemsets=None)
print(rules_conviction)








#FP Tree
print("FP Tree")

import pyfpgrowth
##transactions = [[1, 2, 5],
##                [2, 4],
##                [2, 3],
##                [1, 2, 4],
##                [1, 3],
##                [2, 3],
##                [1, 3],
##                [1, 2, 3, 5],
##                [1, 2, 3]]

import csv

with open("groceries - groceries.csv", encoding="utf8", newline='') as f:
    reader = csv.reader(f)
    data = list(reader)

for i in range(len(data)):
    if '' in data[i]:
        data[i] = [x for x in data[i] if x]
        

data.pop(0)
print("some records",data[0:10])
patterns = pyfpgrowth.find_frequent_patterns(data, 3)
print("pattern",patterns)
print()
rules = pyfpgrowth.generate_association_rules(patterns, 0.8)
print("rules output",rules)




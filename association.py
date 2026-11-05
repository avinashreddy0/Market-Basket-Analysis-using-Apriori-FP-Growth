import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori,fpgrowth,association_rules
import warnings
warnings.filterwarnings('ignore')
import joblib


#loading data 
df = pd.read_csv(r"C:\Users\indur\Downloads\market_basket_dataset.csv")

#converting to list to lists

tran = [i.split(', ') for i in df['Items']]

#transatation 

te = TransactionEncoder()
te_date = te.fit_transform(tran)

#dataframe

market_df = pd.DataFrame(te_date,columns=te.columns_)
print(market_df)

#model 1:        apriori
apriori_itemsets = apriori(market_df, min_support=0.5, use_colnames=True)
rules_ap = association_rules(apriori_itemsets, metric='lift', min_threshold=0.6)

# Convert frozensets to strings
rules_ap['antecedents'] = rules_ap['antecedents'].apply(lambda x: ', '.join([str(i) for i in x]))
rules_ap['consequents'] = rules_ap['consequents'].apply(lambda x: ', '.join([str(i) for i in x]))

print("Apriori Rules:")
print(rules_ap[['antecedents','consequents','support','confidence','lift']])

fp_itemsets = fpgrowth(market_df, min_support=0.5, use_colnames=True)
rules_fp = association_rules(fp_itemsets, metric='lift', min_threshold=0.6)

# Convert frozensets to strings
rules_fp['antecedents'] = rules_fp['antecedents'].apply(lambda x: ', '.join([str(i) for i in x]))
rules_fp['consequents'] = rules_fp['consequents'].apply(lambda x: ', '.join([str(i) for i in x]))

print("FP-Growth Rules:")
print(rules_fp[['antecedents','consequents','support','confidence','lift']])


#visulization

plt.figure(figsize=(10,6))
sns.scatterplot(x=rules_fp['antecedents'], y=rules_fp['consequents'], size=rules_fp['lift'], legend=False)
plt.xticks(rotation=90)
plt.xlabel("Antecedents")
plt.ylabel("Consequents")
plt.title("FP-Growth Association Rules")
plt.show()

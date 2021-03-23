import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

from sklearn.preprocessing import StandardScaler

import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title='Customer Segmentation with K-Means', page_icon="./f.png")
st.title('Customer Segmentation with K-Means')
st.subheader('By [Francisco Tarantuviez](https://www.linkedin.com/in/francisco-tarantuviez-54a2881ab/) -- [Other Projects](https://franciscot.dev/portfolio)')
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.write('---')
st.write("""
Imagine that you have a customer dataset, and you need to apply customer segmentation on this historical data.

Customer segmentation is the practice of partitioning a customer base into groups of individuals that have similar characteristics. It is a significant strategy as a business can target these specific groups of customers and effectively allocate marketing resources. For example, one group might contain customers who are high-profit and low-risk, that is, more likely to purchase products, or subscribe for a service. A business task is to retaining those customers. Another group might include customers from non-profit organizations. And so on.

In this application you can select a custom _k_ (number of groups to segment clients) and predict your own client case.
""")
cust_df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%204/data/Cust_Segmentation.csv")
st.write("## About dataset")
@st.cache
def load_data():
  return pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%204/data/Cust_Segmentation.csv")
cust_df = load_data()
st.dataframe(cust_df)

df = cust_df.drop('Address', axis=1)
X = df.values[:,1:]
X = np.nan_to_num(X)
Clus_dataSet = StandardScaler().fit_transform(X)


st.write(""" 
## Data Visualization
K is equl to three as default. This means that we got three different groups of customers. However you can change this value from left sidebar.

In the below chart we visualize these groups and how the income has a relation with the age.
""")
st.sidebar.header("Customize the project")
clusterNum = st.sidebar.slider("K number (n° of groups)", 2, 8, 3)
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
k_means.fit(X)
labels = k_means.labels_
df["Clus_km"] = labels

area = np.pi * ( X[:, 1])**2  
plt.scatter(X[:, 0], X[:, 3], s=area, c=labels.astype(np.float), alpha=0.5)
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)

st.pyplot()

fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
ax.set_xlabel('Education')
ax.set_ylabel('Age')
ax.set_zlabel('Income')

ax.scatter(X[:, 1], X[:, 0], X[:, 3], c= labels.astype(np.float))

st.write("And here we can visualize them better ")
st.pyplot()


st.write("### Make your prediction!")

col1, col2 = st.beta_columns(2)

age = col1.slider("Age", 5, 90, 20)
years_employed = col1.slider("Years Employed", 0, 50, 5)
edu = col1.selectbox("Edu", range(1, 6))
defaulted = col1.selectbox("Is defaulted?", ["No", "Yes"])

card_debt = col2.number_input("Card Debt", min_value=0, max_value=100)
ohter_debt = col2.number_input("Other Debt", min_value=0, max_value=100)
income = col2.number_input("Income", min_value=0, max_value=500)
defaulted = 1 if defaulted == "Yes" else 0
debt_income_radio = col2.number_input("Debt Income Ratio", min_value=0, max_value=100)

user_df = [[age, edu, years_employed, income, card_debt, ohter_debt, defaulted, debt_income_radio]]
user_df = StandardScaler().fit_transform(pd.DataFrame(user_df).to_numpy()[0][:, np.newaxis]).reshape(1,-1)

predictions = k_means.predict(user_df)

st.write("The given client belongs to the cluster: ")
st.dataframe(pd.DataFrame(pd.Series(predictions), columns=["Cluster"]))

# This app repository

st.write("""
## App repository

[Github](https://github.com/ftarantuviez/)TODO
""")
# / This app repository
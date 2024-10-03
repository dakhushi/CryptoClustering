## Cryptocurrency Clustering Project (Unsupervised Machine Learning)

<img width="500" alt="leaflet-logo" src="https://github.com/user-attachments/assets/f66e5c1c-bce3-4473-acf9-cf75e07cfb09">

In this project, I used unsupervised machine learning techniques to predict whether cryptocurrencies are influenced by short-term (24-hour) or medium-term (7-day) price changes. By applying clustering algorithms like K-means and Principal Component Analysis (PCA), we aim to group cryptocurrencies based on their price fluctuations and investigate the impact of dimensionality reduction.

**Project Steps**

**Data Loading and Exploration:**

The cryptocurrency data from crypto_market_data.csv is loaded into a DataFrame.
Summary statistics and visualizations are generated to understand the dataset.

**Data Preparation:**

The data is normalized using StandardScaler() to ensure all features are on a similar scale.
A new DataFrame is created using the scaled data, keeping the original coin_id as the index.
Elbow Method for Optimal K:

**The elbow method is implemented to find the best number of clusters (K).**
The inertia values are calculated for K ranging from 1 to 11.
An elbow curve is plotted to visually identify the optimal K.
K-means Clustering (Using Scaled DataFrame):

**Question: What is the best value for k?**
**Answer:** The resulting elbow graph shows two sharp points, out of which 4 seems to be the optimal K value with lower inertia of 79.022.

**A K-means model is initialized with the best value of K.**
The model is fitted to the scaled data to predict the cryptocurrency clusters.
A scatter plot is created using hvPlot to visualize the clusters based on 24-hour and 7-day price changes, with each cryptocurrency's cluster and coin ID labeled.
![image](https://github.com/user-attachments/assets/653d7824-be22-42b5-b498-9d184d0d3549)

**Principal Component Analysis (PCA):**
PCA is applied to reduce the dataset to three principal components.
The explained variance is calculated to determine how much information is retained in the reduced dataset.
A new DataFrame is created using the PCA-transformed data.

**Question: What is the total explained variance of the three principal components?**
**Answer:** The first, second and third principal components explain 37.2% , 34.7 and 17.6% of the variance respectively. The relatively even distribution across the first two components indicates that both are almost equally important in explaining the variability in the data.
This PCA model with 3 principal components has captured 89.50% of variability in the original features, which means these three components effectively capture the most significant patterns in the dataset, providing a well-balanced approach to dimensionality reduction and information retention.

Elbow Method for PCA Data:
The elbow method is repeated using the PCA-transformed data to find the best K.
The optimal K is compared with the value from the original scaled data to see if it differs.

**Question: What is the best value for k when using the PCA data?**
**Answer:** The sharp elbow appears at K=4 where the drop in inertia starts to slightly flatten out. So the optimal k value is 4 using PCA data.

**Question: Does it differ from the best k value found using the original data?**
**Answer:** Optimal K value ,determined from both the data sources ,original data and PCA scaled data,consistently reflects K=4. However, k=2 also looks like a candidate with sharp elbow.

**K-means Clustering (Using PCA DataFrame):**
K-means clustering is repeated using the PCA-transformed data and the best K.
A scatter plot is generated to visualize the clusters in the reduced feature space.
![image](https://github.com/user-attachments/assets/4362451a-859b-4ee2-83b3-7d1adee02d18)

**Comparison of Clustering Results:**

Composite plots are created to compare the clustering results from the original scaled data and the PCA data.
The impact of using fewer features (via PCA) on the clustering is analyzed.

**Visualise and Compare the Results**
The PCA-based clustering effectively grouped cryptocurrencies into four distinct clusters, each exhibiting clear behavioral patterns such as stability, growth, and volatility. While k-means clustering without optimization also identified meaningful clusters, it was less effective at capturing distinct differences, particularly in high-dimensional data. Notably, the green and red clusters contained only a single data point, prompting further analysis using alternative clustering models and performance comparisons to achieve a more accurate and optimal k value

**Get Cryptocurrency from original data for each cluster based on PCA (K=4)**

Cluster 2 Cryptocurrencies: ['bitcoin' 'ethereum' 'bitcoin-cash' 'binancecoin' 'chainlink' 'cardano'
 'litecoin' 'monero' 'tezos' 'cosmos' 'wrapped-bitcoin' 'zcash' 'maker']

Cluster 0 Cryptocurrencies: ['tether' 'ripple' 'bitcoin-cash-sv' 'crypto-com-chain' 'usd-coin' 'eos'
 'tron' 'okb' 'stellar' 'cdai' 'neo' 'leo-token' 'huobi-token' 'nem'
 'binance-usd' 'iota' 'vechain' 'theta-token' 'dash' 'ethereum-classic'
 'havven' 'omisego' 'ontology' 'ftx-token' 'true-usd' 'digibyte']

Cluster 3 Cryptocurrencies: ['ethlend']

Cluster 1 Cryptocurrencies: ['celsius-degree-token']

**Perfomance Analysis based on Calinski-Harabasz Score and Silhouette Scores**

**Observations for Calinski-Harabasz Scores:** KMeans performs well when K=4, where it achieves the highest score of 32.46. However, for K=5 and above, its performance is generally lower than Agglomerative Clustering.

**Observations for Silhouette Scores:** KMeans performs best for K=2, achieving the highest silhouette score of 72. However, its performance significantly drops as K increases, especially for K=3, where it has a score of 34.

**Based on both the scores Optimal value for K=2**
![image](https://github.com/user-attachments/assets/b7acfee4-8b4e-42bf-8543-17d77638f688)

**Question: After visually analysing the cluster analysis results, what is the impact of using fewer features to cluster the data using K-Means?**

**Answer:** After evaluating the Silhouette and Calinski-Harabasz scores, the results indicate that KMeans performs optimally with K=2, achieving a highest Silhouette score of 72. Applying K=2 and visualizing the cluster analysis revealed that reducing the number of features had a significant impact on the clustering outcome. Initially, the elbow curve suggested K=4 based on the original data. However, with the highest Silhouette score at K=2 and the application of PCA, the resulting plot demonstrated clearer and more precise clustering.

**KMeans Cluster Grouping and Heatmap Visualization**

KMeans Grouping: The code groups the cryptocurrency data by the clusters generated from the KMeans algorithm, specifically the k_clusters_4 (4 clusters). It calculates the mean values for each feature within each cluster, which gives an overview of how the clusters differ based on the mean values of features.
Heatmap Creation: The grouped data is visualized using a heatmap, which highlights the relationship between KMeans clusters and cryptocurrency features such as price changes over different time periods (24h, 7d, etc.). The color scheme (coolwarm) reflects the range of values, with cooler colors (blue) indicating lower values and warmer colors (red) representing higher values.

**Insights:**

Cluster 0 appears to represent stable cryptocurrencies with less variation in price changes.

Cluster 1 demonstrates more volatility, especially in mid-term time frames (14d, 30d, 60d), indicated by warmer colors.

![image](https://github.com/user-attachments/assets/2778bed7-2173-4569-8f0f-fd9ccf49542b)

**PCA Component Loadings and Heatmap Visualization**
PCA Fit: The Principal Component Analysis (PCA) is applied to reduce the dimensionality of the data, transforming the features into three principal components. This is crucial for simplifying the clustering process and revealing the most significant underlying patterns in the data.
PCA Loadings Heatmap: A heatmap is used to visualize the contribution of each feature to the principal components. This helps in understanding how various price-change features (e.g., 24h, 7d, 30d) influence the principal components.

**Insights:**
PCA1 captures short-term market dynamics, while PCA2 is more reflective of mid-term performance.
The heatmap shows the relationship between features and their importance to each component, which can guide investors in decision-making.

![image](https://github.com/user-attachments/assets/d96f0583-b686-41ee-b64d-8d1a7124ebf1)

**Cluster-Based Cryptocurrency Analysis**
After performing PCA and KMeans clustering, the cryptocurrencies are categorized into different clusters. Specific cryptocurrencies such as Bitcoin, Ethereum, Chainlink, and Cardano are identified within certain clusters that exhibit consistent performance over both short-term and long-term horizons, highlighting investment opportunities.

**Final Analysis**

![image](https://github.com/user-attachments/assets/83dbcc9d-e74a-4404-bd3a-b8a9c692eccf)
price_change_percentage_24 v/s price_change_percentage_7d
- This scatter plot presents a clearer distribution of cryptocurrencies based on their price change behaviors, showing four distinct clusters.
- The Yellow cluster represents cryptocurrencies that are consistently increasing over both timeframes, short term (24 hrs) and long term (7 days) indicating steady growth.
- The Blue cluster contains cryptocurrencies that are either stable or experiencing slight decline, especially over 24 hours.
- The Green cluster shows highly volatile behavior with a dramatic decline in 24 hours but recovery over 7 days, indicating sudden market shocks.
- The Red cluster consists of a very small number of points, with modest +ve change over 24 hours and slightly +ve over 7 day period.
- This scatter plot reflects a slight positive correlation between 24 hours and 7 days price change, specially for orange cluster.

The Mid-Term Cluster - yellow cluster - Cluster 2 (k=4) highlights promising investment opportunities in cryptocurrencies that demonstrate consistent growth across both short-term (24 hours) and long-term (7 days) timeframes. This cluster indicates steady upward momentum, making these assets particularly attractive for investors seeking reliable performance.

This includes notable cryptocurrencies such as Bitcoin, Ethereum, Bitcoin Cash, Binance Coin, Chainlink, Cardano, Litecoin, Monero, Tezos, Cosmos, Wrapped Bitcoin, Zcash, and Maker. These assets show potential for sustained growth over a mid-term horizon, further supporting strategic investment decisions.

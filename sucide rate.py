
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv("master.csv")
df.head()
# Select the relevant columns
df = df[['country', 'suicides/100k pop', 'gdp_per_capita ($)']]
# Remove missing values
df = df.dropna()

# Normalize the data
df_norm = (df[['suicides/100k pop', 'gdp_per_capita ($)']] - df[['suicides/100k pop', 'gdp_per_capita ($)']].mean()) / df[['suicides/100k pop', 'gdp_per_capita ($)']].std()

# Cluster the data
kmeans = KMeans(n_clusters=3)
kmeans.fit(df_norm)

# Add the cluster labels as a new column
df['cluster'] = kmeans.labels_


# Plot the clusters
plt.scatter(df['suicides/100k pop'], df['gdp_per_capita ($)'], c=df['cluster'])
plt.xlabel('Suicides per 100k population')
plt.ylabel('GDP per capita ($)')
plt.title('Clustering countries based on suicide rates and GDP per capita')
plt.show()


# Print the cluster centers
print(kmeans.cluster_centers_)


from scipy.optimize import curve_fit

# Define the model function
def model_func(x, a, b, c):
    return a * x**2 + b * x + c

# Define the err_ranges function
def err_ranges(x, func, param, sigma):
    """
    Calculates the upper and lower limits for the function, parameters, and sigmas for a single value or array x.
    Function values are calculated for all combinations of +/- sigma, and the minimum and maximum are determined.
    Can be used for any number of parameters and sigmas >= 1.
    """

    import itertools as iter

    # Initiate arrays for lower and upper limits
    lower = func(x, *param)
    upper = lower

    uplow = []  # List to hold upper and lower limits for parameters
    for p, s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))

    pmix = list(iter.product(*uplow))

    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)

    return lower, upper

# Fit the model to the data using curve_fit
x_data = np.array([1, 2, 3, 4, 5])  # Example x-values
y_data = np.array([2, 5, 9, 15, 23])  # Example y-values

params, _ = curve_fit(model_func, x_data, y_data)

# Generate predictions for future values
future_x = np.array([6, 7, 8, 9, 10])  # Example future x-values
predictions = model_func(future_x, *params)

# Estimate confidence range using err_ranges function
sigma = np.array([0.1, 0.2, 0.3])  # Example sigma values for confidence range
lower, upper = err_ranges(future_x, model_func, params, sigma)

# Plot the fitted curve and confidence range
plt.plot(x_data, y_data, 'bo', label='Data')
plt.plot(future_x, predictions, 'r-', label='Fitted Curve')
plt.fill_between(future_x, lower, upper, color='gray', alpha=0.3, label='Confidence Range')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Fitted Curve with Confidence Range')
plt.legend()
plt.show()


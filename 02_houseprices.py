import sklearn
import pandas as pd
import os

#for visualizing
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

def load_housing_data():
    csv_path = os.path.join('./housing', "housing.csv")
    return pd.read_csv(csv_path)

#helper function for viz
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join('./02_img', fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

#Load
housing = load_housing_data()

#Print some info about the data set,you can delete this
print(housing.head())
print(housing.info())
print(housing.describe())

#Show
housing.hist(bins=50, figsize=(20,15))
save_fig("attribute_histogram_plots")
plt.show()


#Start

# 1. use train_test_split to split the data
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
# Question: why the shuffle parameter?

# (1.5 Visualize your training data with matplotlib)

# 2. Look for correlations using the standard correlation coefficient
# save these in a variable called 'corr_matrix'
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.corr.html
# https://www.khanacademy.org/math/ap-statistics/bivariate-data-ap/correlation-coefficient-r/v/correlation-coefficient-intuition-exampless

# 3. Compare these to the median house value

# 4. Fix the total_bedrooms problem

# 5. Use a Linear Regression model
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

#6. Evaluate using Cross-Validation
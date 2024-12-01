import pandas as pd

# import dataset
data = pd.read_csv(r'model/data.csv')

# select columns
reduced_data = data[['smoothness_mean', 'concavity_mean', 'concave points_mean', 'fractal_dimension_mean',
                     'smoothness_se', 'symmetry_se', 'area_worst', 'concave points_worst', 'diagnosis']]

# export result
reduced_data.to_csv(r'model/reduced_data.csv')
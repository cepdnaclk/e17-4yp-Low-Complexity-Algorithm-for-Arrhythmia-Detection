import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

# Path to your CSV files
csv_file = "p_signal_108.csv"
result_file = "QRS.csv"

data = []
# Read the data from the first CSV file
with open(csv_file, 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        data.append(float(row[0]))  # Assuming the values are in the first column

# Read the result values from the second CSV file
result = []
with open(result_file, 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        result.append(int(row[0]))  # Assuming the values are in the first column

def interpolate_subarray(sub_array, num_points):
    old_indices = np.arange(0, len(sub_array))
    new_indices = np.linspace(0, len(sub_array) - 1, num_points)
    interpolator = interpolate.interp1d(old_indices, sub_array, kind='linear')
    new_sub_array = interpolator(new_indices)
    return new_sub_array

    
# Create a scatter plot of points where result value is 1
scatter_value = [data[i] for i, val in enumerate(result[:5000]) if val == 1]
scatter_points = [i for i, val in enumerate(result[:5000]) if val == 1]
plt.plot(data[:5000])
plt.scatter(scatter_points, scatter_value, color='red')
plt.xlabel('Data Index')
plt.ylabel('Data Value')
plt.title('Scatter Plot of Points with Result Value = 1')
plt.show()
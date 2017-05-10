import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# The idea for this balancing and plotting function came from Alex 
# Staravoitau's blog: http://navoshta.com/end-to-end-deep-learning/

# Read in driving_log.csv as pandas dataframe.
df = pd.read_csv('sim_data_10_lap/driving_log.csv', names=['C_cam', 'L_cam', 'R_cam', 'steering', 'num1', 'num2', 'num3'])

balanced = pd.DataFrame()   # Balanced dataset
bins = 1000                 # N of bins
bin_n = 60                  # N of examples to include in each bin (at most)

# Iterate through the number of 'bins' between 0 and 1. Add bin_n number
# of data points to the 'balanced' dataframe making sure to match the 
# abs(angle) to the angle range of the bin being filled. Abs() is used 
# because we know we're flipping all the images later anyway. This process
# evens out the angle data in the image set.
start = 0.0
for end in np.linspace(0, 1, num=bins):
    df_range = df[(np.absolute(df.steering) > start) & (np.absolute(df.steering) < end)]
    range_n = min(bin_n, df_range.shape[0])
    print(range_n)
    if range_n > 0:
        balanced = pd.concat([balanced, df_range.sample(range_n)])
    else:
        pass
        print('Failed at start:', start, 'end:', end)
    start = end

# Plot the before and after historgrams.
plottable_balanced = pd.DataFrame(columns=['steering'])
plottable_balanced['steering'] = balanced['steering'].abs() * 25.0

plottable_original = pd.DataFrame(columns=['steering'])
plottable_original['steering'] = df['steering'].abs() * 25.0

plottable_original.hist(column='steering', bins=100)
plt.title('Original Dataset Steering Angle, Total Frames: {}'.format(plottable_original.shape[0]))
plt.axis('tight')

plottable_balanced.hist(column='steering', bins=bins*4)
plt.title('Training Dataset Steering Angle, Total Frames: {}'.format(plottable_balanced.shape[0]))
plt.axis('tight')
plt.show()


print('Total Rows', balanced.shape[0])
balanced.to_csv('sim_data_10_lap/driving_log_balanced.csv', index=False, header=False)



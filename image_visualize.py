import csv
import os
import math
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
from keras.models import load_model
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("model")
args = parser.parse_args()

model = load_model(args.model)

csv_rows = []

with open('sim_data_10_lap/driving_log_balanced.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        csv_rows.append(line)

camera_number = random.randint(0,2)
line_number = random.randint(0, len(csv_rows))

image_filename = 'sim_data_10_lap/IMG/'+csv_rows[line_number][camera_number].split('/')[-1]

# Pick a random image from the directory.
if camera_number == 0:
    print('Center Image')
    train_angle = float(csv_rows[line_number][3]) * 25.0
    print('CNN Training angle:', round(train_angle, 5))
elif camera_number == 1:
    print('Left Image')
    train_angle = float(csv_rows[line_number][3]) * 25.0
    print('CNN Training angle:', round(train_angle, 5))
elif camera_number == 2:
    print('Right Image')
    train_angle = float(csv_rows[line_number][3]) * 25.0
    print('CNN Training angle:', round(train_angle, 5))


img = plt.imread(image_filename)
image_array = np.asarray(img)
steering_angle = float(model.predict(image_array[None, :, :, :], batch_size=1))
steering_angle = steering_angle * 25.0
print('CNN predicted angle:', round(steering_angle, 5))

# Display the image as a figure.
plt.imshow(img)

# Calculate the horizontal shift of the endpoint of the predicted angle's
# line. Plot the line from the center bottom of the used image along the
# predicted angle.
train_offset = int(math.tan(math.radians(train_angle)) * 50.0) + 159
plt.plot([159, train_offset], [130, 80], 'c-', lw=4, alpha=0.8)

# Same idea as before, just for the known angle of steering used in training.
steering_offset = int(math.tan(math.radians(steering_angle)) * 50.0) + 159
plt.plot([159, steering_offset], [130, 80], 'r-', lw=4, alpha=0.8)

title = 'Cyan Training Angle: ' + str(round(train_angle,3)) + ' Red Predicted Angle: ' + str(round(steering_angle,3))
plt.title(title)

# Add overlay to show what part of the image is cropped by the CNN.
plt.axhspan(0, 60, hold=None, alpha=0.75)
plt.axhspan(130, 159, hold=None, alpha=0.75)
plt.show()














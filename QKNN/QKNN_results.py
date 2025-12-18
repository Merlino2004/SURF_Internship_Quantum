import sys
sys.path.append("..")

import numpy as np
from QKNN import QKNN

from CoherenceCalculation.coherence import CoherenceCalculator
from qiskit_ibm_runtime.fake_provider import FakeMontrealV2

fake_backend = FakeMontrealV2() # Fake backend

coherence_calculator = CoherenceCalculator(
    backend=fake_backend,
    SDK='qiskit',
    inputs=None,
    q_params=None
)

# Test Data
testing_image1 = [[0.5,0],[0.5,0]]
testing_image2 = [[0.3,0.3],[0.3,0.3]]
testing_image3 = [[0.4,0],[0,0.9]]
testing_image4 = [[0.2,0],[0.2,0]]

# Train Data
image1 = [[0,0],[0,0]] # Full black
image2 = [[1,1],[1,1]] # Fully white
image3 = [[1,0],[1,0]] # Vertical stripe
image4 = [[0,1],[0,1]] # Vertical stripe
image5 = [[1,1],[0,0]] # Horizontal stripe
image6 = [[0,0],[1,1]] # Horizontal stripe
image7 = [[1,0],[0,1]] # Backward slash
image8 = [[0,1],[1,0]] # Forward slash

# Creating one training array
training_images = [image1, image2, image3, image4, image5, image6, image7, image8]
testing_images = [testing_image1, testing_image2, testing_image3, testing_image4]

# Labels of images
train_labels = np.array([0,0,1,1,2,2,3,3])
test_labels = np.array([1,0,3,1])

#Calculate the number of index qubits
n_index_qubits = int(np.ceil(np.log2(len(training_images))))

# Calculate the number of pixel qubits
h, w = np.array(image1).shape # Since all images have the same resolution
pixels = h*w
amplitudes = 2*pixels
n_pixel_qubits = int(np.log2(amplitudes))

# Give the images the right type of data
training_data = [row for img in training_images for row in img]

# Train the QKNN
qknn = QKNN(training_data, train_labels,n_index_qubits=n_index_qubits,n_pixel_qubits=n_pixel_qubits)

correct = 0 # Amount of correct predictions

# Test QKNN using the test images
for i in range(len(test_labels)):
  # Predict the test image using the overlap
  overlaps, y_pred, swap_test_qc = qknn.forward(testing_images[i], n_neighbors=3)

  print(f'Highest overlap for image {i+1} is from image {np.argmax(overlaps)+1} with {overlaps[np.argmax(overlaps)]*100:.0f}% Overlap')
  print('y_exp:', test_labels[i])
  print('y_pred:', y_pred)

  # If the prediction is correct, increase the amount of correct prediction with 1
  if y_pred == test_labels[i]:
    correct += 1

  total_time = coherence_calculator.forward(swap_test_qc)
  print(f'Coherence time of circuit {i+1}: {total_time*1e6:.3f} \u03BCs')

# Calculate the percentage of correct guesses
correct_percentage = correct/len(test_labels)*100

print(f'Percentage of correct predicted images: {correct_percentage}%')

# Visualize diagram of test data
fig = swap_test_qc.draw('mpl')
fig.savefig("Figures/QKNN_Visual.png")
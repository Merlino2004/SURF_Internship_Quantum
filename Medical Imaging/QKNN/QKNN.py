from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit import transpile
from qiskit.quantum_info import Statevector

import numpy as np

class QKNN:
    def __init__(self, training_data, real_labels, n_index_qubits=2, n_pixel_qubits=3):
        """
        The QKNN is an quantum implementation of the classical K-Nearest Neighbor algorithm. The application uses amplitude encoding
        to create data with a logarithmic complexity. Then Control-SWAP gates are used to measure the overlap between the given states.
        When the overlap is small, the output amplitude of state 0 is larger than state 1. Vice versa indicates indicates a large overlap. Then the
        application predicts the label using a weighted voting scheme, taking the amount of neighbors given into account.

        Parameters
        ----------
        training_data: an array of data used to train the application. The data can contain data from different implementations.
                       The data just needs to be in one final array.
        real_labels: the real labels of the training data.
        n_index_qubits: the number of qubits used to index the different data blocks within the statevector.
        n_pixel_qubits: the number of qubits used to encode each individual data point.
        """  
        self.training_data = training_data
        self.real_labels = real_labels
        self.n_index_qubits = n_index_qubits
        self.n_pixel_qubits = n_pixel_qubits

    def image_to_qc(self, image, circuit_name):
        """
        Amplitude encodes the given data into a quantum state. Tested on images, which gives it its name.

        Parameters
        ----------
        image: an array of data that will be amplitude encoded. Can be multiple arrays in one array.
        circuit_name: name of the circuit when the circuit is drawn.

        Returns
        -------
        qc: the quantum circuit in which the state is encoded.
        statevector: the statevector of the quantum circuit.
        """
        norm_pixels = [] # Create an array for the normalized pixels

        # Create data so it contributes to the born rule
        for row in image:
            for pixel in row:
                norm_pixel = [np.sqrt(pixel), np.sqrt(1 - pixel)]
                norm_pixels.append(norm_pixel)

        statevector = [amp for pair in norm_pixels for amp in pair] # Create the statevector for the data

        #Normalize the statevector
        norm = np.linalg.norm(statevector)
        statevector = [amp / norm for amp in statevector]

        n_qubits = int(np.log2(len(statevector))) # Determine the amount of qubits needed in the circuit

        # Create a quantum circuit for the statevector made
        qc = QuantumCircuit(n_qubits)
        qc.initialize(statevector, qc.qubits)
        qc.name = circuit_name # Give the quantum circuit the given name
        return qc, statevector

    def split_into_blocks(self, psi5):
        """
        Splits the amplitude encoded states into separate states. This is needed, because the swaptest can only work on
        states with the same amount of qubits.

        Parameters
        ----------
        psi5: the full statevector of the quantum circuit.

        Returns
        -------
        blocks: A list where each element is a statevector representing a single data block.
        """

        # Checks if the statevector has the right length for the given qubit split
        dim_index = 2 ** self.n_index_qubits
        dim_pixel = 2 ** self.n_pixel_qubits
        if len(psi5) != dim_index * dim_pixel:
            raise ValueError("Statevector has wrong length for given qubit split.")

        # Creates a array for the blocks get put into
        blocks = [psi5[i*dim_pixel:(i+1)*dim_pixel] for i in range(dim_index)]

        # Normalizes the blocks so it can be used in a quantum system
        for i in range(len(blocks)):
          norm = np.linalg.norm(blocks[i])
          if norm == 0:
              raise ValueError("Encountered zero-norm block.")
          blocks[i] = blocks[i] / norm

        return blocks

    def overlap(self, blocks, statevector_test):
        """
        Measures the overlap between each state and the test state. First an ancilla qubit is put into a 50/50 superposition state
        using a Hadamard gate. Then the CSWAP gates determine the amount of overlap. When the Hadamard gates are applied again, the zero state
        represents the amount of overlap. The amount of overlap is then calculated using the equation.

        Parameters
        ----------
        blocks: a list where each element is a statevector representing a single data block.
        statevector_test: the quantum statevector representing the test state.

        Returns
        -------
        overlaps: an array of ratios where the training state overlaps with the test state.
        swap_test_qc: the last swap test quantum circuit created (useful for visualization)
        """
        #defining certain settings
        sim = AerSimulator()
        shots = 1024

        overlaps = [] #Creates an array for the overlaps

        n_test_qubits = int(np.log2(len(statevector_test))) #Determining the amount of qubits from the test state


        for block_statevector in blocks:
            n_block_qubits = int(np.log2(len(block_statevector)))
            swap_test_qc = QuantumCircuit(1 + n_test_qubits + n_block_qubits, 1) # Creates a swap test circuit for each block


            # Defines qubits for the swap test
            anc = swap_test_qc.qubits[0]
            test_q = swap_test_qc.qubits[1:1+n_test_qubits]
            block_q = swap_test_qc.qubits[1+n_test_qubits:]
            cbit = swap_test_qc.clbits[0]

            # Initializes the test and block qubits
            swap_test_qc.initialize(statevector_test, test_q)
            swap_test_qc.initialize(block_statevector, block_q)

            # Applies Hadamard gates to all the ancilla qubits
            swap_test_qc.h(anc)

            # Applies controlled-swap gates the the needed spots
            for k in range(n_test_qubits):
                swap_test_qc.cswap(anc, test_q[k], block_q[k])

            # Apply Hadamard to all ancilla qubit again
            swap_test_qc.h(anc)

            # Measure the ancilla qubit
            swap_test_qc.measure(anc, cbit)

            # Simulating the result of the circuit made
            tqc = transpile(swap_test_qc, sim)
            result = sim.run(tqc, shots=shots).result()
            counts = result.get_counts()

            # Calculates the overlap between the states when given the measurements
            p0 = counts.get("0", 0) / shots
            overlap_squared = 2 * p0 - 1 #since p0 = 1/2+1/2*overlap_squared, seen in papers.
            overlap = np.sqrt(max(0, overlap_squared))

            # Append the just calculated overlap to the overlaps array
            overlaps.append(overlap)

        return overlaps, swap_test_qc

    def label_predict(self, overlaps, k):
        """
        Predicts the label classically using a weighted voting scheme.

        Parameters
        ----------
        overlaps: an array of ratios where the training state overlaps with the test state.
        k: the number of neighbors to consider.

        Returns
        -------
        y_pred: the predicted label of the test state.
        """

        # Puts the overlaps into their own array with their own index
        result_arr = np.array([[i, overlaps[i]] for i in range(len(overlaps))])

        # If amount of neighbors given is too high, the amount is set to default.
        if k > len(result_arr):
            k = len(result_arr)
            print(f"Warning: k was reduced to the number of training samples ({k}).")

        # Determines the k amount with the highest overlap
        k_nearest_indices = result_arr[:, 1].argsort()[::-1][:k]

        # Determines the labels of the states with the highest overlap
        k_neighbor_labels = self.real_labels[k_nearest_indices]
        k_neighbor_overlaps = result_arr[k_nearest_indices, 1]

        # Weights neighbors with each other for the best prediction afterwards
        weighted_votes = {}
        for label, overlap_value in zip(k_neighbor_labels, k_neighbor_overlaps):
            weighted_votes[label] = weighted_votes.get(label, 0) + overlap_value

        # Caclulates the predicted label
        y_pred = max(weighted_votes, key=weighted_votes.get)

        return y_pred

    def forward(self, test_data, n_neighbors=1):
        """
        The main function to classify a test image.

        Parameters
        ----------
        test_data: the image/data to classify
        n_neighbors: the number of neighbors to consider for the prediction

        Returns
        -------
        overlaps: list of overlap ratios between training data and test data
        y_pred: predicted label using weighted voting
        swap_test_qc: the last swap test quantum circuit (for visualization)
        """

        # Call back the image_to_qc function to be determine the quantum circuits and statevectors of the train and test data
        qc_train, statevector_train = self.image_to_qc(self.training_data, 'Training')
        qc_test, statevector_test = self.image_to_qc(test_data, 'Testing')

        # Call back the split_into_blocks function to split the train data into different block with the right length
        blocks = self.split_into_blocks(statevector_train)

        # Call back the overlap function the determine the states with the most overlap
        overlaps, swap_test_qc = self.overlap(blocks, statevector_test)
        y_pred = self.label_predict(overlaps, n_neighbors)

        # Call back the label_predict function to predict the label with the test image
        return overlaps, y_pred, swap_test_qc

    @staticmethod
    def qc_to_images(qc, image_shape, shots=1024):
        """
        When given a quantum state which is amplitude encoded, this function can be used to determine the array of data
        to represent the images.

        Parameters
        ----------
        qc: the quantum circuit of the image.
        image_shape: the shape of the image, given that it is a squared image.
        shots: the amount of measurements that will calculate the pixels of the state.

        Returns
        -------
        images_from_qc: list of reconstructed images
        """

        # Determines some settings for the measurements
        sim = AerSimulator()
        tqc = transpile(qc, sim)

        # Measures the quantum circuit
        result = sim.run(tqc, shots=shots).result()
        counts = result.get_counts()

        # Determine the number of qubits in the circuit given
        n_qubits = qc.num_qubits

        # Sorts the counts
        ordered = [counts.get(format(i, f'0{n_qubits}b'), 0) for i in range(2**n_qubits)]

        # Orders the counts so it can be used to determine the pixel values
        pixels = [(ordered[i]*len(ordered)/(2*shots), ordered[i+1]*len(ordered)/(2*shots))
                  for i in range(0, len(ordered), 2)]
        
        # Extracts the first index of each pixel pair (representing the "white" channel or intensity)
        white_channel = [p[0] for p in pixels]

        pixels_per_image = image_shape * image_shape # Calculates the number of pixels per image
        num_images = len(white_channel) // pixels_per_image # Calculate the number of full images that can be formed from the extracted pixels

        images_from_qc = [] # Creates an array to store the reconstructed images

        # Put the calculated pixel values back into an image
        for i in range(num_images):
            start = i*pixels_per_image
            end = start+pixels_per_image 
            reshaped_image = np.array(white_channel[start:end]).reshape((image_shape,image_shape)) # Reshape the 1D array of pixel values into a 2D array so the image can be represented
            images_from_qc.append(reshaped_image)

        return images_from_qc

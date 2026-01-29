import pennylane as qml
from pennylane import specs
from qiskit import transpile

class CoherenceCalculator:
    def __init__(self, backend, SDK, inputs, q_params):
        """
        The CoherenceCalculator estimates the total execution time of a quantum circuit and compares it to the minimum coherence time
        (T2) of given backend. This is useful to quickly check whether a circuit could theoretically be run on a given backend without
        decoherence dominating  
        
        Parameters:
        -----------
        backend: chosen backend to simulate the circuit
        SDK: the Software Development Kit used to create the circuit
        inputs: data input encoded in the circuit
        q_params: quantum parameters of the circuit
        """

        self.backend = backend
        self.SDK = SDK
        self.inputs = inputs
        self.q_params = q_params

        # Compute properties of the backend 
        props = backend.properties() 
        dev_time = [q[1].value for q in props.qubits]  # Decoherence time of qubits of the backend used
        self.min_T2 = min(dev_time) # Minimum decoherence time
        print(f"Minimum qubit coherence time of chosen backend: {self.min_T2*1e6:.2f} \u03BCs")

        # Gate times in seconds
        self.gate_times_pennylane = {
            "Rot": 50e-9,
            "RX": 50e-9,
            "RY": 50e-9,
            "RZ": 50e-9,
            "CNOT": 300e-9,
            "CZ": 300e-9,
            "Hadamard": 50e-9,
            "PauliX": 50e-9,
            "PauliY": 50e-9,
            "PauliZ": 50e-9,
        }

    def circuit_time_pennylane(self, qnode, inputs, q_params):
        """
        Calculates the total execution time of a Pennylane quantum circuit

        Parameters:
        -----------
        qnode: qnode of the pennylane circuit
        inputs: data input encoded in the circuit
        q_params: quantum parameters of the circuit
        """

        # Find the gates used in the circuit
        specs_info = qml.specs(qnode)(inputs, q_params)
        resources = specs_info["resources"]

        total_time = 0.0 # Set begin total decoherence time to 0s

        # Add the gate times per gate to the total decoherence time
        for gate, count in resources.gate_types.items():
            if gate in self.gate_times_pennylane:
                total_time += count * self.gate_times_pennylane[gate]

        return total_time
    
    def circuit_time_qiskit(self, circuit):
        """
        Calculates the total execution time of a Qiskit quantum circuit 
        
        Parameters:
        -----------
        circuit: the quantum circuit made with Qiskit
        """

        # Finding the properties of the circuit
        tqc = transpile(circuit, self.backend)
        props = self.backend.properties()

        total_time = 0.0 # Set begin total decoherence time to 0s

        # Finds the names of the gates
        for instr, qargs, _ in tqc.data:
            name = instr.name

            # Skip the barrier and measurement gates
            if name in ['measure', 'barrier']:
                continue
            
            qubits = [q._index for q in qargs] # Number of qubits in the quantum circuit
            gate_time = props.gate_length(name, qubits) # Find the gate time of all gates used. Qiskit does this on his own.

            # Add the decoherence time of all gates 
            if gate_time is not None: 
                total_time += gate_time

        return total_time

    def forward(self, qnode_or_circuit):  
        """
        Computes the total circuit time of the quantum circuit made and checks if the quantum circuit can be run on the backend.
        
        Parameters:
        -----------
        qnode_or_circuit: quantum node or quantum circuit form Pennylane or Qiskit, qnode for Pennylane and circuit for Qiskit
        """

        # Computes the coherence time depending on the SDK
        if self.SDK == 'pennylane':       
            total_time = self.circuit_time_pennylane(qnode_or_circuit, self.inputs, self.q_params)
        else: #qiskit
            total_time = self.circuit_time_qiskit(qnode_or_circuit)

        # prints if the quantum circuit can be run on backend
        if total_time < self.min_T2:
            print(f"Circuit can be run on fake backend, cause it doesn't takes too long")
        else:
            print(f"Circuit cannot be run on fake backend, cause it takes too long")
        return total_time

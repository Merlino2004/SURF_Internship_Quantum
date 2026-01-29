from coherence import CoherenceCalculator

import pennylane as qml
import torch 
import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit_ibm_runtime.fake_provider import FakeMontrealV2
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

def pennylane_circuit(n_qubits, backend):
    dev = qml.device(
        "qiskit.aer",
        wires=n_qubits,
        backend=backend,
        shots=1024
    )

    @qml.qnode(dev, interface="torch", diff_method="finite-diff")
    def circuit(inputs, weights):
        qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
        for i in range(n_qubits):
            qml.RY(weights[i],wires=i)
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    
    return circuit 

def qiskit_circuit(inputs,weights,n_qubits):
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.ry(float(inputs[i]),i)
    
    for i in range(n_qubits):
        qc.ry(float(weights[i]),i)

    qc.measure_all()
    return qc

fake_backend = FakeMontrealV2() # Fake backend
noise_model = NoiseModel.from_backend(fake_backend) # noise model
sim_backend = AerSimulator(noise_model=noise_model, basis_gates=noise_model.basis_gates)

n_qubits = 27
inputs = torch.rand(n_qubits)
weights = torch.rand(n_qubits)

pl_circuit = pennylane_circuit(n_qubits,sim_backend)
pl_calc = CoherenceCalculator(
    backend=fake_backend,
    SDK='pennylane',
    inputs = inputs,
    q_params=weights
)

pl_time = pl_calc.forward(pl_circuit)
print(f'Pennylane circuit time: {pl_time*1e6:.3f} \u03BCs')

qk_circuit = qiskit_circuit(inputs,weights,n_qubits)
qk_calc = CoherenceCalculator(
    backend=fake_backend,
    SDK='qiskit',
    inputs=None,
    q_params=None
)

qk_time = qk_calc.forward(qk_circuit)
print(f'Qiskit circuit time: {qk_time*1e6:.3f} \u03BCs')

coherence_list = [pl_time*1e6,qk_time*1e6]

plt.bar(['Pennylane','Qiskit'],coherence_list)
plt.ylabel('Coherence time (\u03BCs)')
plt.title('Coherence time: Pennylane and Qiskit')
plt.grid()
plt.savefig(f'plots/coherence_time_Pennylane_and_Qiskit.png',dpi=120)
plt.show()
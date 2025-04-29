import torch
import torch.nn as nn

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator
from qiskit.primitives import BackendSamplerV2
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.connectors import TorchConnector

import numpy as np





class NewQuantumLayer(nn.Module):
    def __init__(self, num_qubits, num_qubits_pe, iterations):
        """
        This is a constructor for the NewQuantumLayer class. It initializes key attributes,
        including the number of qubits, number of phase encoding qubits, and the number of
        iterations for the layer. Additionally, it computes all relevant phases and initializes
        a parameterized tensor for the weights, allowing for learnable parameters during quantum
        optimization.

        :param num_qubits: The number of qubits involved in the quantum layer operations.
        :param num_qubits_pe: The number of qubits used for phase estimation in the quantum layer.
        :param iterations: The number of iterations applied within the quantum architecture.
        """

        super(NewQuantumLayer, self).__init__()
        self.num_qubits = num_qubits
        self.num_qubits_pe = num_qubits_pe
        self.iterations = iterations
        self.phases = NewQuantumLayer.return_all_phases(num_qubits_pe)
        self.weight = nn.Parameter(torch.randn((2 * iterations - 1) * (num_qubits - 1)), requires_grad=True)

    def forward(self, x):
        return CustomAngleFunction.apply(x, self.weight, self.num_qubits,
                                         self.num_qubits_pe, self.iterations, self.phases)

    @staticmethod
    def return_all_phases(n_qubits):
        """
        Determines all possible phase values for quantum states, given the number of qubits.
        Uses the binary representation of each quantum state to calculate its corresponding
        phase values in a quantum superposition.

        :param n_qubits: The number of qubits for which all possible phases are computed.
        :type n_qubits: int
        :return: A list of phase values corresponding to all possible quantum states for
            the given number of qubits.
        :rtype: list[float]
        """
        all_states = NewQuantumLayer.return_all_states(n_qubits)
        all_phases = []
        for state in all_states:
            phase = 0
            i = 1
            for bit in state[::-1]:
                phase += int(bit) * 2 ** (-i)
                i += 1

            all_phases.append(phase)
        return all_phases

    @staticmethod
    def return_all_states(n_qubits):
        """
        Generates and returns a list of all possible binary states for a given number of qubits.
        Each state represents a binary representation derived from a qubit configuration.
        The number of states returned will be 2 raised to the power of the number of qubits.

        :param n_qubits: The number of qubits for which to generate all binary states.
                         Must be a non-negative integer.
        :type n_qubits: int
        :return: A list of binary strings, where each string corresponds to a possible qubit state.
        :rtype: list[str]
        """

        all_states = ['0', '1']
        iteration = 0

        while len(all_states) < 2 ** n_qubits:
            new_states = []
            for state in all_states:
                state_zero = state + '0'
                state_one = state + '1'
                new_states.append(state_zero)
                new_states.append(state_one)
            all_states = new_states
            iteration += 1

        return all_states


    # QUANTUM PHASE ESTIMATION
    ##########################################################
    @staticmethod
    def qft_rotations(num_qubits, i):
        qc = QuantumCircuit(num_qubits, name='Rotate')
        for j in range(1, num_qubits - i):
            qc.cp(- np.pi / 2 ** j, i + j, i)

        return qc

    @staticmethod
    def qft_circuit(num_qubits):
        qc = QuantumCircuit(num_qubits, name='QFT')

        for i in range(num_qubits):
            qc.h(i)
            rotation = NewQuantumLayer.qft_rotations(num_qubits, i)
            if i != num_qubits - 1:
                qc = qc.compose(rotation)
            qc.barrier()
        return qc
    ##########################################################



    # ANGLES EVALUATION
    ##########################################################
    @staticmethod
    def return_only_angle(z_rot, y_rot):
        angle = 0
        for i in range(len(z_rot)):
            angle += np.arctan(z_rot[i]**2) + np.arctan(y_rot[i]**2)

        return angle

    @staticmethod
    def return_angles_for_u_gate(angle, power_of_two, qb_ansatz):
        new_angle = (2 ** power_of_two) * 2 * np.pi * angle
        beta = np.arctan(((2 ** qb_ansatz) - 2) * np.tan(new_angle) / (2 ** qb_ansatz))
        alpha = np.arccos(np.cos(new_angle) / (np.cos(beta) + 1e-10))

        gamma = -beta + np.arctan(np.cos(angle) / np.sin(angle))
        delta = -gamma

        return alpha, beta, gamma, delta
    ##########################################################


    @staticmethod
    def measure(x):
        return x

    @staticmethod
    def circuit_builder(num_qubits, num_qubits_pe, num_iterations):
        qc = QuantumCircuit(num_qubits + num_qubits_pe, num_qubits_pe)
        initialization_circuit = QuantumCircuit(num_qubits, name="Init")
        ansatz_circuit = QuantumCircuit(num_qubits, name="Ansatz")
        unitary_gates = QuantumCircuit(num_qubits_pe + 1, name='UGates')
        qft = NewQuantumLayer.qft_circuit(num_qubits_pe).inverse()
        qft.name = 'IFT'

        inputs = ParameterVector('x', num_qubits - 1)
        theta = [ParameterVector(f'θ_{i}', num_qubits - 1) for i in range(2 * num_iterations - 1)]

        for i in range(num_qubits - 1):
            initialization_circuit.h(i)
            initialization_circuit.rz(inputs[i], i)

        initialization_circuit.barrier()

        for j in range(0, 2 * num_iterations - 1, 2):
            qc_temp = QuantumCircuit(num_qubits - 1, name=f"Rot_{j}")
            for i in range(num_qubits - 1):
                if j != 0:
                    qc_temp.rz(theta[j - 1][i], i)
                qc_temp.ry(theta[j][i], i)
            for i in range(1, num_qubits - 1):
                qc_temp.cx(i - 1, i)

            ansatz_circuit = ansatz_circuit.compose(qc_temp)
            ansatz_circuit.barrier()


        angle = NewQuantumLayer.return_only_angle(inputs, theta[0])

        for j in range(1, 2 * num_iterations - 1, 2):
            angle += NewQuantumLayer.return_only_angle(theta[j], theta[j + 1])


        for i in range(1, num_qubits_pe + 1):
            unitary_gates.h(i)
            alpha, beta, gamma, delta = NewQuantumLayer.return_angles_for_u_gate(angle, i - 1, num_qubits - 1)
            u_gate = QuantumCircuit(1, name='U')
            u_gate.rz(beta, 0)
            u_gate.u(2 * alpha, delta, gamma, 0)
            control_u = u_gate.to_gate().control(1)
            unitary_gates.append(control_u, [i, 0])

        ansatz_circuit.mcx([i for i in range(num_qubits - 1)], num_qubits - 1)
        ansatz_circuit.barrier()

        qc = qc.compose(initialization_circuit)
        qc = qc.compose(ansatz_circuit)
        qc = qc.compose(unitary_gates.decompose(), [i for i in range(num_qubits - 1, num_qubits + num_qubits_pe)])
        qc = qc.compose(qft, qubits=[i for i in range(num_qubits, num_qubits + num_qubits_pe)])

        qc.barrier()
        qc.measure([i for i in range(num_qubits, num_qubits + num_qubits_pe)], [i for i in range(num_qubits_pe)])
        # print(qc.draw(fold=-1))

        sampler = BackendSamplerV2(backend=AerSimulator())

        qnn = SamplerQNN(circuit=qc, input_params=initialization_circuit.parameters,
                         weight_params=ansatz_circuit.parameters,
                         input_gradients=True,
                         sampler=sampler,
                         interpret=NewQuantumLayer.measure, output_shape=2 ** (num_qubits_pe)
                         )

        return TorchConnector(qnn)


class CustomAngleFunction(torch.autograd.Function):

    @staticmethod
    def return_only_angle(z_rot, y_rot):
        angle = torch.arctan((z_rot ** 2)).sum(dim=1) + torch.arctan((y_rot ** 2)).sum(dim=1)
        return angle

    @staticmethod
    def return_only_gradients(z_rot, y_rot):
        gradient_z = 2 * z_rot / (z_rot ** 4 + 1)
        gradient_y = 2 * y_rot / (y_rot ** 4 + 1)
        return torch.cat([gradient_z, gradient_y], dim=1)


    @staticmethod
    def forward(ctx, x, weight, num_qubits, num_qubits_pe, iterations, phases):

        ctx.num_qubits = num_qubits
        ctx.num_qubits_pe = num_qubits_pe
        ctx.iterations = iterations
        ctx.phases = torch.tensor(phases, device=x.device, requires_grad=False)
        batch_size = x.size(0)

        new_angle = CustomAngleFunction.return_only_angle(x, weight[:num_qubits - 1].unsqueeze(0).expand(batch_size, -1))

        for j in range(1, 2 * iterations - 1, 2):
            a = weight[j * (num_qubits - 1): (j + 1) * (num_qubits - 1)]
            b = weight[(j + 1) * (num_qubits - 1): (j + 2) * (num_qubits - 1)]

            new_angle = new_angle + CustomAngleFunction.return_only_angle(a.unsqueeze(0).expand(batch_size, -1),
                                                 b.unsqueeze(0).expand(batch_size, -1))


        new_angle = new_angle % 1

        diff = (new_angle.unsqueeze(1).expand(-1, 2**num_qubits_pe) - ctx.phases.unsqueeze(0).expand(batch_size, -1))

        numerator = torch.sin((2 ** num_qubits_pe) * torch.pi * diff)
        denominator = torch.sin(torch.pi * diff)

        denominator = torch.where(denominator.abs() < 1e-10, torch.ones_like(denominator) * 1e-10, denominator)

        new_probs = ((numerator / denominator) ** 2) / (2 ** (2 * num_qubits_pe))

        ctx.save_for_backward(x, weight, new_angle)

        return new_probs


    @staticmethod
    def backward(ctx, grad_output):
        inputs, w, angle = ctx.saved_tensors
        batch_size = inputs.shape[0]

        gradients = CustomAngleFunction.return_only_gradients(inputs, w[0:ctx.num_qubits - 1].unsqueeze(0).expand(batch_size, -1))

        for j in range(1, 2 * ctx.iterations - 1, 2):
            z = w[j * (ctx.num_qubits - 1): (j + 1) * (ctx.num_qubits - 1)].unsqueeze(0).expand(batch_size, -1)
            y = w[(j + 1) * (ctx.num_qubits - 1): (j + 2) * (ctx.num_qubits - 1)].unsqueeze(0).expand(batch_size, -1)
            gradient = CustomAngleFunction.return_only_gradients(z, y)
            gradients = torch.cat([gradients, gradient], dim=1)  # concat lungo feature

        delta = angle.unsqueeze(1).expand(-1, 2 ** ctx.num_qubits_pe) - ctx.phases.unsqueeze(0).expand(batch_size,
                                                                                                       -1)  # broadcast su batch
        a = (2 ** ctx.num_qubits_pe) * torch.pi
        b = torch.pi

        dp = ((torch.sin(a * delta)) / (torch.pow(torch.sin(b * delta), 3) + 1e-10)) * (
                a * torch.cos(a * delta) * torch.sin(b * delta) -
                b * torch.sin(a * delta) * torch.cos(b * delta))
        dp /= 2 ** (2 * ctx.num_qubits_pe - 1)


        dp *= grad_output


        input_gradients = (dp.unsqueeze(2) * gradients[:, :ctx.num_qubits - 1].unsqueeze(1)
                           .expand(batch_size, 2 ** ctx.num_qubits_pe, ctx.num_qubits - 1)).sum(dim=1)

        weight_gradients = (dp.unsqueeze(2) * gradients[:, ctx.num_qubits - 1:].unsqueeze(1)
                            .expand(batch_size, 2 ** ctx.num_qubits_pe, (2 * ctx.iterations - 1) * (ctx.num_qubits - 1))).sum(dim=1)

        return input_gradients, weight_gradients, None, None, None, None


class NewHybridNN(nn.Module):
    def __init__(self, qubits_for_representation, qubits_for_pe, num_iterations):
        """
        Initializes the NewHybridNN class, which combines classical convolutional layers with a quantum
        layer for hybrid neural network processing. The neural network architecture consists of several
        convolutional layers followed by max pooling, average pooling, and integration with a quantum
        layer for advanced feature representation and processing.

        :param qubits_for_representation: Number of qubits to be used specifically for the representation
                                           in the quantum layer.
        :type qubits_for_representation: int
        :param qubits_for_pe: Number of qubits used for phase estimation in the quantum layer.
        :type qubits_for_pe: int
        :param num_iterations: Number of iterations for quantum operations/algorithms executed within
                               the quantum layer.
        :type num_iterations: int
        """
        super(NewHybridNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=qubits_for_representation - 1, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=5, stride=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # self.q_layer = circuit_builder(qubits_for_representation, qubits_for_pe, num_iterations)
        self.q_layer = NewQuantumLayer(qubits_for_representation, qubits_for_pe, num_iterations)




    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.q_layer(x)

        return x



from unicodedata import lookup

import torch.nn as nn
import torch
import torch.nn.functional as F
from numpy.ma.core import identity, arange
from sympy import print_tree, print_glsl
from torch.autograd import Function
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit import Parameter
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit.primitives import StatevectorEstimator as Estimator
from torch.fx.experimental.migrate_gradual_types.constraint_generator import add_layer_norm_constraints
from torch.nn import KLDivLoss
from torch.sparse import log_softmax
from torchvision.models.detection import FasterRCNN

from utility import return_quantum_layer, quantum_layer_1d, TripletDataset, return_all_phases, return_quantum_labels_2d, return_matrix_distances, plot_loss, return_quantum_labels, return_quantum_labels_2d_new
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import time
from prova import quantum_circuit, return_angle_distances
import random
from new import circuit_builder, NewQuantumLayer
from torch.nn.utils import clip_grad_norm_
import torch.nn.init as init




device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class QuantumCircuitBuilder:
    def __init__(self, simulator, shots):
        self.num_qubits = 1
        self.simulator = simulator
        self.shots = shots

        self.theta = Parameter('theta')
        self.qc = QuantumCircuit(self.num_qubits)
        for i in range (self.num_qubits):
            self.qc.h(i)
            self.qc.ry(self.theta, 0)
        self.qc.measure_all()


    def _get_probability(self, theta_value):

        qc_with_value = self.qc.assign_parameters({self.theta: theta_value})
        compiled_circuit = transpile(qc_with_value, self.simulator)
        sim_result = self.simulator.run(compiled_circuit, shots=self.shots).result()
        counts = sim_result.get_counts()

        probability_of_0 = counts.get('0', 0) / self.shots
        return probability_of_0


class QuantumFunction(Function):

    @staticmethod
    def forward(ctx, x, q_circuit_builder, epsilon):
        ctx.q_circuit_builder = q_circuit_builder
        ctx.epsilon = epsilon
        batch_size = x.size(0)

        probabilities_of_0 = []
        for i in range(batch_size):
            probabilities_of_0.append(ctx.q_circuit_builder._get_probability(x[i].item()))

        probabilities_of_0 = torch.tensor(probabilities_of_0, dtype=torch.float32)
        ctx.save_for_backward(x, probabilities_of_0)
        return probabilities_of_0

    @staticmethod
    def backward(ctx, grad_output):
        x, probabilities_of_0 = ctx.saved_tensors

        grad_output = grad_output.to(device)
        grad_output = grad_output.view(32, 1, 1, 1)
        grad_input = torch.zeros_like(x)

        for i in range(x.size(0)):
            theta_plus = x[i].item() + ctx.epsilon
            theta_minus = x[i].item() - ctx.epsilon
            prob_plus = ctx.q_circuit_builder._get_probability(theta_plus)
            prob_minus = ctx.q_circuit_builder._get_probability(theta_minus)

            grad_input[i] = (prob_plus - prob_minus) / (2 * ctx.epsilon)

        grad_input = grad_input * grad_output

        return grad_input, None, None


class QuantumLayer(nn.Module):
    def __init__(self, simulator, epsilon, shots=1024):
        super(QuantumLayer, self).__init__()
        self.q_circuit_builder = QuantumCircuitBuilder(simulator, shots)
        self.epsilon = epsilon

    def forward(self, x):
        return QuantumFunction.apply(x, self.q_circuit_builder, self.epsilon)


class HybridNN(nn.Module):
    def __init__(self, n_qubits):
        super(HybridNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=n_qubits, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.q_layer = return_quantum_layer(n_qubits)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.avg_pool(x)
        x = x.squeeze(2).squeeze(2)
        x = self.q_layer(x)
        x =F.normalize(x)
        return x



class QCNN(nn.Module):
    def __init__(self, simulator=AerSimulator(), epsilon=0.001):
        super(QCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=1)
        # self.q_layer = QuantumLayer(simulator, epsilon)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        qc = QuantumCircuit(1)
        theta = Parameter('θ')
        qc.ry(theta, 0)
        estimator = Estimator()
        qnn = EstimatorQNN(circuit=qc, input_params=[theta], weight_params=[], estimator=estimator, input_gradients=True)

        self.quantum_layer = TorchConnector(qnn)


    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))

        x = self.pool(torch.relu(self.conv3(x)))
        x = F.adaptive_max_pool2d(x, (1, 1))
        x = x.max(dim=1, keepdim=True)[0]
        x = self.quantum_layer(x)
        x = x.view(x.size(0))

        # x = x.view(-1, 64 * 7 * 7)
        # x = self.fc1(x)
        # x = x.view(x.size(0))


        # x = self.q_layer(x)
        return x


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=2)


    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))

        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        return x


class NewHybridNN(nn.Module):
    def __init__(self, qubits_for_representation, qubits_for_pe, num_iterations):
        super(NewHybridNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=qubits_for_representation - 1, kernel_size=3, stride=1, padding=1)
        # self.fc1 = nn.Linear(in_features=16*6*6, out_features=qubits_for_representation - 1)
        # self.fc2 = nn.Linear(in_features=qubits_for_representation - 1, out_features= 2**qubits_for_pe)
        self.pool = nn.MaxPool2d(kernel_size=5, stride=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.q_layer = quantum_layer_1d(qubits_for_representation, qubits_for_pe)

        # self.q_layer = quantum_circuit(qubits_for_representation, qubits_for_pe)
        # self.q_layer = circuit_builder(qubits_for_representation, qubits_for_pe, num_iterations)
        self.q_layer = NewQuantumLayer(qubits_for_representation, qubits_for_pe, num_iterations)




    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))
        x = self.avg_pool(x)
        # x = x.squeeze(2).squeeze(2)
        x = x.view(x.size(0), -1)

        # x = torch.relu(self.fc1(x))
        x = self.q_layer(x)
        # x = torch.relu(self.fc2(x))
        # x = F.softmax(x, dim=1)

        return x




def kl_divergence(p, q):
    eps = 1e-8
    p = torch.clamp(p, eps, 1.0)
    q = torch.clamp(q, eps, 1.0)
    return torch.sum(p * torch.log(p / q), dim=1)

def custom_triplet_kl_loss(anchor, positive, negative, margin=0.5):
    d_ap = kl_divergence(anchor, positive)  # KL tra anchor e positive
    d_an = kl_divergence(anchor, negative)  # KL tra anchor e negative
    print(d_ap.mean(), d_an.mean())
    loss = F.relu(d_ap - d_an + margin)
    return loss.mean()



def distance(x, y):
    x_max, pos_x = torch.max(x, dim=1)
    y_max, pos_y = torch.max(y, dim=1)

    y_related = y[torch.arange(y.size(0)), pos_x]
    x_related = x[torch.arange(x.size(0)), pos_y]

    dist = 0.5 * (torch.abs(x_max - y_related) + torch.abs(y_max - x_related))

    return dist








if __name__ == '__main__':

    wanna_save = True

    # Hyper Parameters
    n_triples = 1000
    batch_size = 256
    n_epochs = 50
    weight_same = 0.25
    qubits_for_representation = 10
    qubits_for_pe = 3
    num_iterations = 5
    list_of_digits = [0, 1, 2, 3, 4, 5, 6, 7]


    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    all_phases = return_all_phases(qubits_for_pe)
    all_phase = torch.tensor(all_phases, requires_grad=False, device=device)






    # Transform definition
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    # Load the dataset
    train = datasets.MNIST(root='./data', train=True, transform=transform)
    test = datasets.MNIST(root='./data', train=False, transform=transform)

    # Build triplet dataset for the first training
    train_triplet_dataset = TripletDataset(train, list_of_digits, n_triples)
    triplets_train_loader = DataLoader(train_triplet_dataset, batch_size=batch_size, shuffle=True)
    print('NUM TRIPLETS: ', len(train_triplet_dataset))



    # Model initialization
    model = NewHybridNN(qubits_for_representation, qubits_for_pe, num_iterations).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    '''scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    max_lr=5e-4,
                                                    steps_per_epoch=len(triplets_train_loader),
                                                    epochs=n_epochs,
                                                    anneal_strategy='cos',
                                                    div_factor=25,
                                                    final_div_factor=1e2)'''

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.2)




    model.eval()
    with torch.no_grad():
        '''quantum_labels = return_quantum_labels_2d(model, train_triplet_dataset, device, list_of_digits,
                                                  qubits_for_pe, all_phases, print_labels=True, print_others=True)'''

        '''quantum_labels = return_quantum_labels_2d(model, train_triplet_dataset, device, list_of_digits,
                                                  qubits_for_pe, all_phases, print_labels=True, print_others=True)'''


    loss_per_epoch = []
    max_angle_dist = 1 / len(list_of_digits)
    criterion = nn.TripletMarginWithDistanceLoss(distance_function=KLDivLoss(reduction='batchmean'), margin=1.0)




    model.train()
    for epoch in range(n_epochs):
        running_loss = 0.0
        start = time.time()
        norm_reg_sum = 0
        reg_sum = 0
        dist_sum = 0

        for a, p, n in triplets_train_loader:
            anchor, pos, neg = a.to(device), p.to(device), n.to(device)

            optimizer.zero_grad()


            vi = model(anchor)
            vi = torch.clamp(vi, min=1e-10)

            vj = model(pos)
            vj = torch.clamp(vj, min=1e-10)

            vk = model(neg)
            vk = torch.clamp(vk, min=1e-10)


            reg = (-torch.pow(vi - vk, 2).sum(dim=1)).mean() * 0.75
            anc_norm = -torch.norm(vi, dim=1).mean() * 0.1
            dist = criterion(vi, vj, vk)
            loss = dist  + reg + anc_norm


            dist_sum += dist
            reg_sum += reg
            norm_reg_sum += anc_norm

            loss.backward()
            # clip_grad_norm_(model.parameters(), max_norm=10.0)

            '''for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        print(f"NaN in gradient of {name}")
                    elif param.grad.abs().mean() > 1e2:
                        print(f"Very large gradient in {name}: {param.grad.abs().mean()}")
                    elif param.grad.abs().mean() < 1e-3:
                        print(f"Very small gradient in {name}: {param.grad.abs().mean()}")
                else:
                    print(f"Gradient of {name} not computed")'''

            '''for name, param in model.named_parameters():
                print(name, param.grad.abs().mean())'''

            optimizer.step()
            running_loss += loss

        scheduler.step(metrics=running_loss)

        stop = time.time()
        print(f"Epoch [{epoch + 1}], Loss: {running_loss:.4f}, Time: {stop - start:.4f}, SAME: {dist_sum:.4f}, REG: {reg_sum:.4f} , NORM: {norm_reg_sum:.4f}")
        loss_per_epoch.append(running_loss.item())



    model.eval()
    with torch.no_grad():
        '''quantum_labels = return_quantum_labels_2d(model, train_triplet_dataset, device, list_of_digits,
                                                  qubits_for_pe, all_phases, print_labels=True, print_others=True)'''

        quantum_labels = return_quantum_labels_2d(model, train_triplet_dataset, device, list_of_digits,
                                                  qubits_for_pe, all_phases, print_labels=True, print_others=False)


        for key, value in quantum_labels.items():
            print(f"{key}: {value.tolist()}")

    plot_loss(loss_per_epoch)

    if wanna_save:
        torch.save(train_triplet_dataset, '2D_Models/0_7/centroids_finder/triplet_dataset.pt')
        torch.save(model.state_dict(), '2D_Models/0_7/centroids_finder/model.pth')
        torch.save(quantum_labels, '2D_Models/0_7/centroids_finder/quantum_labels.pt')


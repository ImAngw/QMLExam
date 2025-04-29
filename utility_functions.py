import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from random import choice





def plot_loss(data, title="Loss", xlabel="epochs", ylabel="Loss"):

    plt.figure(figsize=(8, 5))
    plt.plot(data, marker='o', linestyle='-', color='b', label="Dati")  # Grafico a linee
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def draw_circle_with_rays(dict_of_vectors, all_phases, print_others):
    fig, ax = plt.subplots()
    theta = np.linspace(0, 2 * np.pi, 300)
    x = np.cos(theta)
    y = np.sin(theta)
    ax.plot(x, y, 'black')

    colors = {0: 'blue', 1: 'green', 2: 'purple', 3: 'orange', 4: 'brown',
              5: 'violet', 6: 'black', 7: 'gray', 8: 'yellow', 9: 'lime'}

    for key, value in dict_of_vectors.items():
        centroid = dict_of_vectors[key][0]
        centroid = centroid.tolist()[0]
        position = np.argmax(centroid)

        angle = 0
        angle_max_prob = 360 * all_phases[position]

        for i in range(len(centroid)):
            angle += all_phases[i] * centroid[i]

        angle *= 2 * np.pi

        x_end = 1.1 * np.cos(angle)
        y_end = 1.1 * np.sin(angle)
        ax.plot([0, x_end], [0, y_end], color=colors[key],
                label=f'{key}: P: {centroid[position]:.2f}  A:{angle_max_prob:.0f}°')


        if print_others:
            for v in value[1]:
                angle = 0
                for i in range(len(centroid)):
                    angle += all_phases[i] * v[0][i]

                angle *= 2 * np.pi

                x_end = np.cos(angle)
                y_end = np.sin(angle)
                ax.plot([0, x_end], [0, y_end], color=colors[key])


    ax.set_aspect('equal')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.axhline(0, color='grey', linewidth=0.3)
    ax.axvline(0, color='grey', linewidth=0.3)
    plt.legend()

    plt.show()

def return_quantum_labels(model, triplets, device, list_of_digits, qubits_for_evaluation, all_phases, print_labels=False, print_others=False):
    quantum_labels = {list_of_digits[i]: [torch.zeros([1, 2**qubits_for_evaluation]).to(device), []] for i in range(len(list_of_digits))}
    quadruples = triplets.quadruples
    dataset = triplets.dataset

    for quadruple in quadruples:
        xi = dataset[quadruple[0]][0]
        xi = xi.unsqueeze(1)
        label_i = quadruple[3]

        xj = dataset[quadruple[2]][0]
        xj = xj.unsqueeze(1)
        label_j = quadruple[4]

        vi = model(xi.to(device))

        quantum_labels[label_i][0] += vi
        quantum_labels[label_i][1].append(vi.tolist())

        vj = model(xj.to(device))
        quantum_labels[label_j][0] += vj
        quantum_labels[label_j][1].append(vj.tolist())



    for digit in list_of_digits:
        quantum_labels[digit][0] = torch.div(quantum_labels[digit][0], len(quantum_labels[digit][1]))

    if print_labels:
        draw_circle_with_rays(quantum_labels, all_phases, print_others)

    return {key: value[0] for key, value in quantum_labels.items()}

class TripletDataset(Dataset):
    def __init__(self, dataset, list_of_digits, num_triplets, num_samples_for_type, quadruplets=None):
        self.dataset = dataset
        self.list_of_digits = list_of_digits
        self.num_triplets = num_triplets
        self.num_samples_for_type = num_samples_for_type
        self.quadruples = self.return_quadruplets() if quadruplets is None else quadruplets
        self.subset = self.return_subset()


    def __len__(self):
        return len(self.quadruples)

    def __getitem__(self, idx):
        anchor, pos, neg = self.subset[idx]
        return anchor, pos, neg

    def return_quadruplets(self):
        dict_of_indices = {}
        quadruples = []
        num_samples_for_type = self.num_samples_for_type

        for digit in self.list_of_digits:
            dict_of_indices[digit] = [i for i, label in enumerate(self.dataset.targets) if label == digit]


        for i, first_digit in enumerate(self.list_of_digits):

            for second_digit in self.list_of_digits:
                if first_digit != second_digit:
                    for j in range(num_samples_for_type):
                        a = choice(dict_of_indices[first_digit])
                        p = choice(dict_of_indices[first_digit])
                        n = choice(dict_of_indices[second_digit])
                        quadruples.append([a, p, n, first_digit, second_digit])

        return quadruples

    def return_subset(self):
        subset = []
        for quadruple in self.quadruples:
            a, first_digit = self.dataset[quadruple[0]]
            p, _ = self.dataset[quadruple[1]]
            n, second_digit = self.dataset[quadruple[2]]
            subset.append((a, p, n))
        return subset

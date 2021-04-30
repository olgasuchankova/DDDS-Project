from sklearn.neural_network import MLPClassifier, MLPRegressor
from torch.utils.data import Dataset

from data_compilation import features_loaded
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torch.nn.functional as F
import matplotlib.pylab as plt
import numpy as np

#
#
# clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
#
#
# print(clf.score(X_test, y_test))


torch.manual_seed(2)


class Net(nn.Module):

    # Constructor
    def __init__(self, D_in, H1, H2, D_out):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(D_in, H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, D_out)

    # Prediction
    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        x = self.linear3(x)
        return x


class NetTanh(nn.Module):

    # Constructor
    def __init__(self, D_in, H1, H2, D_out):
        super(NetTanh, self).__init__()
        self.linear1 = nn.Linear(D_in, H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, D_out)

    # Prediction
    def forward(self, x):
        x = torch.tanh(self.linear1(x))
        x = torch.tanh(self.linear2(x))
        x = self.linear3(x)
        return x


class NetRelu(nn.Module):

    # Constructor
    def __init__(self, D_in, H1, H2, D_out):
        super(NetRelu, self).__init__()
        self.linear1 = nn.Linear(D_in, H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, D_out)

    # Prediction
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        return x


def train(model, criterion, train_loader, validation_loader, optimizer, epochs=100):
    i = 0
    useful_stuff = {'training_loss': [], 'validation_accuracy': []}

    for epoch in range(epochs):
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            z = model((train_dataset.__getitem__(i).get('forces').view(-1, 6 * 15)).float())
            #z = int(z[0,0].item())
            y = torch.tensor([int(np.array(train_dataset.__getitem__(i).get('classification')))])
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()
            useful_stuff['training_loss'].append(loss.data.item())

        correct = 0
        for x, y in validation_loader:
            z = model((train_dataset.__getitem__(i).get('forces').view(-1, 6 * 15)).float())
            _, label = torch.max(z, 1)
            y = int(np.array(train_dataset.__getitem__(i).get('classification')))
            print(label)
            print(y)
            correct += (label == y).sum().item()

        accuracy = 100 * (correct / len(validation_dataset))
        useful_stuff['validation_accuracy'].append(accuracy)

    return useful_stuff


class RobotDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data, label, transform=None):
        self._labels = label
        self._temporal = data
        self.transform = transform

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'classification': self._labels[idx], 'forces': self._temporal[idx]}

        if self.transform:
            forces = self.transform(sample.get('forces'))
            sample.update({'forces': forces})

        return sample


num_feat = 4
labels, temporal, aggregate = features_loaded(flat=False, f_type='time', num_feats=num_feat)
X_train, X_test, y_train, y_test = train_test_split(temporal, labels, test_size=0.15, shuffle=True)

train_dataset = RobotDataset(X_train, y_train, transform=transforms.ToTensor())
validation_dataset = RobotDataset(X_test, y_test, transform=transforms.ToTensor())


print(train_dataset)

# Create the criterion function

criterion = nn.CrossEntropyLoss()

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, shuffle=True)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, shuffle=False)

input_dim = 6 * 15
hidden_dim1 = 50
hidden_dim2 = 50
output_dim = 10

cust_epochs = 10

# Train the model with sigmoid function
learning_rate = 0.01
model = Net(input_dim, hidden_dim1, hidden_dim2, output_dim)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
training_results = train(model, criterion, train_loader, validation_loader, optimizer, epochs=cust_epochs)

# # Train the model with tanh function
# learning_rate = 0.01
# model_Tanh = NetTanh(input_dim, hidden_dim1, hidden_dim2, output_dim)
# optimizer = torch.optim.SGD(model_Tanh.parameters(), lr=learning_rate)
# training_results_tanch = train(model_Tanh, criterion, train_loader, validation_loader, optimizer, epochs=cust_epochs)
#
# # Train the model with relu function
# learning_rate = 0.01
# modelRelu = NetRelu(input_dim, hidden_dim1, hidden_dim2, output_dim)
# optimizer = torch.optim.SGD(modelRelu.parameters(), lr=learning_rate)
# training_results_relu = train(modelRelu, criterion, train_loader, validation_loader, optimizer, epochs=cust_epochs)


# Compare the training loss
# plt.plot(training_results_tanch['training_loss'], label='tanh')
plt.plot(training_results['training_loss'], label='sigmoid')
# plt.plot(training_results_relu['training_loss'], label='relu')
plt.ylabel('loss')
plt.title('training loss iterations')
plt.legend()
plt.show()

# Compare the validation loss
# plt.plot(training_results_tanch['validation_accuracy'], label = 'tanh')
plt.plot(training_results['validation_accuracy'], label='sigmoid')
# plt.plot(training_results_relu['validation_accuracy'], label = 'relu')
plt.ylabel('validation accuracy')
plt.xlabel('Iteration')
plt.legend()
plt.show()

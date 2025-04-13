import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from tqdm import tqdm
import matplotlib.pyplot as plt

# Hyper Parameters
input_size = 784
num_classes = 10
num_epochs = 100


def load_data(batch_size):
    """
    MNIST Dataset
    """
    train_dataset = dsets.MNIST(root='./data',
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True)

    test_dataset = dsets.MNIST(root='./data',
                               train=False,
                               transform=transforms.ToTensor())

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)
    return train_loader, test_loader


class Net(nn.Module):
    """
    Neural Network Model
    """
    def __init__(self, input_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        return out


class DeeperNet(nn.Module):
    """
    Deeper Neural Network Model for section C:
    """
    def __init__(self, input_size, num_classes, hidden_dim=500):
        super(DeeperNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


def section_a(batch_size=100, learning_rate=1e-3, optimizer_type='SGD', model_type='a'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tqdm.write(f"Using device: {device}")

    train_loader, test_loader = load_data(batch_size=batch_size)

    if model_type == 'a':
        net = Net(input_size, num_classes).to(device)
    else:
        net = DeeperNet(input_size, num_classes, hidden_dim=500).to(device)

    criterion = nn.CrossEntropyLoss()

    if optimizer_type == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    elif optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    elif optimizer_type == 'Momentum':
        optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

    train_losses = []
    train_accs = []

    # Train the Model
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct, total = 0, 0

        loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch [{epoch + 1}/{num_epochs}]")

        for i, (images, labels) in loop:
            # Convert torch tensor to Variable
            images = images.view(-1, 28 * 28).to(device)
            labels = labels.to(device)

            # Forward + Backward + Optimize
            net.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            avg_loss = running_loss / (i + 1)
            accuracy = 100 * correct / total
            loop.set_postfix(loss=avg_loss, acc=f"{accuracy:.2f}%")

        # Record epoch metrics
        train_losses.append(avg_loss)
        train_accs.append(accuracy)

    # Evaluate on Test Data
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = Variable(images.view(-1, 28 * 28)).to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_acc = 100 * correct / total
    tqdm.write('Test Accuracy of the network: %d %%' % test_acc)

    # Save the trained model
    # torch.save(net.state_dict(), 'model.pkl')

    return train_losses, train_accs, test_acc


def section_b(batch_size, learning_rate, optimizer_type):
    return section_a(batch_size=batch_size, learning_rate=learning_rate, optimizer_type=optimizer_type, model_type='a')


def section_c(batch_size, learning_rate, optimizer_type, model_type):
    return section_a(batch_size=batch_size, learning_rate=learning_rate, optimizer_type=optimizer_type,
                     model_type=model_type)


if __name__ == '__main__':
    results = {}
    # Section A: Baseline model with default hyper-parameters (bs=100, lr=1e-3, optimizer='SGD')
    results['A'] = section_a()

    # Section B: Different configurations with the original (shallow) model
    results['B1'] = section_b(batch_size=128, learning_rate=0.001, optimizer_type='Adam')
    results['B2'] = section_b(batch_size=128, learning_rate=0.0005, optimizer_type='Adam')
    results['B3'] = section_b(batch_size=128, learning_rate=0.001, optimizer_type='Momentum')
    results['B4'] = section_b(batch_size=128, learning_rate=0.0005, optimizer_type='Momentum')

    # Section C: Deeper model with configuration: bs=128, lr=0.001, optimizer='Adam'
    results['C'] = section_c(batch_size=128, learning_rate=0.001, optimizer_type='Adam', model_type='c')

    # Plot training loss curves for all tests
    plt.figure(figsize=(10, 5))
    for key, (losses, accs, test_acc) in results.items():
        epochs = range(1, num_epochs + 1)
        plt.plot(epochs, losses, marker='o', label=f'{key} (Test Acc: {test_acc:.1f}%)')
    plt.title('Training Loss over Epochs for Different Configurations')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    configs = list(results.keys())
    test_accuracies = [results[key][2] for key in configs]
    plt.bar(configs, test_accuracies, color='skyblue')
    plt.title('Final Test Accuracy for Different Configurations')
    plt.xlabel('Configuration')
    plt.ylabel('Test Accuracy (%)')
    plt.ylim(0, 100)

    for i, v in enumerate(test_accuracies):
        plt.text(i, v + 1, f"{v:.1f}%", ha='center', fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

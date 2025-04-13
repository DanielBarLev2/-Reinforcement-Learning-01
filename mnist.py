import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from tqdm import tqdm

# Hyper Parameters
input_size = 784
num_classes = 10
num_epochs = 20
batch_size = 100
learning_rate = 1e-3

# MNIST Dataset
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


# Neural Network Model
class Net(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        return out


def section_a():
    tqdm.write(f"Using device: {torch.device("cuda" if torch.cuda.is_available() else "cpu")}")

    net = Net(input_size, num_classes)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

    # Train the Model
    for epoch in range(num_epochs):
        ### --- ###
        running_loss = 0.0
        correct, total = (0, 0)

        loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch [{epoch + 1}/{num_epochs}]")
        ### --- ###
        for i, (images, labels) in loop:
            # Convert torch tensor to Variable
            images = Variable(images.view(-1, 28*28))
            labels = Variable(labels)

            # Forward + Backward + Optimize
            # TODO: implement training code
            net.zero_grad()
            outputs = net.forward(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update progress bar
            avg_loss = running_loss / (i + 1)
            accuracy = 100 * correct / total
            loop.set_postfix(loss=avg_loss, acc=f"{accuracy:.2f}%")


    # Test the Model
    correct, total = (0, 0)
    for images, labels in test_loader:
        images = Variable(images.view(-1, 28*28))
        # TODO: implement evaluation code - report accuracy
        outputs = net.forward(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

    # Save the Model
    torch.save(net.state_dict(), 'model.pkl')


if __name__ == '__main__':
    section_a()
    """
    Expected results:
        Training Loss after 20 epochs: ~0.6
        Training Accuracy after 20 epochs: ~85%
        Accuracy of the network on the test images: ~86%
    """
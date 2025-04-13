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


def section_a(batch_size=100, learning_rate=1e-3, optimizer='SGD'):
    tqdm.write(f"Using device: {torch.device("cuda" if torch.cuda.is_available() else "cpu")}")

    train_loader, test_loader = load_data(batch_size=batch_size)

    net = Net(input_size, num_classes)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    if optimizer == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    elif optimizer == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    elif optimizer == 'Momentum':
        optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

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

def section_b(batch_size, learning_rate, optimizer):
    section_a(batch_size=batch_size, learning_rate=learning_rate, optimizer=optimizer)


if __name__ == '__main__':
    section_a()
    """
    Expected results: (Using original HPs and optimizer [defined in the signature])
        Training Loss after 20 epochs: ~0.6
        Training Accuracy after 20 epochs: ~85%
        Accuracy of the network on the test images: ~86%
    """
    section_b(batch_size=128, learning_rate=0.001, optimizer='Adam')
    section_b(batch_size=128, learning_rate=0.0005, optimizer='Adam')
    section_b(batch_size=128, learning_rate=0.001, optimizer='Momentum')
    section_b(batch_size=128, learning_rate=0.0005, optimizer='Momentum')
    """
    Test 1: bs = 128 ; lr = 0.001 ; optimizer = 'Adam'
        Training Loss after 20 epochs: ~0.24
        Training Accuracy after 20 epochs: ~93%
        Accuracy of the network on the test images: ~92% 
    Test 2: bs = 256 ; lr = 0.0005 ; optimizer = 'Adam'
        Training Loss after 20 epochs: ~0.26
        Training Accuracy after 20 epochs: ~92%
        Accuracy of the network on the test images: ~92% 
        
    Same results, but slower convergence in test 2.
    
    Test 3: bs = 128 ; lr = 0.001 ; optimizer = 'Momentum'
        Training Loss after 20 epochs: ~0.35
        Training Accuracy after 20 epochs: ~90%
        Accuracy of the network on the test images: ~90% 
    Test 4: bs = 256 ; lr = 0.0005 ; optimizer = 'Momentum'
        Training Loss after 20 epochs: ~0.47
        Training Accuracy after 20 epochs: ~88%
        Accuracy of the network on the test images: ~88% 
        
    Conclusion: Using a batch size of 128, a learning rate of 0.001,
     and the Adam optimizer yields superior performance compared to other configurations.
        
    """
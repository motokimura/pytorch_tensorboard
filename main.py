from __future__ import print_function
import argparse

parser = argparse.ArgumentParser(description='Pytorch & tensorboard example with MNIST')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disable CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status (default: 10)')
parser.add_argument('--log-dir', default=None, metavar='LD',
                    help='directory to output TensorBoard event file (default: runs/<DATETIME>)')
args = parser.parse_args()


# Setup TensorBoardX summary writer
from tensorboardX import SummaryWriter
from datetime import datetime
import os

if args.log_dir is None:
    args.log_dir = os.path.join('runs', datetime.now().strftime('%b%d_%H-%M-%S'))
writer = SummaryWriter(log_dir=args.log_dir)


# Import moddules for PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# Check CUDA/GPU availability
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Initialize the random seed
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Setup data loaders for train/test sets
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        'data', 
        train=True, 
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.01307,), (0.3081,))
        ])),
    batch_size=args.batch_size, 
    shuffle=True, 
    **kwargs
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        'data', 
        train=False, 
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.01307,), (0.3081,))
        ])),
    batch_size=args.test_batch_size, 
    shuffle=True, 
    **kwargs
)

# Define network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320) # (batch_size, units)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return F.log_softmax(x) # = log(y) = log(softmax(x)) = log(exp(x) / sum_k(exp(x_k)))

# Setup the network 
model = Net()
if args.cuda:
    model.cuda()

# Setup optimizer
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

# Training
def train(epoch):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        
        optimizer.zero_grad()
        output = model(data) # forward
        loss = F.nll_loss(output, target) # = sum_k(-t_k * log(y_k))
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0] )
            )

            # Log train/loss to TensorBoard at every iteration
            n_iter = (epoch - 1) * len(train_loader) + batch_idx + 1
            writer.add_scalar('train/loss', loss.data[0], n_iter)

    # Log model parameters to TensorBoard at every epoch
    for name, param in model.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram('{}/{}'.format(layer, attr), param.clone().cpu().data.numpy(), n_iter)

# Testing
def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0

    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)

        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss (later, averaged over all test samples)
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), test_accuracy)
    )

    # Log test/loss and test/accuracy to TensorBoard at every epoch
    n_iter = epoch * len(train_loader)
    writer.add_scalar('test/loss', test_loss, n_iter)
    writer.add_scalar('test/accuracy', test_accuracy, n_iter)


# Visualize network as a graph on TensorBoard
input_tensor = torch.Tensor(1,1,28,28)
if args.cuda:
    input_tensor = input_tensor.cuda()
res = model(Variable(input_tensor, requires_grad=True))
writer.add_graph(model, lastVar=res)


# Start training
for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)

# Close TensorBoardX summary writer
writer.close()

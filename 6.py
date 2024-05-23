import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import time
from tqdm import tqdm

torch.manual_seed(43)
BATCH_SIZE = 32

train_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=None
)
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
    target_transform=None
)

train_dataloader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=False)
test_dataloader = DataLoader(dataset=test_data,
                             batch_size=BATCH_SIZE,
                             shuffle=False)


class ResNetBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Conv2d(in_channels=10,
                      out_channels=10,
                      kernel_size=2,
                      stride=1,
                      padding=0,
                      device=dev),
            nn.ReLU(),
            nn.Conv2d(in_channels=10,
                      out_channels=10,
                      kernel_size=2,
                      stride=1,
                      padding=1,
                      device=dev)
        )
        self.active = nn.ReLU()

    def forward(self, x):
        out = self.layer_stack(x)
        return self.active(x + out)


class MyResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, padding=1, device=dev),
            nn.ReLU(),
            ResNetBlock(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResNetBlock(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResNetBlock(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            ResNetBlock(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=90, out_features=10, device=dev)
        )

    def forward(self, x):
        return self.layer_stack(x)


class_names = train_data.classes


def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn):
    train_loss = 0
    train_acc = 0
    model.train()
    for batch, (X, y) in enumerate(data_loader):
        y_pred = model(X.to(dev))
        y_pred = y_pred.to(dev)
        y = y.to(dev)

        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y,
                                 y_pred=y_pred.argmax(dim=1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 400 == 0:
            print(f"Looked at {batch * len(X)}/{len(train_dataloader.dataset)} samples")
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train acc: {train_acc:.2f} ")

if torch.cuda.is_available():
    dev = 'cuda:0'
else:
    dev = 'cpu'


def test_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              optimizer: torch.optim.Optimizer,
              accuracy_fn):
    test_loss, test_acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            test_pred = model(X)
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true=y,
                                    y_pred=test_pred.argmax(dim=1))
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test loss: {test_loss:.5f} | Test acc: {test_acc:.2f}%\n")


model_2 = MyResNet()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_2.parameters(), lr=0.1)
train_time_start = time.perf_counter()
epochs = 20


def accuracy_function(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


accuracy_fn = accuracy_function

for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch} \n-------")
    train_step(model=model_2,
               data_loader=train_dataloader,
               loss_fn=loss_fn,
               optimizer=optimizer,
               accuracy_fn=accuracy_fn)
    test_step(model=model_2,
              data_loader=test_dataloader,
              loss_fn=loss_fn,
              optimizer=optimizer,
              accuracy_fn=accuracy_fn)
train_time_end = time.perf_counter()
print(f"Elapsed time: {(train_time_end - train_time_start):0.4f} seconds")

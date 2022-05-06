import numpy as np
import pandas as pd
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
from torchvision import transforms

import random
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm


train_transform = transforms.Compose([transforms.RandomResizedCrop(128, (0.5, 1)), transforms.ToTensor()])
test_transform = transforms.Compose([transforms.Resize((128,128)), transforms.ToTensor()])

train_data_IF = ImageFolder('./garbage_classification/', transform=train_transform)
test_data_IF = ImageFolder('./garbage_classification/', transform=test_transform)

num_classes = len(train_data_IF.classes)

n_test = int(0.1 * len(train_data_IF))
test_idx = random.choices(range(len(train_data_IF)), k=n_test)

test_data = torch.utils.data.Subset(test_data_IF, test_idx)
train_data = torch.utils.data.Subset(train_data_IF, [idx for idx in range(len(train_data_IF)) if idx not in test_idx])

batch_size=128
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)

class ParallelCNN(nn.Module):
    def __init__(self, num_classes):
        super(ParallelCNN, self).__init__()
        self.conv1_16 = nn.Conv2d(3, 16, 1)
        self.conv3_16 = nn.Conv2d(3, 16, 3)
        self.conv5_16 = nn.Conv2d(3, 16, 5)

        self.conv1_32 = nn.Conv2d(16, 32, 1)
        self.conv3_32 = nn.Conv2d(16, 32, 3)
        self.conv5_32 = nn.Conv2d(16, 32, 5)

        self.conv3_16_16 = nn.Conv2d(16, 16, 3)
        self.conv5_16_16 = nn.Conv2d(16, 16, 5)

        self.max = nn.MaxPool2d(7)

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(33488, 2048)
        self.linear2 = nn.Linear(2048, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        line1 = self.conv1_16(x)
        line1 = self.relu(line1)
        line1 = self.conv1_32(line1)
        line1 = self.relu(line1)
        line1 = self.max(line1)
        line1 = self.flatten(line1)

        line3 = self.conv3_16(x)
        line3 = self.relu(line3)
        line3 = self.conv3_32(line3)
        line3 = self.relu(line3)
        line3 = self.max(line3)
        line3 = self.flatten(line3)

        line5 = self.conv5_16(x)
        line5 = self.relu(line5)
        line5 = self.conv5_32(line5)
        line5 = self.relu(line5)
        line5 = self.max(line5)
        line5 = self.flatten(line5)

        line135 = self.conv1_16(x)
        line135 = self.relu(line135)
        line135 = self.conv3_16_16(line135)
        line135 = self.relu(line135)
        line135 = self.conv5_16_16(line135)
        line135 = self.relu(line135)
        line135 = self.max(line135)
        line135 = self.flatten(line135)

        out = torch.cat((line1, line3, line5, line135), dim=1)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.linear2(out)

        return out

device = torch.device('cuda')
model = ParallelCNN(num_classes).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def train(model, device, loss_fn, optimizer):
    epochs = 50
    best_acc = 0
    writer = SummaryWriter()

    for epoch in range(epochs):
        running_loss = 0
        acc = 0
        batch_count = 0

        model.train()
        print(f'\nEpoch {epoch}:')
        for x, y in tqdm(train_loader):
            x, y = x.to(device), y.to(device)
            batch_count += 1

            optimizer.zero_grad()

            output = model(x)
            loss = loss_fn(output, y)
            loss.backward()

            running_loss += loss.item()

            optimizer.step()
        running_loss /= batch_count
        writer.add_scalar('Loss/Train', running_loss, epoch)

        batch_count = 0

        with torch.no_grad():
            model.eval()
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                batch_count += 1

                output = model(x)
                acc += (torch.argmax(output, dim=1) == y).sum().item()/len(y)
            acc /= batch_count
            writer.add_scalar('Acc/Eval', acc, epoch)

        print(f'Train loss {running_loss:.6f} \tTest Accuracy {acc:.6f}')

        if acc > best_acc:
            print('New best model...', end='')
            best_acc = acc
            torch.save(model.state_dict(), './model/state_dict.pt')
            print('saved')

    writer.flush()
    writer.close()

train(model, device, loss_fn, optimizer)

model.load_state_dict(torch.load('./model/state_dict.pt'))

compiled_model = torch.jit.script(model)
torch.jit.save(compiled_model, './model.pt')

images = random.choices(test_data, k=6)
predictors = [torch.Tensor(i[0]) for i in images]

with torch.no_grad():
    model.eval()
    for i in range(6):
        prediction = test_data_IF.classes[torch.argmax(model(predictors[i].unsqueeze(0).to(device)))]
        plt.subplot(2, 3, i+1)
        plt.imshow(torch.transpose(torch.transpose(images[i][0], 0, 1), 1, 2))
        plt.title(prediction)
        plt.xticks([])
        plt.yticks([])
    plt.show()

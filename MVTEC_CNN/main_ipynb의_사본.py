# -*- coding: utf-8 -*-
"""main.ipynb의 사본

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1bAZP_H_MSnZy4S1Fw4cOOvUtvL_Qh8zm
"""

DIR = '/content/drive/MyDrive/sk_personal/train_new'
test_DIR = '/content/drive/MyDrive/sk_personal/test_new'
learning_rate = 0.001
training_epochs = 50
batch_size = 16
num_classes =4

import torch
import torch.nn.init
import os
import pandas as pd
from torchvision.io import read_image
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor

from google.colab import drive
drive.mount('/content/drive')

class CustomImageDataset():
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")  # PIL Image로 로드
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.AutoAugment(policy=transforms.autoaugment.AutoAugmentPolicy.IMAGENET, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor()
])

dataset = CustomImageDataset(annotations_file='/content/drive/MyDrive/sk_personal/train_new/train_fianl.csv',img_dir = DIR,transform=transform)

test_dataset = CustomImageDataset(annotations_file='/content/drive/MyDrive/sk_personal/test_new/test_new_3.csv',img_dir = test_DIR,transform=transform)

from torch.utils.data import DataLoader
from PIL import Image


train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=19, shuffle=False)

# 이미지와 정답(label)을 표시합니다.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[2].squeeze()
print(img)
img = img.T

label = train_labels[2]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")

print(train_features[0][2][64])

class CustomConvNet(torch.nn.Module):
    def __init__(self, num_classes):
        super(CustomConvNet, self).__init__()

        self.layer1 = self.conv_module(3, 32)
        self.layer2 = self.conv_module(32, 64)
        self.layer3 = self.conv_module(64, 128)
        self.layer4 = self.conv_module(128, 256)
        self.layer5 = self.conv_module(256,512)
        self.gap = self.global_avg_pool(512, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.gap(out)
        out = out.view(-1, num_classes)

        return out

    def conv_module(self, in_num, out_num):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_num, out_num, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(out_num),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

    def global_avg_pool(self, in_num, out_num):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_num, out_num, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(out_num),
            torch.nn.LeakyReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

custom_model = CustomConvNet(num_classes=num_classes).to(device)

print(custom_model)

criterion = torch.nn.CrossEntropyLoss().to(device)    # 비용 함수에 소프트맥스 함수 포함되어져 있음.
optimizer = torch.optim.Adam(custom_model.parameters(), lr=learning_rate)

total_batch = len(train_dataloader)
print('총 배치의 수 : {}'.format(total_batch))

for epoch in range(training_epochs):
    avg_cost = 0

    for train_features, train_labels in train_dataloader: # 미니 배치 단위로 꺼내온다. X는 미니 배치, Y느 ㄴ레이블.
        # image is already size of (28x28), no reshape
        # label is not one-hot encoded
        train_features = train_features.to(device)
        train_labels = train_labels.to(device)

        optimizer.zero_grad()
        hypothesis = custom_model(train_features)
        cost = criterion(hypothesis, train_labels)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))

for e in range(training_epochs):
    avg_cost = 0
    for train_features, train_labels in train_dataloader: # 미니 배치 단위로 꺼내온다. X는 미니 배치, Y느 ㄴ레이블.
        # image is already size of (28x28), no reshape
        # label is not one-hot encoded
        train_features = train_features.to(device)
        train_labels = train_labels.to(device)

        # Forward pass
        outputs = custom_model(train_features)
        loss = criterion(outputs, train_labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_cost += loss / total_batch

    print('[Epoch: {:>4}] cost = {:>.9}'.format(e + 1, avg_cost))

torch.save(custom_model.state_dict(), '/content/drive/MyDrive/sk_personal/model_1.pt')

custom_model = CustomConvNet(num_classes=4)
custom_model.load_state_dict(torch.load('/content/drive/MyDrive/sk_personal/model_1.pt'))

custom_model.to(device)

custom_model.eval()
correct = 0
total = 0
guess = []
correct_ans = []
with torch.no_grad():
    for test_features, test_labels in test_dataloader: # 미니 배치 단위로 꺼내온다. X는 미니 배치, Y느 ㄴ레이블.
        # image is already size of (28x28), no reshape
        # label is not one-hot encoded
        test_features = test_features.to(device)
        test_labels = test_labels.to(device)
        outputs = custom_model(test_features)
        _, predicted = torch.max(outputs.data, 1)

        total += len(test_labels)
        correct += (predicted == test_labels).sum().item()
print(predicted, test_labels)
print('Test Accuracy of the model on the {} test images: {} %'.format(total, 100 * correct / total))

custom_model.eval()
correct = 0
total = 0
guess = []
correct_ans = []
with torch.no_grad():
    for train_features, train_labels in train_dataloader: # 미니 배치 단위로 꺼내온다. X는 미니 배치, Y느 ㄴ레이블.
        # image is already size of (28x28), no reshape
        # label is not one-hot encoded
        train_features = train_features.to(device)
        train_labels = train_labels.to(device)
        outputs = custom_model(train_features)
        _, predicted = torch.max(outputs.data, 1)

        total += len(train_labels)
        correct += (predicted == train_labels).sum().item()
print(predicted, train_labels)
print('train Accuracy of the model on the {} train images: {} %'.format(total, 100 * correct / total))

from torchvision import models

resnet50_pretrained = models.resnet50(pretrained=True)

print(resnet50_pretrained)

from torchvision import models

resnet101_pretrained = models.resnet101(pretrained=True)

print(resnet101_pretrained)

num_ftrs = resnet50_pretrained.fc.in_features
resnet50_pretrained.fc = torch.nn.Linear(num_ftrs, num_classes)

resnet50_pretrained.to(device)

num_ftrs = resnet101_pretrained.fc.in_features
resnet101_pretrained.fc = torch.nn.Linear(num_ftrs, num_classes)

resnet101_pretrained.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, resnet101_pretrained.parameters()), lr=learning_rate)

for e in range(training_epochs):
    avg_cost = 0
    for train_features, train_labels in train_dataloader: # 미니 배치 단위로 꺼내온다. X는 미니 배치, Y느 ㄴ레이블.
        # image is already size of (28x28), no reshape
        # label is not one-hot encoded
        train_features = train_features.to(device)
        train_labels = train_labels.to(device)

        # Forward pass
        outputs = resnet50_pretrained(train_features)
        loss = criterion(outputs, train_labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_cost += loss / total_batch

    print('[Epoch: {:>4}] cost = {:>.9}'.format(e + 1, avg_cost))

for e in range(training_epochs):
    avg_cost = 0
    for train_features, train_labels in train_dataloader: # 미니 배치 단위로 꺼내온다. X는 미니 배치, Y느 ㄴ레이블.
        # image is already size of (28x28), no reshape
        # label is not one-hot encoded
        train_features = train_features.to(device)
        train_labels = train_labels.to(device)

        # Forward pass
        outputs = resnet101_pretrained(train_features)
        loss = criterion(outputs, train_labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_cost += loss / total_batch

    print('[Epoch: {:>4}] cost = {:>.9}'.format(e + 1, avg_cost))

resnet101_pretrained.eval()
correct = 0
total = 0
guess = []
correct_ans = []
with torch.no_grad():
    for train_features, train_labels in train_dataloader: # 미니 배치 단위로 꺼내온다. X는 미니 배치, Y느 ㄴ레이블.
        # image is already size of (28x28), no reshape
        # label is not one-hot encoded
        train_features = train_features.to(device)
        train_labels = train_labels.to(device)
        outputs = resnet101_pretrained(train_features)
        _, predicted = torch.max(outputs.data, 1)

        total += len(train_labels)
        correct += (predicted == train_labels).sum().item()
print(predicted, train_labels)
print('train Accuracy of the model on the {} train images: {} %'.format(total, 100 * correct / total))

resnet101_pretrained.eval()
correct = 0
total = 0
guess = []
correct_ans = []
with torch.no_grad():
    for test_features, test_labels in test_dataloader: # 미니 배치 단위로 꺼내온다. X는 미니 배치, Y느 ㄴ레이블.
        # image is already size of (28x28), no reshape
        # label is not one-hot encoded
        test_features = test_features.to(device)
        test_labels = test_labels.to(device)
        outputs = resnet101_pretrained(test_features)
        _, predicted = torch.max(outputs.data, 1)

        total += len(test_labels)
        correct += (predicted == test_labels).sum().item()
print(predicted, test_labels)
print('Test Accuracy of the model on the {} test images: {} %'.format(total, 100 * correct / total))

torch.save(resnet101_pretrained.state_dict(), '/content/drive/MyDrive/sk_personal/model_101.pt')

from torchvision import models

resnet101_pretrained = models.resnet101(pretrained=True)

print(resnet101_pretrained)

num_ftrs = resnet101_pretrained.fc.in_features
resnet101_pretrained.fc = torch.nn.Linear(num_ftrs, num_classes)

resnet101_pretrained.to(device)

resnet101_pretrained.load_state_dict(torch.load('/content/drive/MyDrive/sk_personal/model_101.pt'))

resnet101_pretrained.eval()
correct = 0
total = 0
guess = []
correct_ans = []
with torch.no_grad():
    for test_features, test_labels in test_dataloader: # 미니 배치 단위로 꺼내온다. X는 미니 배치, Y느 ㄴ레이블.
        # image is already size of (28x28), no reshape
        # label is not one-hot encoded
        test_features = test_features.to(device)
        test_labels = test_labels.to(device)
        outputs = resnet101_pretrained(test_features)
        _, predicted = torch.max(outputs.data, 1)

        total += len(test_labels)
        correct += (predicted == test_labels).sum().item()
print(predicted, test_labels)
print('Test Accuracy of the model on the {} test images: {} %'.format(total, 100 * correct / total))
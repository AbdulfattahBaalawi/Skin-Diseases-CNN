import numpy as np
# from pandas.libs import TimeStamp
import pandas as pd
# from pandas._libs.tslibs.timestamps import Timestamp as pd
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torchvision import transforms, datasets, models
from torch.utils.data.sampler import SubsetRandomSampler

import os

print(os.listdir("D:/base_dir/train_dir"))
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomVerticalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

validation_transforms = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                 [0.229, 0.224, 0.225])])
img_dir = "D:/base_dir/train_dir"
train_data = datasets.ImageFolder(img_dir, transform=train_transforms)
# number of subprocesses to use for data loading
num_workers = 0
# percentage of training set to use as validation
valid_size = 0.2

test_size = 0.1

# convert data to a normalized torch.FloatTensor
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
# obtain training indices that will be used for validation
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
valid_split = int(np.floor((valid_size) * num_train))
test_split = int(np.floor((valid_size + test_size) * num_train))
valid_idx, test_idx, train_idx = indices[:valid_split], indices[valid_split:test_split], indices[test_split:]

print(len(valid_idx), len(test_idx), len(train_idx))

# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)
test_sampler = SubsetRandomSampler(test_idx)

# prepare data loaders (combine dataset and sampler)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64,
                                           sampler=train_sampler, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size=32,
                                           sampler=valid_sampler, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(train_data, batch_size=8,
                                          sampler=test_sampler, num_workers=num_workers)
model = models.resnet50(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(2048, 8, bias=True)

fc_parameters = model.fc.parameters()

for param in fc_parameters:
    param.requires_grad = True

model
use_cuda = torch.cuda.is_available()
if use_cuda:
    model = model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)


# %%

def train(n_epochs, model, optimizer, criterion, use_cuda, save_path):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf

    for epoch in range(1, n_epochs + 1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()

            # initialize weights to zero
            optimizer.zero_grad()
            output = model(data)

            # calculate loss
            loss = criterion(output, target)

            # back prop
            loss.backward()

            # grad
            optimizer.step()

            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))

            if batch_idx % 100 == 0:
                print('Epoch %d, Batch %d loss: %.6f' %
                      (epoch, batch_idx + 1, train_loss))

        ######################
        # validate the model #
        ######################
        model.eval()
        for batch_idx, (data, target) in enumerate(valid_loader):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## update the average validation loss
            output = model(data)
            loss = criterion(output, target)
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))

        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch,
            train_loss,
            valid_loss
        ))

        ## TODO: save the model if validation loss has decreased
        if valid_loss < valid_loss_min:
            torch.save(model.state_dict(), save_path)
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
            valid_loss_min = valid_loss

    # return trained model
    return model


# %%

# %%
train(21, model, optimizer, criterion, use_cuda, 'SkinDiseases_detection1.pt')

# %%

model.load_state_dict(torch.load('SkinDiseases_detection.pt'))


# %%

def test(model, criterion, use_cuda):
    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    for batch_idx, (data, target) in enumerate(test_loader):
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)

    print('Test Loss: {:.6f}\n'.format(test_loss))
    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))


test(model, criterion, use_cuda)


# %%

def load_input_image(img_path):
    image = Image.open(img_path)
    prediction_transform = transforms.Compose([transforms.Resize(size=(224, 224)),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225])])

    # discard the transparent, alpha channel (that's the :3) and add the batch dimension
    image = prediction_transform(image)[:3, :, :].unsqueeze(0)
    return image


# %%

def predict_glaucoma(model, class_names, img_path):
    img = load_input_image(img_path)
    model = model.cpu()
    model.eval()
    idx = torch.argmax(model(img))
    return class_names[idx]


# %%

from glob import glob
from PIL import Image
from termcolor import colored

class_names = ['akiec', 'bcc','bkl','df','mel','nv','scc','vasc']
inf1 = np.array(glob("D:/base_dir/val_dir/akiec/*"))
inf2 = np.array(glob("D:/base_dir/val_dir/bcc/*"))
inf3 = np.array(glob("D:/base_dir/val_dir/bkl/*"))
inf4 = np.array(glob("D:/base_dir/val_dir/df/*"))
inf5 = np.array(glob("D:/base_dir/val_dir/mel/*"))
inf6 = np.array(glob("D:/base_dir/val_dir/nv/*"))
inf7 = np.array(glob("D:/base_dir/val_dir/scc/*"))
inf8 = np.array(glob("D:/base_dir/val_dir/vasc/*"))

for i in range(2):
    img_path = inf1[i]
    img = Image.open(img_path)
    if predict_glaucoma(model, class_names, img_path) == 'akiec':
        print(colored('akiec', 'green'))
    else:
        print(colored('NP', 'red'))
    plt.imshow(img)
    plt.show()
for i in range(2):
    img_path = inf2[i]
    img = Image.open(img_path)
    if predict_glaucoma(model, class_names, img_path) == 'bcc':
        print(colored('bcc', 'green'))
    else:
        print(colored('negative prediction', 'red'))
    plt.imshow(img)
    plt.show()

for i in range(2):
        img_path = inf3[i]
        img = Image.open(img_path)
        if predict_glaucoma(model, class_names, img_path) == 'bkl':
            print(colored('bkl', 'green'))
        else:
            print(colored('negetive prediction', 'red'))
        plt.imshow(img)
        plt.show()


for i in range(2):
    img_path = inf4[i]
    img = Image.open(img_path)
    if predict_glaucoma(model, class_names, img_path) == 'df':
        print(colored('df', 'green'))
    else:
        print(colored('N p', 'red'))
    plt.imshow(img)
    plt.show()
for i in range(2):
    img_path = inf5[i]
    img = Image.open(img_path)
    if predict_glaucoma(model, class_names, img_path) == 'mel':
        print(colored('mel', 'green'))
    else:
        print(colored('negative prediction', 'red'))
    plt.imshow(img)
    plt.show()

for i in range(2):

    img_path = inf6[i]
    img = Image.open(img_path)
    if predict_glaucoma(model, class_names, img_path) == 'nv':
        print(colored('nv', 'green'))
    else:
       print(colored('negative prediction', 'red'))
    plt.imshow(img)
    plt.show()

for i in range(2):
    img_path = inf7[i]
    img = Image.open(img_path)
    if predict_glaucoma(model, class_names, img_path) == 'scc':
        print(colored('scc', 'green'))
    else:
        print(colored('negative prediction', 'red'))
    plt.imshow(img)
    plt.show()

for i in range(2):
    img_path = inf8[i]
    img = Image.open(img_path)
    if predict_glaucoma(model, class_names, img_path) == 'vasc':
        print(colored('vasc', 'green'))
    else:
        print(colored('negative prediction', 'red'))
    plt.imshow(img)
    plt.show()

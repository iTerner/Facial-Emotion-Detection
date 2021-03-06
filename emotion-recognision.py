"""
final_notebook_update_accuracy.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1pe9KZCz4Bw-wsR40_TQMbRskMaU5wKSd

"""

# Import
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import os
import torchvision
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from random import sample
import numpy as np
import torch.nn as nn
import matplotlib.image as mpimg
# import imutils
# import dlib
import cv2
import pickle
import time
import seaborn as sns
from dataset import PlainDataset
from architecture import TerGat

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# if you dont have the folder 'eameo-faceswap-generator', run this command in the terminal
# the folder might be in a zip folder
# git clone https: // github.com/nicolasmetallo/eameo-faceswap-generator

# precentage of the train set
TRAIN = 0.8

# precentage of data to cut from the data set
CUT = 0.0

d = {
    "train": list(),
    "val": list(),
    "test": list(),
}

emotions = {
    "anger": 0,
    "disgust": 1,
    "fear": 2,
    "joy": 3,
    "neutral": 4,
    "sadness": 5,
    "surprise": 6
}

# path to the original image folder
imgs_path = "/content/drive/MyDrive/deep_project_data/tmp/FERG_DB_256"

# path of where we want to save the data
path = ""

# the pixels for the detection of the mouth, eyebrows and eyes
MOUTH = (48, 68)
LEFT_EYEBROW = (22, 27)
RIGHT_EYEBROW = (17, 22)
LEFT_EYE = (42, 48)
RIGHT_EYE = (36, 42)

# # create the detector and predictor objects
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor(
#     "/eameo-faceswap-generator/shape_predictor_68_face_landmarks.dat")


""" Loading the Data """

# load the data from the pickle file
final_d = {"train": None, "test": None, "val": None}
for key in final_d.keys():
    a_file = open(f"{path}final{key}.pkl", "rb")
    output = pickle.load(a_file)
    final_d[key] = output

# checking we don't mess up :)
count = 0
for key, val in final_d.items():
    s = set()
    l = len(val)
    count += l
    for img1, img2, label in val:
        s.add(label)
    print(f"in {key}, there is total of {l} pics and labels {s}")
print(f"total images {count}")

# print a sample images
plt.imshow(final_d["train"][0][0])

plt.imshow(final_d["train"][0][1])

""" Create DataLoaders"""

# define the batch size
batch_size = 64

# convert the images to PIL.Image object
final_data = {"train": list(), "test": list(), "val": list()}
for key, val in final_d.items():
    for ui, li, label in val:
        rui = Image.fromarray(ui)
        rli = Image.fromarray(li)
        final_data[key].append((rui, rli, label))

# checking we don't mess up :)
count = 0
for key, val in final_data.items():
    s = set()
    l = len(val)
    count += l
    for img1, img2, label in val:
        s.add(label)
    print(f"in {key}, there is total of {l} pics and labels {s}")
print(f"total images {count}")

# define the transformation for the upper and lower images
u_t = transforms.Compose([transforms.Resize(
    (32, 32)), transforms.ToTensor(), transforms.Normalize(0, 0.5)])
l_t = transforms.Compose([transforms.Resize(
    (32, 32)), transforms.ToTensor(), transforms.Normalize(0, 0.5)])

# create the dataset for the train, validation and test sets
train_data = PlainDataset(final_data["train"], u_t, l_t)
val_data = PlainDataset(final_data["val"], u_t, l_t)
test_data = PlainDataset(final_data["test"], u_t, l_t)
# create the data loaders for the train , validation and test sets
train_loader = DataLoader(
    train_data, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_data, batch_size=batch_size,
                        shuffle=True, num_workers=0)
test_loader = DataLoader(
    test_data, batch_size=batch_size, shuffle=True, num_workers=0)

# print example batch
for img1, img2, label in train_loader:
    print(img1.shape)
    print(img2.shape)
    break


""" Training Loop """

# define the model
model = TerGat(latent_dim=100).to(device)

# define the hyperparameters
epochs = 50
lr = 1e-3

# define the criterion and the optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# function to calcualte accuracy of the model


def calculate_accuracy(model, dataloader, device, criterion):
    """
    Helper function to calculate accuracy of the model
    Args:
        model - the selected model
        dataloader - the dataloader to calculate the accuracy on
        device - the device
        criterion - the loss function
    Return:
        the loss and accuracy of the model on the given data, and the confusion matrix
    """
    model.eval()  # put in evaluation mode
    total_correct = 0
    running_loss = 0
    total_images = 0
    confusion_matrix = np.zeros([10, 10], int)
    with torch.no_grad():
        for data in dataloader:
            ui, li, labels = data
            ui = ui.to(device)
            li = li.to(device)
            labels = labels.to(device)
            outputs = model(ui, li)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total_images += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            running_loss += loss.data.item()
            for i, l in enumerate(labels):
                confusion_matrix[l.item(), predicted[i].item()] += 1
    # print(f"total corrected {total_correct}, total images in data {total_images}, diff {total_images - total_correct}")
    model_accuracy = total_correct / total_images * 100
    running_loss /= len(dataloader)
    return running_loss, model_accuracy, confusion_matrix


# training loop
train_accs, test_accs = [], []
train_losses = []
val_losses = []
print('==> Start Training ...')
for epoch in range(1, epochs + 1):
    model.train()  # put in training mode
    running_loss = 0.0
    epoch_time = time.time()
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        ui, li, labels = data
        # send them to device
        ui = ui.to(device)
        li = li.to(device)
        labels = labels.to(device)

        # forward + backward + optimize
        outputs = model(ui, li)  # forward pass
        loss = criterion(outputs, labels)  # calculate the loss
        # always the same 3 steps
        optimizer.zero_grad()  # zero the parameter gradients
        loss.backward()  # backpropagation
        optimizer.step()  # update parameters

        # print statistics
        running_loss += loss.data.item()
        break

    # Normalizing the loss by the total number of train batches
    running_loss /= len(train_loader)
    train_losses.append(running_loss)

    # Calculate training/test set accuracy of the existing model
    _, train_accuracy, _ = calculate_accuracy(
        model, train_loader, device, criterion)
    val_loss, val_accuracy, _ = calculate_accuracy(
        model, val_loader, device, criterion)
    val_losses.append(val_loss)
    train_accs.append(train_accuracy)
    test_accs.append(val_accuracy)
    log = "Epoch: {} | Train Loss: {:.4f} | Validation Loss: {:.4f} |Training accuracy: {:.3f}% | Validation accuracy: {:.3f}% | ".format(
        epoch, running_loss, val_loss, train_accuracy, val_accuracy)
    epoch_time = time.time() - epoch_time
    log += "Epoch Time: {:.2f} secs".format(epoch_time)
    print(log)

print('==> Finished Training ...')


print("==> Plotting results ...")
res_path = "res/"
if not os.path.exists(res_path):
    os.mkdir(res_path)

# plot a comparsion of the train and validation accuracy
plt.clf()
plt.plot(list(range(epochs)), train_accs, label="Train Accuracy")
plt.plot(list(range(epochs)), test_accs, label="Validation Accuracy")
plt.title("Accuracy Train V.S. Validation")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid()
plt.legend()
plt.savefig(f"{res_path}/TerGat_accuracy_{epochs}_{lr}_{batch_size}.png")
plt.show()

# plot a comparsion of the train and validation loss
plt.plot(list(range(epochs)), train_losses, label="Train Loss")
plt.plot(list(range(epochs)), val_losses, label="Validation Loss")
plt.title("Loss Train V.S. Validation")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.legend()
plt.savefig(f"{res_path}/TerGat_loss_{epochs}_{lr}_{batch_size}.png")
plt.show()

# get the model test accuracy and confusion matrix
_, test_accuracy, test_confusion = calculate_accuracy(
    model, test_loader, device, criterion)

print(f"accuracy on Test set is {test_accuracy:.3f}%")

# plot the test confusion matrix
test_confusion = test_confusion[:7, :7]
emotions = ["happiness", "sadness", "anger",
            "fear", "disgust", "surprise", "neutral"]
df_d = {e: test_confusion[i, :] for i, e in enumerate(emotions)}
df = pd.DataFrame(df_d, index=emotions)

sns.heatmap(df, fmt="d", cmap="YlGnBu", annot=True)
plt.savefig(f"{res_path}/heatmap_{epochs}_{lr}_{batch_size}.png")
plt.show()

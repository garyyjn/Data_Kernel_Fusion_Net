import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
from c3d import C3D
import random
from custom_dataset import ucf_c3d
import torch.nn as nn
import graph_util
from data_util import *

batch_size = 8
segment_number = 5
segment_feature = 50
class_number = 101
test_split = 0.2
epochs = 100
validate_rate = 2
save_rate = 4
random.seed(4)
full_dataset = ucf_c3d('video_annotations_resnet','G:\\C3D Data')

test_size = int(len(full_dataset)*.2)
train_size = int(len(full_dataset)*.8)
waste_size = len(full_dataset) - test_size - train_size

print("train ins {}, test ins {}, waste ins {}".format(train_size, test_size, waste_size))
train_dataset, test_dataset, waste = torch.utils.data.random_split(full_dataset, [train_size, test_size, waste_size])

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory = True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory = True)

save_path = 'epoch8_accuracy0.36838152459631995.dict'
model = C3D(num_classes = 101, pretrained = True)


def save_model(model, save_dir):
    torch.save(model, save_dir)

def load_model(model, save_dir):
    model.load_state_dict(torch.load(save_dir))

if(save_path is not None):
    load_model(model, save_path)

if torch.cuda.is_available():
  dev = "cuda:0"
else:
  dev = "cpu"
model.cuda()


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=.0007, momentum=0.7)

auxdict = getauxdict('UCF-101')

global plotter
plotter = graph_util.VisdomLinePlotter(env_name='main')

epoch_train_loss = []
epoch_val_loss = []
epoch_val_accu = []
train_y = []
val_y = []


def train_batch(model, optimizer, videos, labels):
    '''
    :param videos:n*segments, channels, frames, height, width
    :param labels:n*101 1/0
    '''
    optimizer.zero_grad()
    y = model(videos.cuda())
    loss = criterion(y, labels)
    loss.backward()
    optimizer.step()
    return loss

def validate(model, validate_set):
    with torch.no_grad():
        n_correct = 0
        total = 0
        batch_loss_hist = []
        for i_batch, sample_batched in tqdm(enumerate(validate_set)):
            video_matrix = sample_batched['original']
            action_list = sample_batched['action']
            labels = torch.tensor(action2label(list(action_list), auxdict), dtype=torch.long).cuda()
            y = model(video_matrix.cuda())
            loss = criterion(y, labels)
            batch_loss_hist.append(loss)
            n_correct += (torch.max(y, 1)[1].view(labels.size()) == labels).sum().item()
            total += video_matrix.shape[0]
        validate_loss = sum(batch_loss_hist)/len(batch_loss_hist)
        accuracy = n_correct/total
        return validate_loss,accuracy

for curr_epoch in range(epochs):
    batch_loss_hist = []
    if curr_epoch % validate_rate == 0:
        val_loss, accuracy = validate(model, test_dataloader)
        plotter.plot('loss', 'validate', 'val loss', curr_epoch, val_loss.item())
        plotter.plot('accuracy', 'validate', 'val accuracy', curr_epoch, accuracy)
        torch.save(model.state_dict(), 'epoch{}_accuracy{}.dict'.format(curr_epoch, accuracy))
    for i_batch, sample_batched in tqdm(enumerate(train_dataloader)):
        video_matrix = sample_batched['original']
        action_list = sample_batched['action']
        labels = torch.tensor(action2label(list(action_list), auxdict), dtype=torch.long).cuda()
        batch_loss = train_batch(model, optimizer, video_matrix, labels)
        batch_loss_hist.append(batch_loss)



    train_loss = sum(batch_loss_hist)/len(batch_loss_hist)
    plotter.plot('loss', 'train', 'train loss', curr_epoch, train_loss.item())


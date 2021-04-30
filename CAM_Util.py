import torch
from numpy import nper
from medcam import medcam
from c3d import C3D
from custom_dataset import ucf_c3d
from torch.utils.data import Dataset, DataLoader
from vist_util import *
model = C3D(num_classes=101,pretrained=True)
loadfrom = 'c3d46perc.dict'
model.load_state_dict(torch.load(loadfrom))
print(model)
full_dataset = ucf_c3d('video_annotations_resnet','G:\\C3D Data', kernels=['original'])

dataloader = DataLoader(full_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory = True)

model = medcam.inject(model, output_dir="attention_maps", layer = 'conv5b', backend='gcam', return_attention = True, save_maps=True)

model.eval()
for batch in dataloader:
    video_matrix = batch['original']
    action_list = batch['action']
    score, attention = model(video_matrix)
    #print(score.shape)
    attention = torch.squeeze(attention).numpy()
    video_matrix = torch.squeeze(video_matrix).numpy().astype(np.int)#3,16,112,112
    #print(video_matrix)
    video_matrix = np.transpose(video_matrix, (1,2,3,0))
    #print(attention.shape)
    #print(video_matrix.shape)
    showbunch(attention)
    showbunch(video_matrix)
    break
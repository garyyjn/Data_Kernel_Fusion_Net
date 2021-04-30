from c3d import C3D
import torch.nn as nn
import torch

class Fusion_Model(nn.Module):
    def __init__(self, save_path_1, save_path_2):
        super(Fusion_Model, self).__init__()

        self.conv1_k1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1_k1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv1_k2 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1_k2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2_k1 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2_k1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv2_k2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2_k2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(6, 3, 3), stride=(2, 1, 1), padding=(2, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 101)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()




    def forward(self, x_1, x_2):
        x_k1 = self.relu(self.conv1_k1(x_1))
        x_k1 = self.pool1_k1(x_k1)

        x_k1 = self.relu(self.conv2_k1(x_k1))
        x_k1 = self.pool2_k1(x_k1)

        x_k2 = self.relu(self.conv1_k2(x_2))
        x_k2 = self.pool1_k2(x_k2)

        x_k2 = self.relu(self.conv2_k2(x_k2))
        x_k2 = self.pool2_k2(x_k2)

        batch_size = x_k1.shape[0]
        x = torch.ones(batch_size,128,16,28,28).cuda()
        x[:,:,[0,2,4,6,8,10,12,14],:,:] = x_k1
        x[:,:,[1,3,5,7,9,11,13,15],:,:] = x_k2

        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)

        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        x = self.pool4(x)

        x = self.relu(self.conv5a(x))
        x = self.relu(self.conv5b(x))
        x = self.pool5(x)

        x = x.view(-1, 8192)
        x = self.relu(self.fc6(x))
        x = self.dropout(x)
        x = self.relu(self.fc7(x))
        x = self.dropout(x)

        logits = self.fc8(x)
        out = self.softmax(logits)
        return out
'''
input_k1 = torch.rand(1, 3, 16, 112, 112)
input_k2 = torch.rand(1, 3, 16, 112, 112)
model = Fusion_Model(None,None)
y = model.forward(input_k1,input_k2)
print(y.shape)
'''
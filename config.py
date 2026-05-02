import torch

# Dataset config
resize_x = 224
resize_y = 224
input_channels = 3
batch_size = 64
number_of_epochs = 20
val_split = 0.20
num_workers = 4
num_classes = 29

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Training config
cnn_lr = 1e-3
cnn_wd = 1e-4

mnv2_head_epochs = 10
mnv2_head_lr = 1e-3
mnv2_ft_epochs = 10
mnv2_ft_lr = 1e-4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

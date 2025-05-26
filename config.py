import torch

model_path = None
lr = 1e-5
device = 'cuda'
dtype = torch.bfloat16
data_path = '../data/lke_2017'
save_path = './'
epochs = 4
batch_size = 8
num_workers = 32
seed = 107

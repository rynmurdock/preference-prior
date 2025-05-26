import torch

# NOTE model path name changed
model_path = None # './last_epoch_ckpt/'
lr = 1e-6
device = 'cuda'
dtype = torch.bfloat16
data_path = '../data/lke_2017'
save_path = './'
epochs = 4
batch_size = 32
number_k_clip_embed = 32 # divide by this to determine bundling together of sequences -> CLIP
num_workers = 32
seed = 107

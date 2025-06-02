import torch

# NOTE model path name changed
model_path = None # './last_epoch_ckpt/'
lr = 1e-5
device = 'cuda'
dtype = torch.bfloat16
data_path = '../data/lke_2017'
save_path = './'
epochs = 3
batch_size = 16
number_k_clip_embed = 16 # divide by this to determine bundling together of sequences -> CLIP
num_workers = 32
seed = 107
k = 8

# TODO config option to swap to diffusion?

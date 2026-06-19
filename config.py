import torch

model_path = None
lr = 1e-5
device = 'cuda'
dtype = torch.bfloat16
data_path = '../PAMELA/annotations/pamela_train.json'
val_data_path = '../PAMELA/annotations/pamela_val_unseen.json'
save_path = './'
# we sample of the participants vectors so dataset is larger than shown
epochs = 3000

batch_size = 16
number_k_clip_embed = 16 # divide by this to determine bundling together of sequences -> CLIP

num_workers = 32
seed = 107
k = 8

# TODO config option to swap to diffusion?

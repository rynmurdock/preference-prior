import torch

k = 16
cosine_loss_weight = .1

model_path = None # "last_epoch_ckpt"
lr = 1e-5
epochs = 3000

device = 'cuda'
dtype = torch.bfloat16
data_path = '../PAMELA/annotations/pamela_train.json'
val_data_path = '../PAMELA/annotations/pamela_val_unseen.json'
save_path = './'
# we sample of the participants vectors so dataset is larger than shown

batch_size = 4
number_k_clip_embed = 4 # divide by this to determine bundling together of sequences -> CLIP
num_workers = 8
seed = 107

# TODO config option to swap to diffusion?

from dataclasses import dataclass, field
import torch

@dataclass
class Config:
    # Model
    # model_path = None # str = 'kandinsky-community/kandinsky-2-2-prior'
    model_path = './last_epoch_ckpt' # str = 'kandinsky-community/kandinsky-2-2-prior'
    do_diffusion: bool = True

    # Hparams
    k: int = 8
    batch_size: int = 1
    lr: float = 1e-5
    cosine_loss_weight: float = 4

    # Training
    epochs: int = 3000000000000
    max_steps: int = 400000
    do_compile: bool = True
    device: str = 'cuda'
    dtype: torch.dtype = field(default=torch.bfloat16, repr=False)
    seed: int = 107

    # Data
    data_path: str = '../PAMELA/annotations/pamela_train.json'
    val_data_path: str = '../PAMELA/annotations/pamela_val_unseen.json'
    number_k_clip_embed: int = 1  # divide by this to determine bundling together of sequences -> CLIP
    num_workers: int = 8

    # Logging
    save_path: str = './'
    freq: int = 100  # how often we save/log/etc.


config = Config()
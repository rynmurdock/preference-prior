

###########################################
'''
python -m train
'''
###########################################


import os
import sys
import torch
torch.set_float32_matmul_precision('high')
import logging
import numpy as np
from tqdm import tqdm
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data import get_dataloader
from model import get_model_and_tokenizer, get_optimizer_and_lr_sched, get_loss
from config import config
from utils import log_cuda_mem

logging.basicConfig(level=logging.INFO)

def main():
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    model, vision_tokenizer, text_tokenizer = get_model_and_tokenizer(config.model_path, config.device, config.dtype)
    optimizer, lr_sched = get_optimizer_and_lr_sched(list(model.prior.parameters()), 
                                                     config.lr)
    dataloader, val_dataloader = get_dataloader(config.data_path, config.val_data_path,
                                                 config.batch_size, config.num_workers, 
                                model.prior_pipe.image_processor, k=config.k)
    
    train_losses = []
    inner_train_losses = []
    validation_losses = []
    total_inds = 0

    for epoch in range(config.epochs):
        for ind, batch in tqdm(enumerate(iter(dataloader))):
            if total_inds > config.max_steps:
                model.prior.save_pretrained(f'{config.save_path}/last_epoch_ckpt', from_pt=True)
                sys.exit()
            if batch is None:
                continue

            input, input_scores, target, target_scores, sample_prompts, input_prompts = batch
            input = input.to(config.device)
            target = target.to(config.device)

            if total_inds % config.freq == 0:
                # NOTE autocasting because our fp32 training model is also our val model; only want calculations in half.
                with torch.autocast(enabled=True, device_type='cuda', dtype=config.dtype):
#                     # TODO make this not brittle
#                     examples = [
#                         # '../generative_recommender/Blue_Tigers_space/1o.png',
#  '../../generative_recommender/Blue_Tigers_space/2o.png',
# #  '../generative_recommender/Blue_Tigers_space/3o.png',
# #  '../generative_recommender/Blue_Tigers_space/4o.png',
# #  '../generative_recommender/Blue_Tigers_space/5o.png',
#  '../../generative_recommender/Blue_Tigers_space/10o.png',
#  '../../generative_recommender/Blue_Tigers_space/7o.png',
#  '../../generative_recommender/Blue_Tigers_space/9o.png',
#  ]
#                     assert all([os.path.exists(a) for a in examples]), f'{all([os.path.exists(a) for a in examples])=}'
#                     model.do_qual_val([[Image.open(j) for j in examples]], k=config.k)
                    val_loss = model.do_quant_val(val_dataloader)
                    logging.info(f'{val_loss=:.4f}')
                    if total_inds // config.freq != 0:
                        validation_losses.append(val_loss)
                    if len(inner_train_losses) > 0:
                        if total_inds // config.freq != 0:
                            train_losses.append(sum(inner_train_losses)/len(inner_train_losses))
                        inner_train_losses = []

                    train_losses = train_losses
                    plt.plot(train_losses)
                    plt.plot(validation_losses)
                    plt.savefig('latest_loss_curves.png')
                    plt.clf()

            optimizer.zero_grad()
            loss, loss_logging_dict = get_loss(model=model, 
                                                input=input, 
                                                target=target, 
                                                image_encoder=vision_tokenizer,
                                                text_encoder=text_tokenizer,
                                                scores=input_scores, 
                                                target_scores=target_scores,
                                                sample_prompts=sample_prompts,
                                                input_prompts=input_prompts
                                                )
            if total_inds % config.freq == 0:
                mse_loss, cosine_loss = loss_logging_dict.get('mse_loss'), loss_logging_dict.get('cosine_loss')
                logging.info(
                    f'Train MSE: {mse_loss}, '
                    f'Cosine: {cosine_loss},' 
                    f'Weighted Total: {loss.item()}'
                )
            inner_train_losses.append(loss.item())
            loss.backward()
            optimizer.step()
            lr_sched.step()

            total_inds += 1
            if total_inds % config.freq == 0:
                # TODO add loading from path
                model.prior.save_pretrained(f'{config.save_path}/last_epoch_ckpt', from_pt=True)

if __name__ == '__main__':
    main()

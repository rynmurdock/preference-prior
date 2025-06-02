

########################################
# python -m train
###########################################


import torch
import logging
import numpy as np
from tqdm import tqdm
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data import get_dataloader
from model import get_model_and_tokenizer, get_optimizer, get_loss
import config



logging.basicConfig(level=logging.INFO)

def main():
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    model, tokenizer = get_model_and_tokenizer(config.model_path, config.device, config.dtype)
    optimizer = get_optimizer(list(model.prior.parameters()), config.lr)
    dataloader, val_dataloader = get_dataloader(config.data_path, config.batch_size, config.num_workers, 
                                model.prior_pipe.image_processor, k=config.k)
    
    train_losses = []
    inner_train_losses = []
    validation_losses = []

    for epoch in range(config.epochs):
        for ind, batch in tqdm(enumerate(iter(dataloader))):
            if batch is None:
                continue

            input, target = batch
            input = input.to(config.device)
            target = target.to(config.device)

            if ind % 50 == 0:
                with torch.cuda.amp.autocast(enabled=True, dtype=config.dtype): # NOTE using autocast because our training model is also our val model, so don't want to set to full half precision.
                    examples = [
                        # '../generative_recommender/Blue_Tigers_space/1o.png',
 '../generative_recommender/Blue_Tigers_space/2o.png',
#  '../generative_recommender/Blue_Tigers_space/3o.png',
#  '../generative_recommender/Blue_Tigers_space/4o.png',
#  '../generative_recommender/Blue_Tigers_space/5o.png',
 '../generative_recommender/Blue_Tigers_space/10o.png',
 '../generative_recommender/Blue_Tigers_space/7o.png',
 '../generative_recommender/Blue_Tigers_space/9o.png',
 ]
                    model.do_qual_val([[Image.open('../'+j) for j in examples]], k=config.k)
                    val_loss = model.do_quant_val(val_dataloader)
                    validation_losses.append(val_loss)
                    if len(inner_train_losses) > 0:
                        train_losses.append(sum(inner_train_losses)/len(inner_train_losses))
                        inner_train_losses = []


                    if len(train_losses) > 0:
                        plt.plot(train_losses)
                    plt.plot(validation_losses)
                    plt.savefig('latest_loss_curves.png')
                    plt.clf()

            loss = get_loss(model, input, target, tokenizer)
            inner_train_losses.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if ind % 100 == 0:
                # TODO add loading from path
                model.prior.save_pretrained(f'{config.save_path}/last_epoch_ckpt', from_pt=True) 

if __name__ == '__main__':
    main()

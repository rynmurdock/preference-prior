

########################################
# python -m train
###########################################


import torch
import logging
import numpy as np
from tqdm import tqdm
from PIL import Image
    
from data import get_dataloader
from model import get_model_and_tokenizer, get_optimizer
import config

logging.basicConfig(level=logging.INFO)

def get_loss(model, input, target, tokenizer):
    with torch.no_grad():
        assert len(input.shape) == 5 # [batch, s, c, w, h]
        cuts = config.number_k_clip_embed
        assert input.shape[0] * input.shape[1] % cuts == 0, 'batch size * `k` preferred embeds must be divisible by cuts'
        input = input.view(cuts//8, -1, 3, target.shape[-2], target.shape[-1])
        full_seq = []
        for b in input:
            input = tokenizer(b)['image_embeds'] # in our case, tokenizer is a clip embedding model
            full_seq.append(input)
        input = torch.stack(full_seq)

        target = tokenizer(target)['image_embeds']

        input = input.view(target.shape[0], -1, target.shape[-1])
        assert len(input.shape) == 3 # [batch, sequence, inner]
    
    with torch.cuda.amp.autocast(enabled=False, ):
        input = input.to(torch.float32)
        latent = torch.randn(input.shape[0], input.shape[-1], device=input.device)
        output = model(latent, input).predicted_image_embedding

    target = target.to(torch.float32)
    mse_loss = torch.nn.functional.mse_loss(target, output).mean()
    cosine_loss = torch.nn.functional.cosine_similarity(output, target).mean()
    loss =  mse_loss + .2 * cosine_loss

    logging.info(f'MSE: {mse_loss.item()}, Cosine: {cosine_loss.item()}, Weighted Total: {loss.item()}')
    # TODO wandb

    return loss

def main():
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    model, tokenizer = get_model_and_tokenizer(config.model_path, config.device, config.dtype)
    optimizer = get_optimizer(list(model.prior.parameters()), config.lr)
    dataloader = get_dataloader(config.data_path, config.batch_size, config.num_workers, 
                                model.prior_pipe.image_processor)

    for epoch in range(config.epochs):
        for ind, batch in tqdm(enumerate(iter(dataloader))):
            if batch is None:
                continue

            sample, target = batch
            sample = sample.to(config.device)
            target = target.to(config.device)

            if ind % 50 == 0:
                with torch.cuda.amp.autocast(enabled=True, dtype=config.dtype): # NOTE using autocast because our training model is also our val model, so don't want to set to full half precision.
                    examples = ['../generative_recommender/Blue_Tigers_space/1o.png',
 '../generative_recommender/Blue_Tigers_space/2o.png',
 '../generative_recommender/Blue_Tigers_space/3o.png',
 '../generative_recommender/Blue_Tigers_space/4o.png',
 '../generative_recommender/Blue_Tigers_space/5o.png',
 '../generative_recommender/Blue_Tigers_space/6o.png',
 '../generative_recommender/Blue_Tigers_space/7o.png',
 '../generative_recommender/Blue_Tigers_space/8o.png',]
                    model.do_validation([[Image.open('../'+j) for j in examples]])

            loss = get_loss(model, sample, target, tokenizer)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if ind % 100 == 0:
                # TODO add loading from path
                model.prior.save_pretrained(f'{config.save_path}/last_epoch_ckpt', from_pt=True) 

if __name__ == '__main__':
    main()

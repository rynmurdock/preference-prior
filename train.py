

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
        full_seq = []
        for b in input: # TODO may be nice slicing but can run in a batch then view ofc
            input = tokenizer(b)['image_embeds'] # in our case, tokenizer is a clip embedding model
            full_seq.append(input)
        input = torch.stack(full_seq)
        assert len(input.shape) == 3 # [batch, sequence, inner]
        target = tokenizer(target)['image_embeds']
    
    with torch.cuda.amp.autocast(enabled=True, dtype=config.dtype):
        latent = torch.randn(input.shape[0], input.shape[-1], device=input.device)
        output = model(latent, input).predicted_image_embedding

    mse_loss = torch.nn.functional.mse_loss(target, output)
    cosine_loss = torch.nn.functional.cosine_similarity(output, target)
    loss =  mse_loss + .2 * cosine_loss

    logging.info(f'MSE: {mse_loss.item()}, Cosine: {cosine_loss.item()}, Weighted Total: {loss.item()}')
    # TODO wandb


    return loss.mean()

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

            if ind % 100 == 0:
                with torch.cuda.amp.autocast(enabled=True, dtype=config.dtype):
                    examples = ['../generative_recommender/Blue_Tigers_space/1o.png',
 '../generative_recommender/Blue_Tigers_space/2o.png',
 '../generative_recommender/Blue_Tigers_space/3o.png',
 '../generative_recommender/Blue_Tigers_space/4o.png',
 '../generative_recommender/Blue_Tigers_space/5o.png',
 '../generative_recommender/Blue_Tigers_space/6o.png',
 '../generative_recommender/Blue_Tigers_space/7o.png',
 '../generative_recommender/Blue_Tigers_space/8o.png',
 '../generative_recommender/Blue_Tigers_space/9o.png',
 '../generative_recommender/Blue_Tigers_space/10o.png']
                    model.do_validation([[Image.open('../'+j) for j in examples]])

            loss = get_loss(model, sample, target, tokenizer)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if ind % 1000 == 0:
                # TODO add loading from path
                model.prior.save_pretrained(f'{config.save_path}/{epoch}_epoch_ckpt.pt', from_pt=True) 

if __name__ == '__main__':
    main()

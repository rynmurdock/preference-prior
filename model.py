
import torch
import logging
from diffusers import DiffusionPipeline

from prior.pipeline_kandinsky_prior import KandinskyPriorPipeline
from prior.prior_transformer import PriorTransformer


class Zoo(torch.nn.Module):
    def __init__(self, prior, prior_pipe, kandinsky_pipe, ) -> None:
        super().__init__()
        self.prior = prior
        self.prior_pipe = prior_pipe
        self.kandinsky_pipe = kandinsky_pipe
        self.pre_prior_transformer = None 
        # NOTE we may get better perf from freezing our prior 
        #     and only training a transformer adapter?

    def forward(self, preferred_embeds, latents):
        pred = self.prior(latents, preferred_embeds)
        return pred
    
    def do_validation(self, images): # TODO constant val seed
        assert all([len(i) == 8 for i in images]), f'We have must have `k` images, not {len(images)}.'
        image_embeds, negative_image_embeds = self.prior_pipe(images).to_tuple()
        images = self.kandinsky_pipe(
            num_inference_steps=50,
            image_embeds=image_embeds,
            negative_image_embeds=negative_image_embeds,
        ).images
        images[0].save('latest_val.png')
        return images

def get_model_and_tokenizer(path, device, dtype):
    prior = PriorTransformer.from_pretrained("ECLIPSE-Community/ECLIPSE_KandinskyV22_Prior" 
                                             if path is None else path).to(device)
        
    pipe_prior = KandinskyPriorPipeline.from_pretrained("kandinsky-community/kandinsky-2-2-prior", prior=prior).to(device)
    pipe_prior.image_encoder = pipe_prior.image_encoder.to(device, dtype)
    # Note: don't set the prior to `dtype`` as it may be half precision, 
    #     and we're training with mixed precision
    #     so we need to keep our full-precision weight for trained params
    kandinsky_pipe = DiffusionPipeline.from_pretrained("kandinsky-community/kandinsky-2-2-decoder").to(device, dtype)
    model = Zoo(prior, pipe_prior, kandinsky_pipe).to(device)

    return model, model.prior_pipe.image_encoder

def get_optimizer(params, lr):
    logging.info(f'Training: {params}')
    optimizer = torch.optim.AdamW(params, lr=lr)
    return optimizer

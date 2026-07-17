
import torch
import logging
from diffusers import DiffusionPipeline
from config import config
from prior.pipeline_kandinsky_prior import KandinskyPriorPipeline
from prior.prior_transformer import PriorTransformer

def get_loss(model, input, target, image_encoder, text_encoder, scores, target_scores, **kwargs):
    with torch.no_grad():
        assert len(input.shape) == 5 # [batch, s, c, w, h]
        cuts = config.number_k_clip_embed
        assert cuts < config.k * config.batch_size
        assert input.shape[0] * input.shape[1] % cuts == 0, 'batch size * `k` preferred embeds must be divisible by cuts'
        input = input.view(cuts, -1, 3, target.shape[-2], target.shape[-1])

        full_seq = []
        for b in input:
            # in our case, tokenizer is a clip embedding model
            input = image_encoder(b)['image_embeds']
            full_seq.append(input)
        input = torch.stack(full_seq)
        input = input.view(-1, config.k, input.shape[-1])
        # rng drop out embeddings
        drop_mask = torch.rand((input.shape[0], input.shape[1])) < .3

        # TODO we ought to attention mask & pad to largest
        input[drop_mask] = 0
        scores[drop_mask] = 0

        target = image_encoder(target)['image_embeds']
        assert len(input.shape) == 3 # [batch, sequence, inner]


        latent = torch.randn(input.shape[0], input.shape[-1], device=input.device)
        if config.do_diffusion:
            ts = torch.randint(0, 1000, (latent.shape[0],)).to(input.device)
            # drop scores at some probability
            drop_mask = torch.rand(latent.shape[0]) < .2
            
            scores[drop_mask] = 0
            target_scores[drop_mask] = 0
            latent = model.prior_pipe.scheduler.add_noise(target, 
                                                          noise=latent, timesteps=ts)
    
    with torch.autocast(device_type='cuda', enabled=True, dtype=config.dtype):
        output = model(latent, input, 
                       scores=scores, 
                       target_scores=target_scores,
                       timesteps=ts if config.do_diffusion else None,
                       ).predicted_image_embedding

    output = output.to(torch.float32)
    target = target.to(torch.float32)
    mse_loss = torch.nn.functional.mse_loss(target, output).mean()
    
    assert len(target.shape) == 2 and len(output.shape) == 2
    if config.do_diffusion:
        cosine_loss = torch.zeros_like(mse_loss)        
    else:
        cosine_loss = 1 - torch.nn.functional.cosine_similarity(output, target).mean()
    
    # # # take advantage of negatives but deal with cases where plausibly they aren't negatives
    # output = torch.nn.functional.normalize(output.view(-1, output.shape[-1]))
    # target = torch.nn.functional.normalize(target.view(-1, output.shape[-1]))
    # logits = output @ target.T
    # labels = torch.arange(len(target)).to(logits.device)
    # # each group is a positive sample to itself and a negative to all others
    # cls_loss = torch.nn.functional.cross_entropy(logits, labels, reduction='mean')

    loss =  mse_loss + config.cosine_loss_weight * cosine_loss

    logging_dict = {'mse_loss': mse_loss.item(), 
                    'cosine_loss': cosine_loss.item()}
    return loss, logging_dict


class Zoo(torch.nn.Module):
    def __init__(self, prior, prior_pipe, kandinsky_pipe=None, ) -> None:
        super().__init__()
        self.prior = prior
        self.prior_pipe = prior_pipe
        self.kandinsky_pipe = kandinsky_pipe
        self.pre_prior_transformer = None 
        # NOTE we may get better perf from freezing our prior 
        #     and only training a transformer adapter?

    def forward(self, latent, input, **kwargs):
        latent = latent.to('cuda')
        input = input.to('cuda')
        pred = self.prior(latent, input, 
                          scores=kwargs['scores'], 
                          target_scores=kwargs.get('target_scores'),
                          timesteps=kwargs.get('timesteps'),
                          )
        return pred
    
    @torch.no_grad()
    def do_qual_val(self, images, k, scores=None, path=None, 
                    prior_guidance_scale=3,
                    decoder_guidance_scale=3,
                    negative_by_options=False,
                    ):
        generator = torch.Generator(device="cpu").manual_seed(787)
        # NOTE if you use diffusion at some point, could set seed.
        # TODO config.k should really absorb into model class's self.k
        image_embeds, negative_image_embeds = self.prior_pipe(prompt=images, 
                                                              scores=scores,
                                                              k=k,
                                                              guidance_scale=prior_guidance_scale,
                                                              negative_by_options=negative_by_options,
                                                              ).to_tuple()
        images = self.kandinsky_pipe(
            num_inference_steps=50,
            image_embeds=image_embeds,
            negative_image_embeds=negative_image_embeds,
            guidance_scale=decoder_guidance_scale,
            generator=generator
        ).images
        images[0].save('latest_val.png' if path is None else path)
        return images
    
    @torch.no_grad()
    def do_quant_val(self, val_dataloader):
        losses = []
        for batch in val_dataloader:
            if batch is None:
                continue

            input, input_scores, target, target_scores = batch
            input = input.to(config.device)
            target = target.to(config.device)
            loss, loss_logging_dict = get_loss(self, 
                                               input, target, self.prior_pipe.image_encoder, 
                                               scores=input_scores, target_scores=target_scores)
            losses.append(loss.item())
        return sum(losses) / len(losses)
    
def get_model_and_tokenizer(path, device, dtype, compile=None):
    if path:
        prior = PriorTransformer.from_pretrained(path, subfolder='prior' if 'kandinsky-community' in path else None,
                                                 do_diffusion=config.do_diffusion)
    else:
        prior = PriorTransformer(do_diffusion=config.do_diffusion)
    prior = prior.to(device)

    if compile is not None:
        config.do_compile = compile
    if config.do_compile:
        prior = torch.compile(prior)
    
    pipe_prior = KandinskyPriorPipeline.from_pretrained("kandinsky-community/kandinsky-2-2-prior", 
                                                        prior=prior, do_diffusion=config.do_diffusion).to(device)
    pipe_prior.image_encoder = pipe_prior.image_encoder.to(device, dtype)
    pipe_prior.text_encoder = pipe_prior.text_encoder.to(device, dtype)
    # Note: don't set the prior to `dtype`` as it may be half precision, 
    #     and we're training with mixed precision
    #     so we need to keep our full-precision weight for trained params
    kandinsky_pipe = DiffusionPipeline.from_pretrained("kandinsky-community/kandinsky-2-2-decoder").to(device, dtype)
    model = Zoo(prior, pipe_prior, kandinsky_pipe).to(device)
    model.k = config.k

    return model, model.prior_pipe.image_encoder, model.prior_pipe.text_encoder

def get_optimizer_and_lr_sched(params, lr):
    logging.info(f'Training: {params}')
    optimizer = torch.optim.AdamW(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
    return optimizer, scheduler

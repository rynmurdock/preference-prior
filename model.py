
import torch
import logging
from diffusers import DiffusionPipeline
import config

from prior.pipeline_kandinsky_prior import KandinskyPriorPipeline
from prior.prior_transformer import PriorTransformer

def get_loss(model, input, target, tokenizer, **kwargs):
    with torch.no_grad():
        assert len(input.shape) == 5 # [batch, s, c, w, h]
        cuts = config.number_k_clip_embed
        assert cuts < config.k * config.batch_size
        assert input.shape[0] * input.shape[1] % cuts == 0, 'batch size * `k` preferred embeds must be divisible by cuts'
        input = input.view(cuts, -1, 3, target.shape[-2], target.shape[-1])

        full_seq = []
        for b in input:
            # in our case, tokenizer is a clip embedding model
            input = tokenizer(b)['image_embeds']
            full_seq.append(input)
        input = torch.stack(full_seq)
        input = input.view(-1, config.k, input.shape[-1])
        # rng drop out embeddings
        drop_mask = torch.rand((input.shape[0], input.shape[1])) < .3

        # TODO we ought to attention mask & pad to largest
        input[drop_mask] = 0
        kwargs['scores'][drop_mask] = 0


        target = tokenizer(target)['image_embeds']
        assert len(input.shape) == 3 # [batch, sequence, inner]
    
    with torch.autocast(device_type='cuda', enabled=True, dtype=config.dtype):
        latent = torch.randn(input.shape[0], input.shape[-1], device=input.device)
        # TODO use latent's rope to specify whether emb is high/low scoring;
        #    i.e., don't only predict preferred embs
        output = model(latent, input, scores=kwargs['scores'], target_scores=kwargs['target_scores']).predicted_image_embedding

    output = output.to(torch.float32)
    target = target.to(torch.float32)
    mse_loss = torch.nn.functional.mse_loss(target, output).mean()
    
    assert len(target.shape) == 2 and len(output.shape) == 2
    cosine_loss = 1 - torch.nn.functional.cosine_similarity(output, target).mean()
    
    # # # take advantage of negatives but deal with cases where plausibly they aren't negatives
    # output = torch.nn.functional.normalize(output.view(-1, output.shape[-1]))
    # target = torch.nn.functional.normalize(target.view(-1, output.shape[-1]))
    # logits = output @ target.T
    # labels = torch.arange(len(target)).to(logits.device)
    # # each group is a positive sample to itself and a negative to all others
    # cls_loss = torch.nn.functional.cross_entropy(logits, labels, reduction='mean')
    loss =  mse_loss + config.cosine_loss_weight * cosine_loss

    logging.info(f'MSE: {mse_loss.item()}, Cosine: {cosine_loss.item()}, Weighted Total: {loss.item()}')
    return loss


class Zoo(torch.nn.Module):
    def __init__(self, prior, prior_pipe, kandinsky_pipe, ) -> None:
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
        pred = self.prior(latent, input, scores=kwargs['scores'], target_scores=kwargs.get('target_scores'))
        return pred
    
    @torch.no_grad()
    def do_qual_val(self, images, k):
        generator = torch.Generator(device="cpu").manual_seed(787)
        # NOTE if you use diffusion at some point, could set seed.
        # TODO must setup giving scores now that we have them.
        image_embeds, negative_image_embeds = self.prior_pipe(images, k=k).to_tuple()
        images = self.kandinsky_pipe(
            num_inference_steps=50,
            image_embeds=image_embeds,
            negative_image_embeds=negative_image_embeds,
            generator=generator
        ).images
        images[0].save('latest_val.png')
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
            loss = get_loss(self, input, target, self.prior_pipe.image_encoder, scores=input_scores, target_scores=target_scores)
            losses.append(loss.item())
        return sum(losses) / len(losses)
    
def get_model_and_tokenizer(path, device, dtype):
    if path:
        prior = PriorTransformer.from_pretrained(path)
    else:
        prior = PriorTransformer()
    prior = prior.to(device)
        
    pipe_prior = KandinskyPriorPipeline.from_pretrained("kandinsky-community/kandinsky-2-2-prior", prior=prior).to(device)
    pipe_prior.image_encoder = pipe_prior.image_encoder.to(device, dtype)
    # Note: don't set the prior to `dtype`` as it may be half precision, 
    #     and we're training with mixed precision
    #     so we need to keep our full-precision weight for trained params
    kandinsky_pipe = DiffusionPipeline.from_pretrained("kandinsky-community/kandinsky-2-2-decoder").to(device, dtype)
    model = Zoo(prior, pipe_prior, kandinsky_pipe).to(device)
    model.k = config.k

    return model, model.prior_pipe.image_encoder

def get_optimizer_and_lr_sched(params, lr):
    logging.info(f'Training: {params}')
    optimizer = torch.optim.AdamW(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
    return optimizer, scheduler

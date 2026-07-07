import torch
from PIL import Image

from config import config
from model import get_model_and_tokenizer

model, model.prior_pipe.image_encoder = get_model_and_tokenizer(config.model_path, 
                                                                'cuda', torch.bfloat16,
                                                                compile=False,)
# full half precision
model = model.to(torch.bfloat16)

examples = [
 '../../generative_recommender/Blue_Tigers_space/2o.png',
 '../../generative_recommender/Blue_Tigers_space/10o.png',
 '../../generative_recommender/Blue_Tigers_space/7o.png',
 '../../generative_recommender/Blue_Tigers_space/9o.png',
]

torch.manual_seed(0);
model.do_qual_val([[Image.open(j) for j in examples]], 
                  prior_guidance_scale=3,
                  decoder_guidance_scale=3,
                  k=config.k)

scores = torch.tensor([1]+[5]*(config.k))[None];
torch.manual_seed(0);
model.do_qual_val([[Image.open(j) for j in examples]], path='negval.png',
                  k=config.k, scores=scores,
                  prior_guidance_scale=3,
                  decoder_guidance_scale=3,
                  )

breakpoint()


'''
python app.py
'''

import random
import torch
import uuid

import gradio as gr
import pandas as pd
from PIL import Image

from config import config
from model import get_model_and_tokenizer

model, model.prior_pipe.image_encoder = get_model_and_tokenizer(config.model_path, 
                                                                'cuda', torch.bfloat16,
                                                                compile=True,)
k = model.k
device = "cuda"



torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@torch.no_grad
def generate_gpu(in_im_embs):
    in_im_embs = in_im_embs.to('cuda')

    positive_image_embeds = in_im_embs[0]
    negative_image_embeds = in_im_embs[1]

    images = model.kandinsky_pipe(
        num_inference_steps=50,
        image_embeds=positive_image_embeds,
        negative_image_embeds=negative_image_embeds,
        guidance_scale=3,
    ).images[0]

    cond = (
                model.prior_pipe.image_processor(images, return_tensors="pt")
                .pixel_values[0]
                .unsqueeze(0)
                .to(dtype=model.prior_pipe.image_encoder.dtype, device=device)
                )
    im_emb = model.prior_pipe.image_encoder(cond)["image_embeds"].detach().to('cpu').to(torch.float32)
    return images, im_emb

@torch.no_grad
def generate(in_im_embs, ):
    output, im_emb = generate_gpu(in_im_embs)
    nsfw = False#maybe_nsfw(output.images[0])
    
    name = str(uuid.uuid4()).replace("-", "")
    path = f"/tmp/{name}.png"
    
    if nsfw:
        gr.Warning("NSFW content detected.")
        # TODO could return an automatic dislike on the backend or neither as well; just would need refactoring.
        return None, im_emb
    
    output.save(path)
    return path, im_emb

@torch.no_grad
def sample_embs(prompt_embeds, scores):
    # we would like a beloved image
    target_score = scores.new_ones((scores.shape[0], 1)) * 5
    scores = torch.cat([target_score, scores], 1)
    if isinstance(prompt_embeds, list):
        prompt_embeds = torch.cat(prompt_embeds, 0)[None]

    if prompt_embeds.shape[1] < k:
        scores = torch.nn.functional.pad(scores, [0, 1+k-scores.shape[1]])
    
    prompt_embeds = prompt_embeds.to(device)
    scores = scores.to(device)
    image_embeds = model.prior_pipe(prompt_embeds=prompt_embeds, 
                                    k=k,
                                    scores=scores,
                                    guidance_scale=3,
                                    ).to_tuple()
    return image_embeds

def get_user_emb(embs, ys, greedyish=True):
    '''
    Take in previous embeddings and scores, and return a new sampled embedding.
    
    We can be "greedyish" by taking the latest scored embeddings.
    
    '''
    if len(ys) > k:
        inds = range(len(ys))
        negs_inds = [ind for ind in inds if ys[ind] < 3]
        pos_inds = [ind for ind in inds if ys[ind] >= 3]
        if len(pos_inds) < k:
            # we have not-enough pos inds to lose any
            inds = pos_inds + random.sample(inds, k-len(pos_inds))
        else:
            # arbitrary minimum supported ratings
            assert k > 4, f'{k=} must be greater than 4.'

            if greedyish:
                split = k//2
                inds = pos_inds[-split:] + negs_inds[-split:]
            else:
                # we'll have at least 4 positive ratings
                min_pos = min(len(pos_inds), 4)
                # and a random number of others
                pos_split = random.randint(min_pos, k-min_pos)
                inds = random.sample(pos_inds, pos_split) + random.sample(negs_inds, k-pos_split)
    else:
        # barely any ratings
        inds = range(len(ys))

    # shuffle as we still have an absolute position embedding
    random.shuffle(inds)
    picked_embeddings = [embs[i] for i in inds]
    scores = torch.tensor([ys[i] for i in inds])[None]

    image_embeds = sample_embs(picked_embeddings, scores,)
    image_embeds = torch.stack(image_embeds, 0)
    return image_embeds

def next_image(calibrate_prompts, embs, ys):
    with torch.no_grad():
        if len(calibrate_prompts) > 0:
            cal_path = calibrate_prompts.pop(0)
            pil_im = Image.open(cal_path).convert('RGB')
            im_emb = encode_space(pil_im)
            image = cal_path
        else:
            im_embs_stack = get_user_emb(embs, ys)
            image, im_emb = generate(im_embs_stack)
        return image, im_emb, calibrate_prompts


def start(calibrate_prompts):
    embs, ys = [], []
    image, im_emb, calibrate_prompts = next_image(calibrate_prompts, embs, ys)
    return (
        image,
        calibrate_prompts,
        embs,
        ys,
        im_emb,
        gr.update(value=None, visible=True),   # rating: reset + reveal
        gr.update(visible=False),              # start_btn: hide
        gr.update(visible=True),               # submit_btn: reveal
    )

def choose(rating, img_path, calibrate_prompts, embs, ys, current_emb):
    '''
    Runs once per round: scores the image just shown, then serves the next one.
    '''
    if rating is None:
        gr.Warning("Please pick a rating from 1 to 5 before continuing.")
        return img_path, calibrate_prompts, embs, ys, current_emb, rating

    if img_path is None:
        rating = 1 # filtered/NSFW image -> treat as a strong dislike

    embs = embs + [current_emb]
    ys = ys + [rating]

    image, im_emb, calibrate_prompts = next_image(calibrate_prompts, embs, ys)
    return image, calibrate_prompts, embs, ys, im_emb, None


css = '''.gradio-container{max-width: 700px !important}
#description{text-align: center}
#description h1, #description h3{display: block}
#description p{margin-top: 0}
.fade-in-out {animation: fadeInOut 3s forwards}
@keyframes fadeInOut {
    0% {
      background: var(--bg-color);
    }
    100% {
      background: var(--button-secondary-background-fill);
    }
}
'''


js_head = '''
<script>
document.addEventListener('keydown', function(event) {
    const key = event.key;
    if (['1', '2', '3', '4', '5'].includes(key)) {
        const rating = key;
        const radioInputs = document.querySelectorAll('#rating input[type="radio"]');
        let matched = null;
        radioInputs.forEach(function(input) {
            if (input.value === rating) {
                matched = input;
            }
        });
        if (matched) {
            matched.checked = true;
            matched.dispatchEvent(new Event('change', { bubbles: true }));
            const submitBtn = document.getElementById('submit_rating');
            if (submitBtn) submitBtn.click();
        }
    }
});

document.body.addEventListener('click', function(event) {
    const startBtn = document.getElementById('start_btn');
    const submitBtn = document.getElementById('submit_rating');
    if (event.target === startBtn || event.target.closest('#start_btn')) {
        fadeInOut(startBtn, '#ff7a17');
    } else if (event.target === submitBtn || event.target.closest('#submit_rating')) {
        fadeInOut(submitBtn, '#ff7a17');
    }
});

function fadeInOut(el, color) {
  if (!el) return;
  el.style.setProperty('--bg-color', color);
  el.classList.remove('fade-in-out');
  void el.offsetWidth;
  el.classList.add('fade-in-out');
  el.addEventListener('animationend', () => {
    el.classList.remove('fade-in-out');
  }, {once: true});
}
</script>
'''

with gr.Blocks(css=css, head=js_head) as demo:
    gr.Markdown('''# Zahir
### Generative Recommenders for Exporation of Possible Images

Explore the latent space without text prompts based on your preferences. Learn more on [the write-up](https://rynmurdock.github.io/writing/generative_recommenders).
    ''', elem_id="description")
    user_id = gr.State()
    # calibration videos -- this is a misnomer now :D
    calibrate_prompts = gr.State([
    './image_init/5o.png',
    './image_init/2o.png',
    './image_init/6o.png',
    './image_init/7o.png',
    './image_init/1o.png',
    './image_init/8o.png',
    './image_init/3o.png',
    './image_init/4o.png',
    './image_init/10o.png',
    './image_init/9o.png',
    ])
    embs = gr.State([])        # collected image embeddings, one per rated image
    ys = gr.State([])          # collected int ratings, 1-5, aligned with embs
    current_emb = gr.State(None)  # embedding of the image currently displayed
    def l():
        return None

    with gr.Row(elem_id='output-image'):
        img = gr.Image(
            label='Lightning',
            interactive=False,
            elem_id="output_im",
            type='filepath',
            height=512,
        )
    
    with gr.Row(equal_height=False):
        rating = gr.Radio(
            choices=[1, 2, 3, 4, 5],
            value=None,
            type="value",
            label="Hated (1) to Loved (5)",
            elem_id="rating",
        )
        submit_btn = gr.Button("Next", elem_id="submit_rating", scale=.5)
    
    with gr.Row():
        start_btn = gr.Button("Start", elem_id="start_btn")

    start_btn.click(
        fn=start,
        inputs=[calibrate_prompts],
        outputs=[img, calibrate_prompts, embs, ys, current_emb, rating, start_btn, submit_btn],
    )

    submit_btn.click(
        fn=choose,
        inputs=[rating, img, calibrate_prompts, embs, ys, current_emb],
        outputs=[img, calibrate_prompts, embs, ys, current_emb, rating],
    )
        

    with gr.Row():
        html = gr.HTML('''<div style='text-align:center; font-size:20px'>You will calibrate for several images and then roam. </ div><br><br><br>

<br><br>
<div style='text-align:center; font-size:14px'>Thanks to @multimodalart for their contributions to the demo, esp. the interface and @maxbittker for feedback.
</ div>''')

@torch.no_grad
def encode_space(x):
    im = (
            model.prior_pipe.image_processor(x, return_tensors="pt")
            .pixel_values[0]
            .unsqueeze(0)
            .to(dtype=model.prior_pipe.image_encoder.dtype, device=device)
            )
    im_emb = model.prior_pipe.image_encoder(im)["image_embeds"]
    return im_emb.detach().to('cpu').to(torch.float32)


demo.launch(share=True,)


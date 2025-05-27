
import gradio as gr
import random
import time
import torch


import config
from model import get_model_and_tokenizer

model, model.prior_pipe.image_encoder = get_model_and_tokenizer(config.model_path, 
                                                                'cuda', torch.bfloat16)

# TODO unify/merge origin and this
# TODO save & restart from (if it exists) dataframe parquet

device = "cuda"


import spaces
import matplotlib.pyplot as plt

import os
import gradio as gr
import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler

import random
import time
from PIL import Image
# from safety_checker_improved import maybe_nsfw


torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

prevs_df = pd.DataFrame(columns=['paths', 'embeddings', 'ips', 'user:rating', 'latest_user_to_rate', 'from_user_id', 'text', 'gemb'])

import spaces
start_time = time.time()

####################### Setup Model
from diffusers import EulerDiscreteScheduler
from PIL import Image
import uuid


@spaces.GPU()
def generate_gpu(in_im_embs, prompt='the scene'):
    with torch.no_grad():
        in_im_embs = in_im_embs.to('cuda')

        negative_image_embeds = in_im_embs[0] # model.prior_pipe.get_zero_embed()
        positive_image_embeds = in_im_embs[1]

        images = model.kandinsky_pipe(
            num_inference_steps=50,
            image_embeds=positive_image_embeds,
            negative_image_embeds=negative_image_embeds,
            guidance_scale=11,
        ).images[0]
        cond = (
                    model.prior_pipe.image_processor(images, return_tensors="pt")
                    .pixel_values[0]
                    .unsqueeze(0)
                    .to(dtype=model.prior_pipe.image_encoder.dtype, device=device)
                    )
        im_emb = model.prior_pipe.image_encoder(cond)["image_embeds"]
    return images, im_emb


def generate(in_im_embs, ):
    output, im_emb = generate_gpu(in_im_embs)
    nsfw = False#maybe_nsfw(output.images[0])
    
    name = str(uuid.uuid4()).replace("-", "")
    path = f"/tmp/{name}.png"
    
    if nsfw:
        gr.Warning("NSFW content detected.")
        # TODO could return an automatic dislike of auto dislike on the backend for neither as well; just would need refactoring.
        return None, im_emb
    
    output.save(path)
    return path, im_emb


#######################

@spaces.GPU()
def sample_embs(prompt_embeds):
    latent = torch.randn(prompt_embeds.shape[0], 1, prompt_embeds.shape[-1])
    if prompt_embeds.shape[1] < 8: # TODO grab as `k` arg from config
            prompt_embeds = torch.nn.functional.pad(prompt_embeds, [0, 0, 0, 8-prompt_embeds.shape[1]])
    assert prompt_embeds.shape[1] == 8, f"The model is set to take `k`` cond image embeds but is shape {prompt_embeds.shape}"
    image_embeds = model(latent.to('cuda'), prompt_embeds.to('cuda')).predicted_image_embedding

    return image_embeds

@spaces.GPU()
def get_user_emb(embs, ys):
    positives = [e for e, ys in zip(embs, ys) if ys == 1]
    embs = random.sample(positives, min(8, len(positives)))
    positives = torch.stack(embs, 1)

    negs = [e for e, ys in zip(embs, ys) if ys == 0]
    negative_embs = random.sample(negs, min(8, len(negs)))
    negatives = torch.stack(negative_embs, 1)

    image_embeds = torch.stack([sample_embs(negatives), sample_embs(positives)])

    return image_embeds


def background_next_image():
        global prevs_df
        # only let it get N (maybe 3) ahead of the user
        #not_rated_rows = prevs_df[[i[1]['user:rating'] == {' ': ' '} for i in prevs_df.iterrows()]]
        rated_rows = prevs_df[[i[1]['user:rating'] != {' ': ' '} for i in prevs_df.iterrows()]]
        if len(rated_rows) < 4:
            time.sleep(.1)
        #    not_rated_rows = prevs_df[[i[1]['user:rating'] == {' ': ' '} for i in prevs_df.iterrows()]]
            return

        user_id_list = set(rated_rows['latest_user_to_rate'].to_list())
        for uid in user_id_list:
            rated_rows = prevs_df[[i[1]['user:rating'].get(uid, None) is not None for i in prevs_df.iterrows()]]
            not_rated_rows = prevs_df[[i[1]['user:rating'].get(uid, None) is None for i in prevs_df.iterrows()]]
            
            # we need to intersect not_rated_rows from this user's embed > 7. Just add a new column on which user_id spawned the 
            #   media. 
            
            unrated_from_user = not_rated_rows[[i[1]['from_user_id'] == uid for i in not_rated_rows.iterrows()]]

            # we don't compute more after n are in the queue for them
            if len(unrated_from_user) >= 10:
                continue
            
            if len(rated_rows) < 5:
                continue            

            global glob_idx
            glob_idx += 1
            
            ems = rated_rows['embeddings'].to_list()
            ys = [i[uid][0] for i in rated_rows['user:rating'].to_list()]

            emz = get_user_emb(ems, ys)
            img, embs = generate(emz)
            
            if img:
                tmp_df = pd.DataFrame(columns=['paths', 'embeddings', 'ips', 'user:rating', 'latest_user_to_rate', 'text', 'gemb'])
                tmp_df['paths'] = [img]
                tmp_df['embeddings'] = [embs.to(torch.float32).to('cpu')]
                tmp_df['user:rating'] = [{' ': ' '}]
                tmp_df['from_user_id'] = [uid]
                tmp_df['text'] = ['']
                prevs_df = pd.concat((prevs_df, tmp_df))
                # we can free up storage by deleting the image
                if len(prevs_df) > 500:
                    oldest_path = prevs_df.iloc[6]['paths']
                    if os.path.isfile(oldest_path):
                        os.remove(oldest_path)
                    else:
                        # If it fails, inform the user.
                        print("Error: %s file not found" % oldest_path)
                    # only keep 50 images & embeddings & ips, then remove oldest besides calibrating
                    prevs_df = pd.concat((prevs_df.iloc[:6], prevs_df.iloc[7:]))
    
def pluck_img(user_id):
    rated_rows = prevs_df[[i[1]['user:rating'].get(user_id, None) is not None for i in prevs_df.iterrows()]]
    ems = rated_rows['embeddings'].to_list()
    ys = [i[user_id][0] for i in rated_rows['user:rating'].to_list()]
    user_emb = get_user_emb(ems, ys)

    not_rated_rows = prevs_df[[i[1]['user:rating'].get(user_id, 'gone') == 'gone' for i in prevs_df.iterrows()]]
    while len(not_rated_rows) == 0:
        not_rated_rows = prevs_df[[i[1]['user:rating'].get(user_id, 'gone') == 'gone' for i in prevs_df.iterrows()]]
        time.sleep(.1)
        # TODO optimize this lol
    best_sim = -10000000
    for i in not_rated_rows.iterrows():
        # TODO sloppy .to but it is 3am.
        sim = torch.cosine_similarity(i[1]['embeddings'].detach().to('cpu'), user_emb.detach().to('cpu'), -1)
        if len(sim) > 1: sim = sim[1]
        if sim.squeeze() > best_sim:
            best_sim = sim
            best_row = i[1]
    img = best_row['paths']
    return img

def next_image(calibrate_prompts, user_id):
    with torch.no_grad():
        if len(calibrate_prompts) > 0:
            cal_video = calibrate_prompts.pop(0)
            image = prevs_df[prevs_df['paths'] == cal_video]['paths'].to_list()[0]
            return image, calibrate_prompts, 
        else:
            image = pluck_img(user_id)
            return image, calibrate_prompts









def start(_, calibrate_prompts, user_id, request: gr.Request):
    user_id = int(str(time.time())[-7:].replace('.', ''))
    image, calibrate_prompts = next_image(calibrate_prompts, user_id)
    return [
            gr.Button(value='👍', interactive=True), 
            gr.Button(value='Neither (Space)', interactive=True, visible=False), 
            gr.Button(value='👎', interactive=True),
            gr.Button(value='Start', interactive=False),
            gr.Button(value='👍 Content', interactive=True, visible=False),
            gr.Button(value='👍 Style', interactive=True, visible=False),
            image,
            calibrate_prompts,
            user_id,
            
            ]


def choose(img, choice, calibrate_prompts, user_id, request: gr.Request):
    global prevs_df
    
    
    if choice == '👍':
        choice = [1, 1]
    elif choice == 'Neither (Space)':
        img, calibrate_prompts,  = next_image(calibrate_prompts, user_id)
        return img, calibrate_prompts, 
    elif choice == '👎':
        choice = [0, 0]
    elif choice == '👍 Style':
        choice = [0, 1]
    elif choice == '👍 Content':
        choice = [1, 0]
    else:
        assert False, f'choice is {choice}'
    
    # if we detected NSFW, leave that area of latent space regardless of how they rated chosen.
    # TODO skip allowing rating & just continue

    if img is None:
        print('NSFW -- choice is disliked')
        choice = [0, 0]
    
    row_mask = [p.split('/')[-1] in img for p in prevs_df['paths'].to_list()]
    # if it's still in the dataframe, add the choice
    if len(prevs_df.loc[row_mask, 'user:rating']) > 0:
        prevs_df.loc[row_mask, 'user:rating'][0][user_id] = choice
        print(row_mask, prevs_df.loc[row_mask, 'latest_user_to_rate'], [user_id])
        prevs_df.loc[row_mask, 'latest_user_to_rate'] = [user_id]
    img, calibrate_prompts = next_image(calibrate_prompts, user_id)
    return img, calibrate_prompts

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
    if (event.key === 'a' || event.key === 'A') {
        // Trigger click on 'dislike' if 'A' is pressed
        document.getElementById('dislike').click();
    } else if (event.key === ' ' || event.keyCode === 32) {
        // Trigger click on 'neither' if Spacebar is pressed
        document.getElementById('neither').click();
    } else if (event.key === 'l' || event.key === 'L') {
        // Trigger click on 'like' if 'L' is pressed
        document.getElementById('like').click();
    }
});
function fadeInOut(button, color) {
  button.style.setProperty('--bg-color', color);
  button.classList.remove('fade-in-out');
  void button.offsetWidth; // This line forces a repaint by accessing a DOM property
  
  button.classList.add('fade-in-out');
  button.addEventListener('animationend', () => {
    button.classList.remove('fade-in-out'); // Reset the animation state
  }, {once: true});
}
document.body.addEventListener('click', function(event) {
    const target = event.target;
    if (target.id === 'dislike') {
      fadeInOut(target, '#ff1717');
    } else if (target.id === 'like') {
      fadeInOut(target, '#006500');
    } else if (target.id === 'neither') {
      fadeInOut(target, '#cccccc');
    }
});

</script>
'''

with gr.Blocks(css=css, head=js_head) as demo:
    gr.Markdown('''# Zahir
### Generative Recommenders for Exporation of Possible Images

Explore the latent space without text prompts based on your preferences. Learn more on [the write-up](https://rynmurdock.github.io/posts/2024/3/generative_recomenders/).
    ''', elem_id="description")
    user_id = gr.State()
    # calibration videos -- this is a misnomer now :D
    calibrate_prompts = [
    './5o.png',
    './2o.png',
    './6o.png',
    './7o.png',
    './1o.png',
    './8o.png',
    './3o.png',
    './4o.png',
    './10o.png',
    './9o.png',
    ]
    calibrate_prompts = gr.State(['image_init/'+c for c in calibrate_prompts])
    def l():
        return None

    with gr.Row(elem_id='output-image'):
        img = gr.Image(
            label='Lightning',
            interactive=False,
            elem_id="output_im",
            type='filepath',
        )
    
    
    
    with gr.Row(equal_height=True):
        b3 = gr.Button(value='👎', interactive=False, elem_id="dislike")

        b2 = gr.Button(value='Neither (Space)', interactive=False, elem_id="neither", visible=False)

        b1 = gr.Button(value='👍', interactive=False, elem_id="like")
    with gr.Row(equal_height=True):
        b6 = gr.Button(value='👍 Style', interactive=False, elem_id="dislike like", visible=False)
        
        b5 = gr.Button(value='👍 Content', interactive=False, elem_id="like dislike", visible=False) 
        
        b1.click(
        choose, 
        [img, b1, calibrate_prompts, user_id],
        [img, calibrate_prompts, ],
        )
        b2.click(
        choose, 
        [img, b2, calibrate_prompts, user_id],
        [img, calibrate_prompts, ],
        )
        b3.click(
        choose, 
        [img, b3, calibrate_prompts, user_id],
        [img, calibrate_prompts, ],
        )
        b5.click(
        choose, 
        [img, b5, calibrate_prompts, user_id],
        [img, calibrate_prompts, ],
        )
        b6.click(
        choose, 
        [img, b6, calibrate_prompts, user_id],
        [img, calibrate_prompts, ],
        )
    with gr.Row():
        b4 = gr.Button(value='Start')
        b4.click(start,
                 [b4, calibrate_prompts, user_id],
                 [b1, b2, b3, b4, b5, b6, img, calibrate_prompts, user_id, ]
                 )
    with gr.Row():
        html = gr.HTML('''<div style='text-align:center; font-size:20px'>You will calibrate for several images and then roam. </ div><br><br><br>

<br><br>
<div style='text-align:center; font-size:14px'>Thanks to @multimodalart for their contributions to the demo, esp. the interface and @maxbittker for feedback.
</ div>''')

# TODO quiet logging
scheduler = BackgroundScheduler()
scheduler.add_job(func=background_next_image, trigger="interval", seconds=.2)
scheduler.start()

# TODO shouldn't call this before gradio launch, yeah?
@spaces.GPU()
def encode_space(x):
    im = (
            model.prior_pipe.image_processor(x, return_tensors="pt")
            .pixel_values[0]
            .unsqueeze(0)
            .to(dtype=model.prior_pipe.image_encoder.dtype, device=device)
            )
    im_emb = model.prior_pipe.image_encoder(im)["image_embeds"]
    return im_emb.detach().to('cpu').to(torch.float32)

# prep our calibration videos
m_calibrate = [ # DO NOT NAME THESE PNGs JUST NUMBERS! apparently we assign images by number
    ('./1o.png', 'describe the scene: omens in the suburbs'),
    ('./2o.png', 'describe the scene: geometric abstract art of a windmill'),
    ('./3o.png', 'describe the scene: memento mori'),
    ('./4o.png', 'describe the scene: a green plate with anespresso'),
    ('./5o.png', '5 '),
    ('./6o.png', '6 '),
    ('./7o.png', '7 '),
    ('./8o.png', '8 '),
    ('./9o.png', '9 '),
    ('./10o.png', '10 '),
    ]
m_calibrate = [('image_init/'+c[0], c[1]) for c in m_calibrate]
for im, txt in m_calibrate:
    tmp_df = pd.DataFrame(columns=['paths', 'embeddings', 'ips', 'user:rating', 'text', 'gemb'])
    tmp_df['paths'] = [im]
    image = Image.open(im).convert('RGB')
    im_emb = encode_space(image)
    
    tmp_df['embeddings'] = [im_emb.detach().to('cpu')]
    tmp_df['user:rating'] = [{' ': ' '}]
    tmp_df['text'] = [txt]
    prevs_df = pd.concat((prevs_df, tmp_df))

glob_idx = 0
demo.launch(share=True,)



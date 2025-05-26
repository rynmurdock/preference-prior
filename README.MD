# The Other Tiger

## tl;dr
Train on embeddings of media preferred by a specific user -> produce embeddings of media they may enjoy.

In our case here, we take the ECLIPSE `text embedding -> image embedding` prior (https://arxiv.org/abs/2312.04655) and finetune it to become a `preferred image embeddings -> heldout image embedding` prior.

### Related work:

Patron et al. models preference using a diffusion prior and condition on user ids with ratings: https://arxiv.org/abs/2502.18477

Wang et al. models preference using a generator conditioned on averaged CLIP embeddings of users:  https://arxiv.org/abs/2304.03516

My previous work based on Collaborative Filtering with CLIP embeddings: https://github.com/rynmurdock/generative_recommender


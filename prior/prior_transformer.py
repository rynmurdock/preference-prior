import sys
sys.path.append("..")

from dataclasses import dataclass
from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.attention_processor import AttentionProcessor
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin



class AttnProcessor:
    r"""
    Default processor for performing attention-related computations.
    """

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        rope_emb: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = apply_rotary_emb(query, rope_emb).squeeze()
        key = apply_rotary_emb(key, rope_emb).squeeze()

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states



def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: torch.Tensor | tuple[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.
    Returns:
        tuple[torch.Tensor, torch.Tensor]: tuple of modified query tensor and key tensor with rotary embeddings.
    """
    cos, sin = freqs_cis  # [B, S, D]
    cos = cos[:, :, :]
    sin = sin[:, :, :]

    cos, sin = cos.to(x.device), sin.to(x.device)

    # Used for flux, cogvideox, hunyuan-dit
    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, H, S, D//2]
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(2)

    out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
    return out


def get_1d_rotary_pos_embed(
    dim: int,
    pos: np.ndarray | int,
    theta: float = 1000.0,
    linear_factor=1.0,
    ntk_factor=1.0,
    freqs_dtype=torch.float32,  #  torch.float32, torch.float64 (flux)
):
    '''
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.
    '''

    assert dim % 2 == 0

    if isinstance(pos, int):
        pos = torch.arange(pos)
    if isinstance(pos, np.ndarray):
        pos = torch.from_numpy(pos)  # type: ignore  # [S]

    theta = theta * ntk_factor
    freqs = (
        1.0 / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=pos.device) / dim)) / linear_factor
    )  # [D/2]
    freqs = torch.outer(pos, freqs)  # type: ignore   # [S, D/2]

    freqs_cos = freqs.cos().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()  # [S, D]
    freqs_sin = freqs.sin().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()  # [S, D]
    return freqs_cos, freqs_sin


class FluxPosEmbed(nn.Module):
    # modified from https://github.com/black-forest-labs/flux/blob/c00d7c60b085fce8058b9df845e036090873f2ce/src/flux/modules/layers.py#L11
    def __init__(self, theta: int, axes_dim: list[int]):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        n_axes = ids.shape[-1]
        pos = ids.float()
        # batch
        ba_cos_out = []
        ba_sin_out = []
        for b in range(len(ids)):
            cos_out = []
            sin_out = []
            for i in range(n_axes):
                cos, sin = get_1d_rotary_pos_embed(
                    self.axes_dim[i],
                    pos[b, :, i,],
                    theta=self.theta,
                    freqs_dtype=torch.float32,
                )
                cos_out.append(cos)
                sin_out.append(sin)
            freqs_cos = torch.cat(cos_out, dim=-1).to(ids.device)
            freqs_sin = torch.cat(sin_out, dim=-1).to(ids.device)
            ba_cos_out.append(freqs_cos)
            ba_sin_out.append(freqs_sin)
        fin_freqs_cos = torch.stack(ba_cos_out, dim=0).to(ids.device)
        fin_freqs_sin = torch.stack(ba_sin_out, dim=0).to(ids.device)
        
        sh = fin_freqs_sin.shape
        assert sh[0] == ids.shape[0], f'{ids.shape=} {sh=}'
        assert len(sh) == 3, f'{sh=}'
        
        return fin_freqs_cos, fin_freqs_sin

@dataclass
class PriorTransformerOutput(BaseOutput):
    """
    The output of [`PriorTransformer`].

    Args:
        predicted_image_embedding (`torch.FloatTensor` of shape `(batch_size, embedding_dim)`):
            The predicted CLIP image embedding conditioned on the CLIP text embedding input.
    """

    predicted_image_embedding: torch.FloatTensor


class PriorTransformer(ModelMixin, ConfigMixin):
    """
    A Prior Transformer model.

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 32): The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 64): The number of channels in each head.
        num_layers (`int`, *optional*, defaults to 20): The number of layers of Transformer blocks to use.
        embedding_dim (`int`, *optional*, defaults to 768): The dimension of the model input `hidden_states`
        num_embeddings (`int`, *optional*, defaults to 77):
            The number of embeddings of the model input `hidden_states`
        additional_embeddings (`int`, *optional*, defaults to 4): The number of additional tokens appended to the
            projected `hidden_states`. The actual length of the used `hidden_states` is `num_embeddings +
            additional_embeddings`.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        time_embed_act_fn (`str`, *optional*, defaults to 'silu'):
            The activation function to use to create timestep embeddings.
        norm_in_type (`str`, *optional*, defaults to None): The normalization layer to apply on hidden states before
            passing to Transformer blocks. Set it to `None` if normalization is not needed.
        embedding_proj_norm_type (`str`, *optional*, defaults to None):
            The normalization layer to apply on the input `proj_embedding`. Set it to `None` if normalization is not
            needed.
        encoder_hid_proj_type (`str`, *optional*, defaults to `linear`):
            The projection layer to apply on the input `encoder_hidden_states`. Set it to `None` if
            `encoder_hidden_states` is `None`.
        added_emb_type (`str`, *optional*, defaults to `prd`): Additional embeddings to condition the model.
            Choose from `prd` or `None`. if choose `prd`, it will prepend a token indicating the (quantized) dot
            product between the text embedding and image embedding as proposed in the unclip paper
            https://arxiv.org/abs/2204.06125 If it is `None`, no additional embeddings will be prepended.
        time_embed_dim (`int, *optional*, defaults to None): The dimension of timestep embeddings.
            If None, will be set to `num_attention_heads * attention_head_dim`
        embedding_proj_dim (`int`, *optional*, default to None):
            The dimension of `proj_embedding`. If None, will be set to `embedding_dim`.
        clip_embed_dim (`int`, *optional*, default to None):
            The dimension of the output. If None, will be set to `embedding_dim`.
    """

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 20,
        attention_head_dim: int = 64,
        num_layers: int = 36,
        embedding_dim: int = 1280,
        num_embeddings=77,
        additional_embeddings=3, # as we have removed the time embedding
        dropout: float = 0.0,
        # time_embed_act_fn: str = "silu",
        norm_in_type: Optional[str] = None,  # layer
        embedding_proj_norm_type: Optional[str] = None,  # layer
        encoder_hid_proj_type: Optional[str] = "linear",  # linear
        added_emb_type: Optional[str] = "prd",  # prd
        # time_embed_dim: Optional[int] = None,
        embedding_proj_dim: Optional[int] = None,
        clip_embed_dim: Optional[int] = None,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim
        self.additional_embeddings = additional_embeddings

        # time_embed_dim = time_embed_dim or inner_dim
        embedding_proj_dim = embedding_proj_dim or embedding_dim
        clip_embed_dim = clip_embed_dim or embedding_dim

        # self.time_proj = Timesteps(inner_dim, True, 0)
        # self.time_embedding = TimestepEmbedding(inner_dim, time_embed_dim, out_dim=inner_dim, act_fn=time_embed_act_fn)

        self.proj_in = nn.Linear(embedding_dim, inner_dim)

        if embedding_proj_norm_type is None:
            self.embedding_proj_norm = None
        elif embedding_proj_norm_type == "layer":
            self.embedding_proj_norm = nn.LayerNorm(embedding_proj_dim)
        else:
            raise ValueError(f"unsupported embedding_proj_norm_type: {embedding_proj_norm_type}")

        self.embedding_proj = nn.Linear(embedding_proj_dim, inner_dim)

        if encoder_hid_proj_type is None:
            self.encoder_hidden_states_proj = None
        elif encoder_hid_proj_type == "linear":
            self.encoder_hidden_states_proj = nn.Linear(embedding_dim, inner_dim)
        else:
            raise ValueError(f"unsupported encoder_hid_proj_type: {encoder_hid_proj_type}")

        self.positional_embedding = nn.Parameter(torch.zeros(1, num_embeddings + additional_embeddings, inner_dim))

        # separate rope embedding for scores over embeddings
        self.rope_embed_layer = FluxPosEmbed(1000, [inner_dim])

        if added_emb_type == "prd":
            self.prd_embedding = nn.Parameter(torch.zeros(1, 1, inner_dim))
        elif added_emb_type is None:
            self.prd_embedding = None
        else:
            raise ValueError(
                f"`added_emb_type`: {added_emb_type} is not supported. Make sure to choose one of `'prd'` or `None`."
            )

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    activation_fn="gelu",
                    attention_bias=True,
                )
                for d in range(num_layers)
            ]
        )

        if norm_in_type == "layer":
            self.norm_in = nn.LayerNorm(inner_dim)
        elif norm_in_type is None:
            self.norm_in = None
        else:
            raise ValueError(f"Unsupported norm_in_type: {norm_in_type}.")

        self.norm_out = nn.LayerNorm(inner_dim)

        self.proj_to_clip_embeddings = nn.Linear(inner_dim, clip_embed_dim)

        causal_attention_mask = torch.full(
            [num_embeddings + additional_embeddings, num_embeddings + additional_embeddings], -10000.0
        )
        causal_attention_mask.triu_(1)
        causal_attention_mask = causal_attention_mask[None, ...]
        self.register_buffer("causal_attention_mask", causal_attention_mask, persistent=False)

        self.clip_mean = nn.Parameter(torch.zeros(1, clip_embed_dim))
        self.clip_std = nn.Parameter(torch.zeros(1, clip_embed_dim))

        self.set_default_attn_processor()

    @property
    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "set_processor"):
                processors[f"{name}.processor"] = module.processor

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.set_default_attn_processor
    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        self.set_attn_processor(AttnProcessor())

    def forward(
        self,
        hidden_states,
        # timestep: Union[torch.Tensor, float, int],
        proj_embedding: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        return_dict: bool = True,
        scores=None,
        target_scores=None,
    ):
        """
        The [`PriorTransformer`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, embedding_dim)`):
                The currently predicted image embeddings.
            timestep (`torch.LongTensor`):
                Current denoising step.
            proj_embedding (`torch.FloatTensor` of shape `(batch_size, embedding_dim)`):
                Projected embedding vector the denoising process is conditioned on.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, num_embeddings, embedding_dim)`):
                Hidden states of the text embeddings the denoising process is conditioned on.
            attention_mask (`torch.BoolTensor` of shape `(batch_size, num_embeddings)`):
                Text mask for the text embeddings.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.prior_transformer.PriorTransformerOutput`] instead of a plain
                tuple.

        Returns:
            [`~models.prior_transformer.PriorTransformerOutput`] or `tuple`:
                If return_dict is True, a [`~models.prior_transformer.PriorTransformerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.
        """
        batch_size = hidden_states.shape[0]
        assert encoder_hidden_states == None, 'not using this'

        # timesteps = timestep
        # if not torch.is_tensor(timesteps):
        #     timesteps = torch.tensor([timesteps], dtype=torch.long, device=hidden_states.device)
        # elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
        #     timesteps = timesteps[None].to(hidden_states.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        # timesteps = timesteps * torch.ones(batch_size, dtype=timesteps.dtype, device=timesteps.device)

        # timesteps_projected = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might be fp16, so we need to cast here.
        # timesteps_projected = timesteps_projected.to(dtype=self.dtype)
        # time_embeddings = self.time_embedding(timesteps_projected)

        if self.embedding_proj_norm is not None:
            proj_embedding = self.embedding_proj_norm(proj_embedding)

        proj_embeddings = self.embedding_proj(proj_embedding)
        if self.encoder_hidden_states_proj is not None and encoder_hidden_states is not None:
            encoder_hidden_states = self.encoder_hidden_states_proj(encoder_hidden_states)
        # elif self.encoder_hidden_states_proj is not None and encoder_hidden_states is None:
        #     raise ValueError("`encoder_hidden_states_proj` requires `encoder_hidden_states` to be set")

        hidden_states = self.proj_in(hidden_states)

        # TODO this really also ought to derive from config's `k`
        positional_embeddings = self.positional_embedding.to(hidden_states.dtype)

        additional_embeds = []
        additional_embeddings_len = 0

        if encoder_hidden_states is not None:
            additional_embeds.append(encoder_hidden_states)
            additional_embeddings_len += encoder_hidden_states.shape[1]

        if len(proj_embeddings.shape) == 2:
            proj_embeddings = proj_embeddings[:, None, :]

        if len(hidden_states.shape) == 2:
            hidden_states = hidden_states[:, None, :]

        additional_embeds = additional_embeds + [
            hidden_states,
            proj_embeddings,
            # time_embeddings[:, None, :],
        ]

        if self.prd_embedding is not None:
            prd_embedding = self.prd_embedding.to(hidden_states.dtype).expand(batch_size, -1, -1)
            additional_embeds.append(prd_embedding)

        hidden_states = torch.cat(
            additional_embeds,
            dim=1,
        )

        # Allow positional_embedding to not include the `addtional_embeddings` and instead pad it with zeros for these additional tokens
        additional_embeddings_len = additional_embeddings_len + proj_embeddings.shape[1] + 1
        if positional_embeddings.shape[1] < hidden_states.shape[1]:
            positional_embeddings = F.pad(
                positional_embeddings,
                (
                    0,
                    0,
                    additional_embeddings_len,
                    self.prd_embedding.shape[1] if self.prd_embedding is not None else 0,
                ),
                value=0.0,
            )

        hidden_states = hidden_states + positional_embeddings[:, :hidden_states.shape[1]]

        if target_scores is not None:
            scores = torch.cat([target_scores, scores], dim=1)
        
        # append for additional states
        prd_ids = scores.new_zeros((scores.shape[0], 1))
        scores = torch.cat([scores, prd_ids], 1)

        # same range as timesteps; better bet of similar functioning 
        #   than changing theta
        pref_ids = scores.unsqueeze(-1) * 200
        rope_emb = self.rope_embed_layer(pref_ids)

        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = F.pad(attention_mask, (0, self.additional_embeddings), value=0.0)
            attention_mask = (attention_mask[:, None, :] + self.causal_attention_mask).to(hidden_states.dtype)
            attention_mask = attention_mask.repeat_interleave(self.config.num_attention_heads, dim=0)

        if self.norm_in is not None:
            hidden_states = self.norm_in(hidden_states)

        assert pref_ids.shape[1] == hidden_states.shape[1], f'{hidden_states.shape=} should == {pref_ids.shape=}'
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, attention_mask=attention_mask,
                                  cross_attention_kwargs={'rope_emb':rope_emb})

        hidden_states = self.norm_out(hidden_states)

        if self.prd_embedding is not None:
            hidden_states = hidden_states[:, -1]
        else:
            hidden_states = hidden_states[:, additional_embeddings_len:]

        predicted_image_embedding = self.proj_to_clip_embeddings(hidden_states)

        if not return_dict:
            return (predicted_image_embedding,)

        return PriorTransformerOutput(predicted_image_embedding=predicted_image_embedding)

    def post_process_latents(self, prior_latents):
        prior_latents = (prior_latents * self.clip_std) + self.clip_mean
        return prior_latents
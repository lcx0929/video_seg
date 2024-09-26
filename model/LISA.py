from typing import List

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from transformers import BitsAndBytesConfig, CLIPVisionModel

from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_PATCH_TOKEN)

from .llava.model.language_model.llava_llama import (LlavaLlamaForCausalLM,
                                                     LlavaLlamaModel)
from .segment_anything import build_sam_vit_h

from typing import Optional
from math import sqrt
from PIL import Image
import numpy as np


class MultiHeadSelfAttention(nn.Module):
    dim_in: int  # input dimension
    dim_k: int   # key and query dimension
    dim_v: int   # value dimension
    num_heads: int  # number of heads, for each head, dim_* = dim_* // num_heads
 
    def __init__(self, dim_in=256, dim_k=256, dim_v=256, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        #维度必须能被num_head 整除
        assert dim_k % num_heads == 0 and dim_v % num_heads == 0, "dim_k and dim_v must be multiple of num_heads"
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = num_heads
        #定义线性变换矩阵
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_k // num_heads)
 
    def forward(self, x,kv):
        # x: tensor of shape (batch, n, dim_in)
        batch, n, dim_in = x.shape
        assert dim_in == self.dim_in
 
        nh = self.num_heads
        dk = self.dim_k // nh  # dim_k of each head
        dv = self.dim_v // nh  # dim_v of each head
 
        q = self.linear_q(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        k = self.linear_k(kv).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        v = self.linear_v(kv).reshape(batch, n, nh, dv).transpose(1, 2)  # (batch, nh, n, dv)
 
        dist = torch.matmul(q, k.transpose(2, 3)) * self._norm_fact  # batch, nh, n, n
        dist = torch.softmax(dist, dim=-1)  # batch, nh, n, n
 
        att = torch.matmul(dist, v)  # batch, nh, n, dv
        att = att.transpose(1, 2).reshape(batch, n, self.dim_v)  # batch, n, dim_v
        return att




class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead=8, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = MultiHeadSelfAttention(dim_in=d_model, dim_k=d_model, dim_v=d_model, num_heads=nhead)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        #memory = self.norm(memory)

        tgt2 = self.multihead_attn(tgt, memory)[0]
        tgt = tgt + self.dropout(tgt2)
        #tgt = self.norm(tgt)
        
        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(dim_in=d_model, dim_k=d_model, dim_v=d_model, num_heads=nhead)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(tgt, tgt)[0]
        tgt = tgt + self.dropout(tgt2)
        #tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        
        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        #tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        #tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")



def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,  # 100000.0,
    eps=1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss


class LisaMetaModel:
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(LisaMetaModel, self).__init__(config)

        self.config = config
        if not hasattr(self.config, "train_mask_decoder"):
            self.config.train_mask_decoder = kwargs["train_mask_decoder"]
            self.config.out_dim = kwargs["out_dim"]
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
        else:
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
            self.initialize_lisa_modules(self.config)

    def initialize_lisa_modules(self, config):
        # SAM
        self.visual_model = build_sam_vit_h(self.vision_pretrained)
        for param in self.visual_model.parameters():
            param.requires_grad = False
        if config.train_mask_decoder:
            self.visual_model.mask_decoder.train()
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True

        # Projection layer
        in_dim = config.hidden_size
        out_dim = config.out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True




class LisaModel(LisaMetaModel, LlavaLlamaModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(LisaModel, self).__init__(config, **kwargs)

        self.config.use_cache = False
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False


class LISAForCausalLM(LlavaLlamaForCausalLM):
    def __init__(
        self,
        config,
        tokenizer=None,
        **kwargs,
    ):
        if not hasattr(config, "train_mask_decoder"):
            config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
            config.mm_vision_tower = kwargs.get(
                "vision_tower", "/home/lcx/projcet/models/clip-vit-large-patch14"
            )
            self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
            self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
            self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)
        else:
            config.mm_vision_tower = config.vision_tower
        self.ce_loss_weight = 1.0
        self.dice_loss_weight = 0.5
        self.bce_loss_weight =2.0
        self.seg_token_idx = kwargs.pop("seg_token_idx")
        self.deams = 256
        self.num_frames = 8
        super().__init__(config)

        self.model = LisaModel(config, **kwargs)
        self.tokenizer = tokenizer
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        
        self.cross_attention = CrossAttentionLayer(d_model=self.deams, nhead=8)
        self.self_attention = SelfAttentionLayer(d_model=self.deams, nhead=8)
        self.ffn = FFNLayer(d_model=self.deams)
        self.query_feat = nn.Embedding(self.num_frames ,self.deams)
        # learnable query p.e.self.deams
        self.query_embed = nn.Embedding(self.num_frames ,self.deams)

        #self.fq_pos = nn.Embedding(num_frame_queries, hidden_dim)

    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            image_embeddings_list = []
            for i in range(pixel_values.shape[0]):
                torch.cuda.empty_cache()
                image_embeddings = self.model.visual_model.image_encoder(
                    pixel_values[i].unsqueeze(0)
                )
                image_embeddings_list.append(image_embeddings)
            torch.cuda.empty_cache()
            image_embeddings = torch.cat(image_embeddings_list, 0)
        return image_embeddings

    def forward(self, **kwargs):
        #import pdb;pdb.set_trace()
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)

    def model_forward(
        self,
        images: torch.FloatTensor,
        images_clip: torch.FloatTensor,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        offset: torch.LongTensor,
        masks_list: List[torch.FloatTensor],
        label_list: List[torch.Tensor],
        resize_list: List[tuple],
        #object_ids: List[torch.Tensor],
        inference: bool = False,
        **kwargs,
    ):  
        

        '''self.config.eos_token_id = self.tokenizer.eos_token_id
        self.config.bos_token_id = self.tokenizer.bos_token_id
        self.config.pad_token_id = self.tokenizer.pad_token_id
        self.resize_token_embeddings(len(self.tokenizer))'''
        #for frame_id in range(images.shape[1]):
        if len(images.shape)==4:
            image_embeddings = self.get_visual_embs(images)
            
        else:
            image_embeddings = [] 
            for j in range(images.shape[0]):
                for i in range(images.shape[1]):
                    image = images[j,i,:,:,:].squeeze(1).unsqueeze(0)
                    image_embeddings.append(self.get_visual_embs(image))
            image_embeddings = torch.cat(image_embeddings,0)
        batch_size = images.shape[0]
        assert batch_size == len(offset) - 1
        #seg_token_mask1 = input_ids[:, 1:] == self.seg_token_idx 
        #a = torch.sum(seg_token_mask1)
        '''if len(object_ids) != 0 :
            #seg_token_mask = torch.zeros_like(input_ids[:, 1:], dtype=torch.bool)
            mask = torch.zeros_like(input_ids[:, 1:], dtype=torch.bool)
            seg_token_mask = input_ids[:, 1:] == self.seg_token_idx 

            for object_id in object_ids:
                mask = mask_consecutive_sequence(mask, input_ids[:, 1:],object_id)
            mask = seg_token_mask | mask
            a = torch.sum(mask)
                #seg_token_mask = input_ids[:, 1:] == object_id
                #mask = mask | seg_token_mask
            seg_token_mask = mask

        else: 
            seg_token_mask = input_ids[:, 1:] == self.seg_token_idx '''
 
        ''' mask = torch.zeros_like(input_ids[:, 1:], dtype=torch.bool)
        for i in self.seg_token_idx: 
            seg_token_mask = input_ids[:, 1:] == i
            mask = mask | seg_token_mask
        seg_token_mask = mask'''
        seg_token_mask = input_ids[:, 1:] == self.seg_token_idx 


        b = torch.sum(seg_token_mask)
        #print(b)
        seg_token_mask = torch.cat(
            [
                seg_token_mask,
                torch.zeros((seg_token_mask.shape[0], 1)).bool().cuda(),
            ],
            dim=1,
        )
        # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
        seg_token_mask = torch.cat(
            [torch.zeros((seg_token_mask.shape[0], 255)).bool().cuda(), seg_token_mask],
            dim=1,
        )
        if len(images.shape)==4:

            if inference:
                n_batch = 1
                length = input_ids.shape[0]
                assert images_clip.shape[0] == 1
                images_clip_extend = images_clip.expand(length, -1, -1, -1).contiguous()

                output_hidden_states = []
                for i in range(n_batch):
                    start_i, end_i = i * length, min((i + 1) * length, input_ids.shape[0])
                    output_i = super().forward(
                        images=images_clip_extend[: end_i - start_i],
                        attention_mask=attention_masks[start_i:end_i],
                        input_ids=input_ids[start_i:end_i],
                        output_hidden_states=True,
                    )
                    output_hidden_states.append(output_i.hidden_states)
                    torch.cuda.empty_cache()

                output_hidden_states_list = []
                output_hidden_states_level = torch.cat(output_hidden_states, dim=0)
                output_hidden_states_list.append(output_hidden_states_level)
                output_hidden_states = output_hidden_states_list
                output = None

            
            else:

                images_clip_list = []
                for i in range(len(offset) - 1):
                    start_i, end_i = offset[i], offset[i + 1]
                    images_clip_i = (
                        images_clip[i]
                        .unsqueeze(0)
                        .expand(end_i - start_i, -1, -1, -1)
                        .contiguous()
                    )
                    images_clip_list.append(images_clip_i)
                images_clip = torch.cat(images_clip_list, dim=0)
                # images_clip torch.Size([3, 3, 224, 224])
                # images_clip_i torch.Size([offset[1], 3, 224, 224])

                output = super().forward(
                    images=images_clip, # images_clip torch.Size([3, 3, 224, 224])

                    attention_mask=attention_masks,#(3,62)
                    input_ids=input_ids,
                    labels=labels,#(3,62)
                    output_hidden_states=True,
                )
                output_hidden_states = output.hidden_states
            hidden_states = []

            assert len(self.model.text_hidden_fcs) == 1
            hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states[-1]))

            last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
            pred_embeddings = last_hidden_state[seg_token_mask]
            seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]

            seg_token_offset = seg_token_counts.cumsum(-1)
            seg_token_offset = torch.cat(
                [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
            )

            seg_token_offset = seg_token_offset[offset]

            pred_embeddings_ = []
            for i in range(len(seg_token_offset) - 1):
                start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
                pred_embeddings_.append(pred_embeddings[start_i:end_i])
            pred_embeddings = pred_embeddings_

            multimask_output = False
            pred_masks = []
            for i in range(len(pred_embeddings)):   #torch.Size([3, 256])
                (
                    sparse_embeddings, #torch.Size([3, 1, 256])
                    dense_embeddings,  #torch.Size([3, 256, 64, 64])
                ) = self.model.visual_model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                    text_embeds=pred_embeddings[i].unsqueeze(1),
                )
                sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
                low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                    image_embeddings=image_embeddings[i].unsqueeze(0),
                    image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )
                pred_mask = self.model.visual_model.postprocess_masks(
                    low_res_masks,
                    input_size=resize_list[i],
                    original_size=label_list[i].shape,
                )
                pred_masks.append(pred_mask[:, 0])

            model_output = output
            gt_masks = masks_list
            #gt_masks = expand_masks(gt_masks, seg_token_counts)
            gt_list = []
            if not inference:
                for gt_mask in gt_masks:
                    gt_list.append(self.expand_masks(gt_mask, seg_token_counts))
                gt_masks = torch.stack(gt_list, dim=0)

            #if not inference:
            #    gt_masks = self.expand_masks(gt_masks[0],gt_masks[1], seg_token_counts)
                #gt_masks = self.expand_masks(gt_masks[0], seg_token_counts)


            if inference and gt_masks is not None:
                return {
                    "pred_masks": pred_masks,
                    "gt_masks": gt_masks,
                }
            elif inference and gt_masks is None:
                return {
                    "pred_masks": pred_masks,
                }
            output = model_output.logits

            ce_loss = model_output.loss
            ce_loss = ce_loss * self.ce_loss_weight
            mask_bce_loss = 0
            mask_dice_loss = 0
            num_masks = 0
            for batch_idx in range(len(pred_masks)):
                gt_mask = gt_masks[batch_idx]
                #a= torch.sum(gt_mask[0])
                #b= torch.sum(gt_mask[1])
                #c= torch.sum(gt_mask[2])
                pred_mask = pred_masks[batch_idx]
                gt_mask_list = []
                '''if gt_mask.shape!= pred_mask.shape:
                    for i in range(gt_mask.shape[0]):
                        #if gt_mask[i].max()>0:
                        #    gt_mask_list.append(gt_mask[i])
                        gt_mask_list.append(gt_mask[i].unsqueeze(0).expand(seg_token_counts[i+2],-1,-1))
                    gt_mask = torch.stack(gt_mask_list, dim=0)'''
                    #import pdb;pdb.set_trace()


                assert (
                    gt_mask.shape[0] == pred_mask.shape[0]
                ), "gt_mask.shape: {}, pred_mask.shape: {}".format(
                    gt_mask.shape, pred_mask.shape
                )
                mask_bce_loss += (
                    sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                    * gt_mask.shape[0]
                )
                mask_dice_loss += (
                    dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                    * gt_mask.shape[0]
                )
                num_masks += gt_mask.shape[0]

            mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
            mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
            mask_loss = mask_bce_loss + mask_dice_loss

            loss = ce_loss + mask_loss

            return {
                "loss": loss,
                "ce_loss": ce_loss,
                "mask_bce_loss": mask_bce_loss,
                "mask_dice_loss": mask_dice_loss,
                "mask_loss": mask_loss,
            }

            
        else:
            if inference:
                n_batch = 1
                length = input_ids.shape[0]
                assert images_clip.shape[0] == 1
                #images_clip_extend = images_clip.expand(length, -1, -1, -1).contiguous()
                images_clip_extend = images_clip.squeeze(0).contiguous()

                output_hidden_states = []
                for i in range(n_batch):
                    start_i, end_i = i * length, min((i + 1) * length, input_ids.shape[0])
                    output_i = super().forward(
                        images=images_clip_extend[: end_i - start_i],
                        attention_mask=attention_masks[start_i:end_i],
                        input_ids=input_ids[start_i:end_i],
                        output_hidden_states=True,
                    )
                    output_hidden_states.append(output_i.hidden_states)
                    torch.cuda.empty_cache()

                output_hidden_states_list = []
                output_hidden_states_level = torch.cat(output_hidden_states, dim=0)
                output_hidden_states_list.append(output_hidden_states_level)
                output_hidden_states = output_hidden_states_list
                output = None
            else:
                #output_hidden_states_list = []

                #for j in range(images_clip.shape[0]):

                images_clip_list = []
                for i in range(len(offset) - 1):
                    start_i, end_i = offset[i], offset[i + 1]
                    images_clip_i = (
                        images_clip[0][i]
                        .unsqueeze(0)
                        .expand(end_i - start_i, -1, -1, -1)
                        .contiguous()
                    )
                    images_clip_list.append(images_clip_i)
                images_clip = torch.cat(images_clip_list, dim=0)
                # images_clip torch.Size([3, 3, 224, 224])
                # images_clip_i torch.Size([offset[1], 3, 224, 224])

                output = super().forward(
                    images=images_clip, # images_clip torch.Size([3, 3, 224, 224])

                    attention_mask=attention_masks,#(3,62)
                    input_ids=input_ids,
                    labels=labels,#(3,62)
                    output_hidden_states=True,
                )
                #output_hidden_states = output.hidden_states
                output_hidden_states = output.hidden_states
                
            pred_embeddings_list = []

            #output_hidden_states_list.append(output_hidden_states)
            hidden_states = []

            assert len(self.model.text_hidden_fcs) == 1
            hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states[-1]))

            last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
            pred_embeddings = last_hidden_state[seg_token_mask]
            seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]

            seg_token_offset = seg_token_counts.cumsum(-1)
            seg_token_offset = torch.cat(
                [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
            )

            seg_token_offset = seg_token_offset[offset]

            pred_embeddings_ = []
            
            for i in range(len(seg_token_offset) - 1):
                start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
                pred_embeddings_.append(pred_embeddings[start_i:end_i])
                pred_embeddings_list.append(pred_embeddings[start_i:end_i])
            pred_embeddings = pred_embeddings_
                #pred_embeddings_list.append(pred_embeddings)
            

            pred_embeddings_video = torch.stack(pred_embeddings_list,dim=0)
            pred_embeddings_video = pred_embeddings_video.permute(1,0,2)
            #value = torch.randn(10,8,256).cuda()
            video_feat = self.query_feat.weight.unsqueeze(1).expand(pred_embeddings_video.shape[0],-1,-1).repeat(1, pred_embeddings_video.shape[1], 1)
            video_feat = self.cross_attention(video_feat,pred_embeddings_video)
            video_feat = self.self_attention(video_feat)
            video_feat = self.ffn(video_feat)
            video_feat = video_feat.permute(1,0,2)
            pred_embeddings_video_list = []
            for i in range(video_feat.shape[0]):
                pred_embeddings_video_list.append(video_feat[i])
            #for pred_embeddings in pred_embeddings_video_list: 


            #pred_embeddings_video_list = pred_embeddings
            multimask_output = False
            pred_masks = []
            #label_list = label_list[0] + label_list[1]
            for i in range(len(pred_embeddings_video_list)):
                #image_embeddings_=image_embeddings[i],

                (
                    sparse_embeddings,
                    dense_embeddings,
                ) = self.model.visual_model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                    text_embeds=pred_embeddings_video_list[i].unsqueeze(1),
                )
                sparse_embeddings = sparse_embeddings.to(pred_embeddings_video_list[i].dtype)
                low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                    image_embeddings=image_embeddings[i].unsqueeze(0),
                    image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )
                pred_mask = self.model.visual_model.postprocess_masks(
                    low_res_masks,
                    input_size=resize_list[0],
                    original_size=label_list[0][i].shape,
                )
                pred_masks.append(pred_mask[:, 0])

            model_output = output
            gt_masks = masks_list
            '''masks_array = gt_masks[0].cpu().numpy()
            for i, mask_array in enumerate(masks_array):
                # 将数组转换为图像
                mask_image = Image.fromarray(mask_array.astype(np.uint8) * 255).convert('L')

                # 保存图像
                mask_image.save(f"mask_{i}.png")
'''
            #pred_masks = torch.stack(pred_masks)
            if inference:
                return {
                    "pred_masks": pred_masks,
                    "gt_masks": gt_masks,
                }
            


            output = model_output.logits

            ce_loss = model_output.loss
            ce_loss = ce_loss * self.ce_loss_weight
            mask_bce_loss = 0
            mask_dice_loss = 0
            num_masks = 0
            for batch_idx in range(len(pred_masks)):
                gt_mask = gt_masks[batch_idx]
                pred_mask = pred_masks[batch_idx]# = torch.stack(pred_masks)
                gt_mask_list = []
                '''if gt_mask.shape!= pred_mask.shape:
                    for i in range(len(gt_mask)):
                        if gt_mask[i].max()>0:
                            gt_mask_list.append(gt_mask[i])

                    gt_mask = torch.stack(gt_mask_list, dim=0)'''
                    #import pdb;pdb.set_trace()


                assert (
                    gt_mask.shape[0] == pred_mask.shape[0]
                ), "gt_mask.shape: {}, pred_mask.shape: {}".format(
                    gt_mask.shape, pred_mask.shape
                )
                mask_bce_loss += (
                    sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                    * gt_mask.shape[0]
                )
                mask_dice_loss += (
                    dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                    * gt_mask.shape[0]
                )
                num_masks += gt_mask.shape[0]

            mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
            mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
            mask_loss = mask_bce_loss + mask_dice_loss

            loss = ce_loss + mask_loss

            return {
                "loss": loss,
                "ce_loss": ce_loss,
                "mask_bce_loss": mask_bce_loss,
                "mask_dice_loss": mask_dice_loss,
                "mask_loss": mask_loss,
            }

    def evaluate(
        self,
        images_clip,
        images,
        input_ids,
        resize_list,
        original_size_list,
        max_new_tokens=32,
        tokenizer=None,
    ):
        with torch.no_grad():
            outputs = self.generate(
                images=images_clip,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            output_hidden_states = outputs.hidden_states[-1]
            output_ids = outputs.sequences

            seg_token_mask = output_ids[:, 1:] == self.seg_token_idx
            # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
            seg_token_mask = torch.cat(
                [
                    torch.zeros((seg_token_mask.shape[0], 255)).bool().cuda(),
                    seg_token_mask,
                ],
                dim=1,
            )

            hidden_states = []

            assert len(self.model.text_hidden_fcs) == 1
            hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states))

            last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
            pred_embeddings = last_hidden_state[seg_token_mask]

            seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]
            seg_token_offset = seg_token_counts.cumsum(-1)
            seg_token_offset = torch.cat(
                [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
            )

            pred_embeddings_ = []
            for i in range(len(seg_token_offset) - 1):
                start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
                pred_embeddings_.append(pred_embeddings[start_i:end_i])
            pred_embeddings = pred_embeddings_

            image_embeddings = self.get_visual_embs(images)

            multimask_output = False
            pred_masks = []
            for i in range(len(pred_embeddings)):
                (
                    sparse_embeddings,
                    dense_embeddings,
                ) = self.model.visual_model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                    text_embeds=pred_embeddings[i].unsqueeze(1),
                )

                sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
                low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                    image_embeddings=image_embeddings[i].unsqueeze(0),
                    image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )
                pred_mask = self.model.visual_model.postprocess_masks(
                    low_res_masks,
                    input_size=resize_list[i],
                    original_size=original_size_list[i],
                )
                pred_masks.append(pred_mask[:, 0])

        return output_ids, pred_masks
    def expand_masks(self, gt_masks1,expand_list):
        #gt_masks = torch.cat(gt_masks,dim=0)
        if len(gt_masks1)!= 0 :
            result_masks1 = []
            for i in range(gt_masks1.shape[0]):
                gt_masks1_ = gt_masks1[i]
                if expand_list[i] == 0:
                    continue
                else:
                    gt_masks1_ = gt_masks1_.unsqueeze(0).expand(expand_list[i],-1,-1)
                result_masks1.append(gt_masks1_)
            gt_masks1 = torch.cat(result_masks1,dim=0)
        '''if len(gt_masks2)!= 0 :
            result_masks2 = []
            start_indx = len(expand_list) - gt_masks2.shape[0]
            for j in range(gt_masks2.shape[0]):
                gt_masks2_ = gt_masks2[j]
                if expand_list[start_indx + j] == 0:
                    continue
                else:
                    gt_masks2_ = gt_masks2_.unsqueeze(0).expand(expand_list[start_indx + j],-1,-1)
                result_masks2.append(gt_masks2_)
            gt_masks2 = torch.cat(result_masks2,dim=0)

        return [gt_masks1, gt_masks2]'''
        return gt_masks1
def find_consecutive_sequence(tensor, sequence):
    sequence_len = len(sequence)
    
    # 使用 torch.where 查找所有匹配的起始位置
    start_indices = torch.where(tensor == sequence[0])
    
    # 遍历起始位置
    for start_row, start_col in zip(start_indices[0], start_indices[1]):
        # 检查当前起始位置是否存在相连序列
        if start_col + sequence_len <= tensor.shape[1] and torch.all(tensor[start_row, start_col:start_col+sequence_len] == sequence):
            return start_row.item(), start_col.item()
    
    return -1, -1

def mask_consecutive_sequence(mask,tensor, sequence):
    row, col = find_consecutive_sequence(tensor, sequence)
    if row != -1 and col != -1:
        # 创建一个与输入张量形状相同的布尔类型张量，所有元素初始化为False
        mask[row, col:col+len(sequence)] = True
        return mask
    else:
        return torch.zeros_like(tensor, dtype=torch.bool)
    

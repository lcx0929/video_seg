'''
Inference code for SgMg, on refer_youtube_vos
Modified from DETR (https://github.com/facebookresearch/detr)
'''
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

import utils.misc as utils
#from models import build_model
import torchvision.transforms as T
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image, ImageDraw
import math
import torch.nn.functional as F
import json

import opts
from tqdm import tqdm

import multiprocessing as mp
import threading
import warnings
warnings.filterwarnings("ignore")

from utils.misc import colormap
from torch.cuda.amp import autocast
import time
from model.LISA import LISAForCausalLM
import transformers
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         AverageMeter, ProgressMeter, Summary, dict_to_cuda,
                         intersectionAndUnionGPU)
from utils.utils import ANSWER_LIST, SHORT_QUESTION_LIST
from model.llava import conversation as conversation_lib
from transformers import CLIPImageProcessor
from model.segment_anything.utils.transforms import ResizeLongestSide
from model.llava.constants import (DEFAULT_IMAGE_TOKEN, IGNORE_INDEX,
                                   IMAGE_TOKEN_INDEX)
from model.llava.mm_utils import tokenizer_image_token

short_question_list = SHORT_QUESTION_LIST
answer_list = ANSWER_LIST
# colormap
color_list = colormap()
color_list = color_list.astype('uint8').tolist()
clip_image_processor = CLIPImageProcessor.from_pretrained('/root/autodl-tmp/clip-vit-large-patch14')
# build transform
transform = ResizeLongestSide(1024)
tokenizer = transformers.AutoTokenizer.from_pretrained(
    '/root/autodl-tmp/youtube_lisa_pretrain',
    cache_dir=None,
    model_max_length=512,
    padding_side="right",
    use_fast=False,
)
tokenizer.pad_token = tokenizer.unk_token
num_added_tokens = tokenizer.add_tokens("[SEG]")
def main(args):
    args.dataset_file = "ytvos"
    args.masks = True
    args.batch_size == 1
    args.eval = True

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    split = args.split
    # save path
    output_dir = args.output_dir
    save_path_prefix = os.path.join(output_dir, "Annotations")
    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix)

    save_visualize_path_prefix = os.path.join(output_dir, split + '_images')
    '''if args.visualize:
        if not os.path.exists(save_visualize_path_prefix):
            os.makedirs(save_visualize_path_prefix)'''

    # load data
    root = Path(args.ytvos_path) # data/refer_youtube_vos
    img_folder = os.path.join(root, split, "JPEGImages")
    meta_file = os.path.join(root, "meta_expressions", split, "meta_expressions.json")
    with open(meta_file, "r") as f:
        data = json.load(f)["videos"]
    valid_test_videos = set(data.keys())
    # for some reasons the competition's validation expressions dict contains both the validation (202) &
    # test videos (305). so we simply load the test expressions dict and use it to filter out the test videos from
    # the validation expressions dict:
    test_meta_file = os.path.join(root, "meta_expressions", "test", "meta_expressions.json")
    with open(test_meta_file, 'r') as f:
        test_data = json.load(f)['videos']
    test_videos = set(test_data.keys())
    valid_videos = valid_test_videos - test_videos
    video_list = sorted([video for video in valid_videos])
    assert len(video_list) == 202, 'error: incorrect number of validation videos'

    # create subprocess
    thread_num = 1
    global result_dict
    result_dict = mp.Manager().dict()

    processes = []
    lock = threading.Lock()

    video_num = len(video_list)
    print("Total video num is {}.".format(video_num))
    per_thread_video_num = video_num // thread_num

    start_time = time.time()
    print('Start inference')
    for i in range(thread_num):
        if i == thread_num - 1:
            sub_video_list = video_list[i * per_thread_video_num:]
        else:
            sub_video_list = video_list[i * per_thread_video_num: (i + 1) * per_thread_video_num]
        p = mp.Process(target=sub_processor, args=(lock, i, args, data,
                                                   save_path_prefix, save_visualize_path_prefix,
                                                   img_folder, sub_video_list))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    end_time = time.time()
    total_time = end_time - start_time

    result_dict = dict(result_dict)
    num_all_frames_gpus = 0
    for pid, num_all_frames in result_dict.items():
        num_all_frames_gpus += num_all_frames

    print("Total inference time: %.4f s" %(total_time))

def sub_processor(lock, pid, args, data, save_path_prefix, save_visualize_path_prefix, img_folder, video_list):
    text = 'processor %d' % pid
    with lock:
        progress = tqdm(
            total=len(video_list),
            position=pid,
            desc=text,
            ncols=0
        )
    torch.cuda.set_device(pid)

    # model
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    #num_added_tokens = tokenizer.add_tokens("[SEG1]")
    #num_added_tokens = tokenizer.add_tokens("[SEG2]")
    #num_added_tokens = tokenizer.add_tokens("[SEG3]")
    #args.seg_token_idx  = [] 
    #args.seg_token_idx.append(tokenizer("[SEG]", add_special_tokens=False).input_ids[0])
    #args.seg_token_idx.append(tokenizer("[SEG1]", add_special_tokens=False).input_ids[0])
    #args.seg_token_idx.append(tokenizer("[SEG2]", add_special_tokens=False).input_ids[0])
    #args.seg_token_idx.append(tokenizer("[SEG3]", add_special_tokens=False).input_ids[0])
    args.seg_token_idx  = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    if args.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )
    model_args = {
        "train_mask_decoder": args.train_mask_decoder,
        "out_dim": args.out_dim,
        "ce_loss_weight": args.ce_loss_weight,
        "dice_loss_weight": args.dice_loss_weight,
        "bce_loss_weight": args.bce_loss_weight,
        "seg_token_idx": args.seg_token_idx,
        "vision_pretrained": args.vision_pretrained,
        "vision_tower": args.vision_tower,
        "use_mm_start_end": args.use_mm_start_end,
    }
    model = LISAForCausalLM.from_pretrained(
    args.version, torch_dtype=torch_dtype, tokenizer=tokenizer,low_cpu_mem_usage=True, **model_args
)
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=args.local_rank)
    device = args.device
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if pid == 0:
        print('number of params:', n_parameters)

    '''if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        if 'args' in checkpoint:
            print("Loaded Checkpoint args: {}".format(checkpoint['args']))
        del checkpoint
    else:
        raise ValueError('Please specify the checkpoint for inference.')'''


    num_all_frames = 0
    model.eval()
    for idx_, video in enumerate(video_list):
        # if idx_ < 47:
        #     continue
        torch.cuda.empty_cache()
        metas = []
        expressions = data[video]["expressions"]
        expression_list = list(expressions.keys())
        num_expressions = len(expression_list)
        video_len = len(data[video]["frames"])

        # read all the anno meta
        for i in range(num_expressions):
            meta = {}
            meta["video"] = video
            meta["exp"] = expressions[expression_list[i]]["exp"]
            meta["exp_id"] = expression_list[i]
            meta["frames"] = data[video]["frames"]
            metas.append(meta)
        meta = metas

        # store images
        frames = data[video]["frames"]
        video_name = video
        imgs = []
        image_clips = []
        labels_list = []
        resize_list = []
        for t in range(video_len):
            frame = frames[t]
            img_path = os.path.join(img_folder, video_name, frame + ".jpg")
            img = Image.open(img_path).convert('RGB')
            origin_w, origin_h = img.size

            img = np.array(img)
            image_clip = clip_image_processor.preprocess(
                img, return_tensors="pt"
            )["pixel_values"][0]
            img = transform.apply_image(img)
            resize = img.shape[:2]
            img = preprocess(torch.from_numpy(img).permute(2, 0, 1).contiguous())

            imgs.append(img) 
            image_clips.append(image_clip)
            labels_list.append(np.zeros([720,1280], dtype=np.uint8))
            resize_list.append(resize)
            #imgs.append(transform(img))  # list[img]
        imgs = torch.stack(imgs, dim=0).to(args.device).bfloat16().unsqueeze(0)  # [video_len, 3, h, w]
        image_clips = torch.stack(image_clips, dim=0).to(args.device).bfloat16().unsqueeze(0)  # [video_len, 3, h, w]
        img_h, img_w = imgs.shape[-2:]
        size = torch.as_tensor([int(img_h), int(img_w)]).to(args.device)
        target = {"size": size}

        for i in range(num_expressions):
            input_ids,labels,attention_masks, conversations, questions, offset_list =  prepare_data(meta,video_len,expression_list,expressions,i)
            masks_list = None
            inference = True
            # 2. For each expression

            video_name = meta[i]["video"]
            exp = meta[i]["exp"]
            exp_id = meta[i]["exp_id"]
            frames = meta[i]["frames"] 
            sliced_tensors = []

            total_samples = video_len

            # 每次划分8组，直到所有样本都被划分完毕
            pred_masks_list = []
            for i in range(0, total_samples, 8):
                # 如果剩余样本不足8组，则取剩余的所有样本
                if i + 8 > total_samples:
                    input_ids_ = input_ids[-8:, :]
                    labels_ = labels[-8:, :]
                    attention_masks_ = attention_masks[-8:, :]
                    image_clips_ = image_clips[:,-8:, :,:,:]
                    imgs_ = imgs[:,-8:, :,:,:]
                    labels_list_ = labels_list[-8:]
                    resize_list_ = resize_list[-8:]
                else:
                    input_ids_ = input_ids[i:i+8, :]
                    labels_ = labels[i:i+8, :]
                    attention_masks_ = attention_masks[i:i+8, :]
                    image_clips_ = image_clips[:,i:i+8, :, :,:]
                    imgs_ = imgs[:,i:i+8, :,:,:]
                    labels_list_ = labels_list[i:i+8]
                    resize_list_ = resize_list[i:i+8]
                #sliced_tensors.append(sliced_tensor)
                input_dict = {
                        "images": imgs_,
                        "images_clip": image_clips_,
                        "input_ids": input_ids_.to(args.device),
                        "labels": labels_.to(args.device),
                        "attention_masks": attention_masks_.to(args.device),
                        "masks_list": masks_list,
                        "label_list": [labels_list_],
                        "resize_list": resize_list_,
                        "offset": torch.LongTensor(offset_list),
                        #"questions_list": questions_list,
                        #"sampled_classes_list": sampled_classes_list,
                        "inference": inference,
                        #"conversation_list": conversation_list,
                        #"object_ids": object_ids
                        }
            # 打印划分后的张量列表
                video_len = len(frames)
                with torch.no_grad():
                    with autocast(args.amp):

                        #outputs = model(imgs, image_clips, input_ids,labels, attention_masks,offset_list,masks_list,labels_list, resize_list, inference=True)
                        outputs = model(**input_dict)
                        pred_masks_list.append(outputs["pred_masks"][0])
            s = video_len % 8

            if s == 0:
                pass
            else:
                pred_masks_list[-1] = pred_masks_list[-1][-s:]
            pred_masks = torch.cat(pred_masks_list, dim=0)
            '''pred_logits = outputs["pred_logits"][0]
            pred_masks = outputs["pred_masks"][0]

            # according to pred_logits, select the query index
            pred_scores = pred_logits.sigmoid() # [t, q, k]
            pred_scores = pred_scores.mean(0)   # [q, k]
            max_scores, _ = pred_scores.max(-1) # [q,]
            _, max_ind = max_scores.max(-1)     # [1,]
            max_inds = max_ind.repeat(video_len)
            pred_masks = pred_masks[range(video_len), max_inds, ...] # [t, h, w]
            pred_masks = pred_masks.unsqueeze(0)'''
            #pred_masks = outputs["pred_masks"][0]
            # unpad
            #pred_masks = pred_masks[:, :, :img_h, :img_w]
            #pred_masks = F.interpolate(pred_masks, size=(origin_h, origin_w), mode='bilinear', align_corners=False)
            pred_masks = (pred_masks.sigmoid() > args.threshold).squeeze(0).detach().cpu().numpy()   # 0.5

            # save binary image
            save_path = os.path.join(save_path_prefix, video_name, exp_id)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            for j in range(video_len):
                frame_name = frames[j]
                mask = pred_masks[j].astype(np.float32)
                mask = Image.fromarray(mask * 255).convert('L')
                save_file = os.path.join(save_path, frame_name + ".png")
                mask.save(save_file)

        with lock:
            progress.update(1)

    result_dict[str(pid)] = num_all_frames
    with lock:
        progress.close()


# visuaize functions
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b.cpu() * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


# Visualization functions
def draw_reference_points(draw, reference_points, img_size, color):
    W, H = img_size
    for i, ref_point in enumerate(reference_points):
        init_x, init_y = ref_point
        x, y = W * init_x, H * init_y
        cur_color = color
        draw.line((x-10, y, x+10, y), tuple(cur_color), width=4)
        draw.line((x, y-10, x, y+10), tuple(cur_color), width=4)

def draw_sample_points(draw, sample_points, img_size, color_list):
    alpha = 255
    for i, samples in enumerate(sample_points):
        for sample in samples:
            x, y = sample
            cur_color = color_list[i % len(color_list)][::-1]
            cur_color += [alpha]
            draw.ellipse((x-2, y-2, x+2, y+2), 
                            fill=tuple(cur_color), outline=tuple(cur_color), width=1)

def vis_add_mask(img, mask, color):
    origin_img = np.asarray(img.convert('RGB')).copy()
    color = np.array(color)

    mask = mask.reshape(mask.shape[0], mask.shape[1]).astype('uint8') # np
    mask = mask > 0.5

    origin_img[mask] = origin_img[mask] * 0.5 + color * 0.5
    origin_img = Image.fromarray(origin_img)
    return origin_img


def prepare_data(meta,video_len,expression_list,expressions, idx):
    questions = []
    answers = []
    class_ids = []
    offset_list=[0]
    num = len(meta)
    cnt = 0
    
    for i in range(video_len):
        sampled_cls = expression_list[idx]
        text = expressions[sampled_cls]["exp"]
        #text = meta[i]["exp"]
        text = text.strip()

        assert len(text.split("||")) == 1
        question_template = random.choice(short_question_list)
        questions.append(question_template.format(class_name=text.lower()))

        answers.append(random.choice(answer_list))

        #if ds in ["paco_lvis", "pascal_part"]:
        #    continue

        #class_id = self.data2classes[ds].tolist().index(sampled_cls)
        #class_ids.append(class_id)

    conversations = []
    conv = conversation_lib.conv_llava_v1.copy()
    i = 0
    while i < len(questions):
        conv.messages = []
        conv.append_message(conv.roles[0], questions[i])
        conv.append_message(conv.roles[1], answers[i])
        conversations.append(conv.get_prompt())
        i += 1
    conversation_list = conversations
    for i in range(len(conversation_list)):
        replace_token = DEFAULT_IMAGE_TOKEN
        replace_token = (
            DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
        )
        conversation_list[i] = conversation_list[i].replace(
            DEFAULT_IMAGE_TOKEN, replace_token
        )
    input_ids = [
        tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        for prompt in conversation_list
    ]
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_masks = input_ids.ne(tokenizer.pad_token_id)
    conv = conversation_lib.conv_llava_v1.copy()
    targets = input_ids.clone()
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversation_list, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            # if len(parts) != 2:
            #     break
            assert len(parts) == 2, (len(parts), rou)
            parts[0] += sep

            if DEFAULT_IMAGE_TOKEN in conversation:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX
        if cur_len < tokenizer.model_max_length:
            assert cur_len == total_len
    cnt += len(conversations)
    offset_list.append(8)
    return input_ids,targets,attention_masks, conversations, questions, offset_list
def preprocess(x: torch.Tensor) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors

    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    x = (x - pixel_mean) / pixel_std

    # Pad
    h, w = x.shape[-2:]
    padh = 1024 - h
    padw = 1024 - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SgMg inference script', parents=[opts.get_args_parser()])
    args = parser.parse_args()
    main(args)
    print("Save results at: {}.".format(os.path.join(args.output_dir, "Annotations")))

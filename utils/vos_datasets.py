"""
Ref-YoutubeVOS data loader
"""
from pathlib import Path

import torch
from torch.autograd.grad_mode import F
import torch.nn.functional as Fa

from torch.utils.data import Dataset
#import datasets.transforms_video as T
import sys
import os
from PIL import Image
import json
import numpy as np
import random
from transformers import CLIPImageProcessor
sys.path.append('/root/LISA')

from model.segment_anything.utils.transforms import ResizeLongestSide
from .utils import ANSWER_LIST, SHORT_QUESTION_LIST
from model.llava import conversation as conversation_lib

from .categories import ytvos_category_dict as category_dict
args = None
import cv2
import matplotlib.pyplot as plt


class YTVOSDataset(Dataset):
    """
    A dataset class for the Refer-Youtube-VOS dataset which was first introduced in the paper:
    "URVOS: Unified Referring Video Object Segmentation Network with a Large-Scale Benchmark"
    (see https://link.springer.com/content/pdf/10.1007/978-3-030-58555-6_13.pdf).
    The original release of the dataset contained both 'first-frame' and 'full-video' expressions. However, the first
    dataset is not publicly available anymore as now only the harder 'full-video' subset is available to download
    through the Youtube-VOS referring video object segmentation competition page at:
    https://competitions.codalab.org/competitions/29139
    Furthermore, for the competition the subset's original validation set, which consists of 507 videos, was split into
    two competition 'validation' & 'test' subsets, consisting of 202 and 305 videos respectively. Evaluation can
    currently only be done on the competition 'validation' subset using the competition's server, as
    annotations were publicly released only for the 'train' subset of the competition.

    """

    ignore_label = 255

    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    def __init__(self,  base_image_dir, vision_tower,image_size, train=True,
                 num_frames=8, max_skip=3):
        self.train = train
        if train:
            img_folder = os.path.join(base_image_dir, 'ytvos/train')
            ann_file = os.path.join(base_image_dir, 'ytvos/train/meta_expressions.json')
        else:
            img_folder = os.path.join(base_image_dir, 'ytvos/valid')
            ann_file = os.path.join(base_image_dir, 'ytvos/valid/meta_expressions.json')
        self.img_folder = img_folder     
        self.ann_file = ann_file
        if 'train' in str(self.img_folder):
            self.mode = "train"
        elif 'valid' in str(self.img_folder):
            self.mode = 'valid'
        else:
            raise NotImplementedError

        #self._transforms = transforms
        #self.return_masks = return_masks
        self.num_frames = num_frames     
        self.max_skip = max_skip
        
        # create video meta data
        if train:
            self.prepare_metas()   
        else:
            self.prepare_metas_test()     
        self.is_noun = lambda pos: pos[:2] == "NN"
        print('\n video num: ', len(self.videos), ' clip num: ', len(self.metas))  
        print('\n')    

        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
        
        self.transform = ResizeLongestSide(image_size)
        self.img_size = image_size
        self.short_question_list = SHORT_QUESTION_LIST
        self.answer_list = ANSWER_LIST

        self.num_classes_per_sample = 3



    def prepare_metas_test(self):
        # read object information
        with open(os.path.join(str(self.img_folder), 'meta.json'), 'r') as f:
            subset_metas_by_video = json.load(f)['videos']
        
        # read expression data
        with open(str(self.ann_file), 'r') as f:
            subset_expressions_by_video = json.load(f)['videos']
        self.videos = list(subset_expressions_by_video.keys())

        self.metas = []
        # for each video
        for vid in self.videos:
            vid_meta = subset_metas_by_video[vid]
            vid_data = subset_expressions_by_video[vid]
            vid_frames = sorted(vid_data['frames'])
            vid_len = len(vid_frames)
            # for each expression
            for exp_id, exp_dict in vid_data['expressions'].items():
                exp = exp_dict['exp']
                #oid = int(exp_dict['obj_id'])
                oid = int(int(exp_id) / 2 + 1)
                # for each frame
                for frame_id in range(0, vid_len, self.num_frames):
                    meta = {}
                    meta['video'] = vid
                    meta['exp'] = exp
                    meta['obj_id'] = oid
                    meta['frames'] = vid_frames
                    meta['frame_id'] = frame_id
                    obj_id = oid
                    #meta['category'] = vid_meta['objects'][obj_id]['category']
                    self.metas.append(meta)

                    

    def prepare_metas(self):
        # read object information
        with open(os.path.join(str(self.img_folder), 'meta.json'), 'r') as f:
            subset_metas_by_video = json.load(f)['videos']
        
        # read expression data
        with open(str(self.ann_file), 'r') as f:
            subset_expressions_by_video = json.load(f)['videos']
        self.videos = list(subset_expressions_by_video.keys())

        self.metas = []
        # for each video
        for vid in self.videos:
            vid_meta = subset_metas_by_video[vid]
            vid_data = subset_expressions_by_video[vid]
            vid_frames = sorted(vid_data['frames'])
            vid_len = len(vid_frames)
            # for each expression
            for exp_id, exp_dict in vid_data['expressions'].items():
                exp = exp_dict['exp']
                oid = int(exp_dict['obj_id'])
                # for each frame
                for frame_id in range(0, vid_len, self.num_frames):
                    meta = {}
                    meta['video'] = vid
                    meta['exp'] = exp
                    meta['obj_id'] = oid
                    meta['frames'] = vid_frames
                    meta['frame_id'] = frame_id
                    obj_id = exp_dict['obj_id']
                    meta['category'] = vid_meta['objects'][obj_id]['category']
                    self.metas.append(meta)

    @staticmethod
    def bounding_box(img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax # y1, y2, x1, x2 
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = Fa.pad(x, (0, padw, 0, padh))
        return x


    def __len__(self):
        return len(self.metas)
        
    def __getitem__(self, idx):
        if self.train:
            idx = random.randint(0, len(self.metas) - 1)
            meta = self.metas[idx]  # dict
            video, exp, obj_id, category, frames, frame_id = \
                        meta['video'], meta['exp'], meta['obj_id'], meta['category'], meta['frames'], meta['frame_id']
            exp = " ".join(exp.lower().split())
            category_id = category_dict[category]
            vid_len = len(frames)


            num_frames = self.num_frames
            sample_indx = [frame_id]

            if self.num_frames != 1:
                # local sample [before and after].
                sample_id_before = random.randint(1, 3)
                sample_id_after = random.randint(1, 3)
                local_indx = [max(0, frame_id - sample_id_before), min(vid_len - 1, frame_id + sample_id_after)]
                sample_indx.extend(local_indx)

                # global sampling [in rest frames]
                if num_frames > 3:
                    all_inds = list(range(vid_len))
                    global_inds = all_inds[:min(sample_indx)] + all_inds[max(sample_indx):]
                    global_n = num_frames - len(sample_indx)
                    if len(global_inds) > global_n:
                        select_id = random.sample(range(len(global_inds)), global_n)
                        for s_id in select_id:
                            sample_indx.append(global_inds[s_id])
                    elif vid_len >= global_n:
                        select_id = random.sample(range(vid_len), global_n)
                        for s_id in select_id:
                            sample_indx.append(all_inds[s_id])
                    else:
                        select_id = random.sample(range(vid_len), global_n - vid_len) + list(range(vid_len))
                        for s_id in select_id:
                            sample_indx.append(all_inds[s_id])
            sample_indx.sort()
            # random reverse
            if self.mode == "train" and np.random.rand() < 0.3:
                sample_indx = sample_indx[::-1]

                
            imgs, labels, boxes, masks, valid = [], [], [], [], []
            image_paths = []
            image_clips = []
            resizes = []
            for j in range(self.num_frames):
                frame_indx = sample_indx[j]
                frame_name = frames[frame_indx]
                img_path = os.path.join(str(self.img_folder), 'JPEGImages', video, frame_name + '.jpg')
                mask_path = os.path.join(str(self.img_folder), 'Annotations', video, frame_name + '.png')
                image_paths.append(img_path)
                img = cv2.imread(img_path)
                image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                image_clip = self.clip_image_processor.preprocess(
                    image, return_tensors="pt"
                )["pixel_values"][0]
                image = self.transform.apply_image(image)  # preprocess image for sam

                resize = image.shape[:2]
                image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

                #img = Image.open(img_path).convert('RGB')
                #mask = cv2.imread(mask_path)
                #mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

                label = np.uint8(Image.open(mask_path).convert('P'))
                label = torch.from_numpy(label)
                mask = (label==obj_id) # 0,1 binary 
                mask_img = Image.fromarray(mask.numpy())
                #mask_img.save("tensor_image.png")

                s = torch.sum(mask)
                #obj_ids = list(np.unique(label))
                #print(np.unique(label))
                #label_ = Image.open(mask_path)
                #label_.save('label.jpg')
                #mask = self.transform.apply_image(mask)
                # create the target
                category_id = torch.tensor(category_id)
                #label = np.array(label)
                #label = torch.from_numpy(label)
                #frame_masks = []
                #for obj_id in obj_ids[1:]:
                #    frame_masks.append(label == obj_id)
                #frame_masks = torch.stack(frame_masks, axis=0)
                #a = torch.sum(frame_masks, dim=0)
                #b = torch.sum(frame_masks)
                #if np.any(masks[0]):
                #    print(1)
                #mask = (mask==obj_id).astype(np.float32)
                '''if (mask > 0).any():
                    y1, y2, x1, x2 = self.bounding_box(mask)
                    box = torch.tensor([x1, y1, x2, y2]).to(torch.float)
                    valid.append(1)
                else: # some frame didn't contain the instance
                    box = torch.tensor([0, 0, 0, 0]).to(torch.float) 
                    valid.append(0)'''
                #mask = torch.from_numpy(mask)

                # append
                imgs.append(image)
                labels.append(label)
                masks.append(mask)
                #boxes.append(box)
                image_clips.append(image_clip)
                resizes.append(resize)
            #masks = torch.stack(masks, dim=0)
            #imgs= torch.stack(imgs, dim=0)
            #image_clips = torch.stack(image_clips, dim=0)
            #for sampled_cls in sampled_classes:

            questions = []
            answers = []
            class_ids = []

            text = exp
            for i in range(len(masks)):
                text = text.strip()

                assert len(text.split("||")) == 1
                question_template = random.choice(self.short_question_list)
                questions.append(question_template.format(class_name=text.lower()))

                answers.append(random.choice(self.answer_list))

                #if ds in ["paco_lvis", "pascal_part"]:
                #    continue

                #class_id = self.data2classes[ds].tolist().index(sampled_cls)
                #class_ids.append(class_id)

            conversations = []
            conv = conversation_lib.default_conversation.copy()
            i = 0
            while i < len(questions):
                conv.messages = []
                conv.append_message(conv.roles[0], questions[i])
                conv.append_message(conv.roles[1], answers[i])
                conversations.append(conv.get_prompt())
                i += 1
            sampled_classes = category
            #masks = torch.stack(masks,axis=0),
            #image_clips = torch.stack(image_clips,dim=0)
            #imgs = torch.stack(imgs,dim=0)
            #label = torch.ones(mask.shape[1], mask.shape[2]) * self.ignore_label
            #a = torch.sum(masks,dim=0)
            return (
                image_paths,
                torch.stack(imgs, dim=0),
                torch.stack(image_clips,dim=0),
                conversations,
                torch.stack(masks,dim=0),
                labels,
                resizes[0],
                questions,
                sampled_classes
                )
            
        else:
            meta = self.metas[idx]  # dict
            video, exp, obj_id, frames, frame_id = \
                        meta['video'], meta['exp'], meta['obj_id'], meta['frames'], meta['frame_id']
            exp = " ".join(exp.lower().split())
            #category_id = category_dict[category]
            vid_len = len(frames)


            num_frames = self.num_frames
            sample_indx = [frame_id]

            if self.num_frames != 1:
                # local sample [before and after].
                sample_id_before = random.randint(1, 3)
                sample_id_after = random.randint(1, 3)
                local_indx = [max(0, frame_id - sample_id_before), min(vid_len - 1, frame_id + sample_id_after)]
                sample_indx.extend(local_indx)

                # global sampling [in rest frames]
                if num_frames > 3:
                    all_inds = list(range(vid_len))
                    global_inds = all_inds[:min(sample_indx)] + all_inds[max(sample_indx):]
                    global_n = num_frames - len(sample_indx)
                    if len(global_inds) > global_n:
                        select_id = random.sample(range(len(global_inds)), global_n)
                        for s_id in select_id:
                            sample_indx.append(global_inds[s_id])
                    elif vid_len >= global_n:
                        select_id = random.sample(range(vid_len), global_n)
                        for s_id in select_id:
                            sample_indx.append(all_inds[s_id])
                    else:
                        select_id = random.sample(range(vid_len), global_n - vid_len) + list(range(vid_len))
                        for s_id in select_id:
                            sample_indx.append(all_inds[s_id])
            sample_indx.sort()
            # random reverse
            if self.mode == "train" and np.random.rand() < 0.3:
                sample_indx = sample_indx[::-1]

                
            imgs, labels, boxes, masks, valid = [], [], [], [], []
            image_paths = []
            image_clips = []
            resizes = []
            for j in range(self.num_frames):
                frame_indx = sample_indx[j]
                frame_name = frames[frame_indx]
                img_path = os.path.join(str(self.img_folder), 'JPEGImages', video, frame_name + '.jpg')
                mask_path = os.path.join(str(self.img_folder), 'Annotations', video, frame_name + '.png')
                image_paths.append(img_path)
                img = cv2.imread(img_path)
                image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                image_clip = self.clip_image_processor.preprocess(
                    image, return_tensors="pt"
                )["pixel_values"][0]
                image = self.transform.apply_image(image)  # preprocess image for sam

                resize = image.shape[:2]
                image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

                #img = Image.open(img_path).convert('RGB')
                #mask = cv2.imread(mask_path)
                #mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
                if os.path.exists(mask_path):
                    label = np.uint8(Image.open(mask_path).convert('P'))
                else:
                    label = np.zeros([720,1280], dtype=np.uint8)

                    #imgs.append(imgs[-1])
                    #labels.append(labels[-1])
                    #boxes.append(box)
                    #image_clips.append(image_clips[-1])
                    #resizes.append(resize)
                    #continue
                #print(np.unique(label))

                label = torch.from_numpy(label)
                mask = (label==obj_id) # 0,1 binary
                s = torch.sum(mask)
                #obj_ids = list(np.unique(label))
                #label_ = Image.open(mask_path)
                #label_.save('label.jpg')
                #mask = self.transform.apply_image(mask)
                # create the target
                #category_id = torch.tensor(category_id)
                #label = np.array(label)
                #label = torch.from_numpy(label)
                #frame_masks = []
                #for obj_id in obj_ids[1:]:
                #    frame_masks.append(label == obj_id)
                #frame_masks = torch.stack(frame_masks, axis=0)
                #a = torch.sum(frame_masks, dim=0)
                #b = torch.sum(frame_masks)
                #if np.any(masks[0]):
                #    print(1)
                #mask = (mask==obj_id).astype(np.float32)
                '''if (mask > 0).any():
                    y1, y2, x1, x2 = self.bounding_box(mask)
                    box = torch.tensor([x1, y1, x2, y2]).to(torch.float)
                    valid.append(1)
                else: # some frame didn't contain the instance
                    box = torch.tensor([0, 0, 0, 0]).to(torch.float) 
                    valid.append(0)'''
                #mask = torch.from_numpy(mask)

                # append
                imgs.append(image)
                labels.append(label)
                masks.append(mask)
                #boxes.append(box)
                image_clips.append(image_clip)
                resizes.append(resize)
            #masks = torch.stack(masks, dim=0)
            #imgs= torch.stack(imgs, dim=0)
            #image_clips = torch.stack(image_clips, dim=0)
            #for sampled_cls in sampled_classes:

            questions = []
            answers = []
            class_ids = []

            text = exp
            for i in range(len(masks)):
                text = text.strip()

                assert len(text.split("||")) == 1
                question_template = random.choice(self.short_question_list)
                questions.append(question_template.format(class_name=text.lower()))

                answers.append(random.choice(self.answer_list))

                #if ds in ["paco_lvis", "pascal_part"]:
                #    continue

                #class_id = self.data2classes[ds].tolist().index(sampled_cls)
                #class_ids.append(class_id)

            conversations = []
            conv = conversation_lib.default_conversation.copy()
            i = 0
            while i < len(questions):
                conv.messages = []
                conv.append_message(conv.roles[0], questions[i])
                conv.append_message(conv.roles[1], answers[i])
                conversations.append(conv.get_prompt())
                i += 1
            #sampled_classes = category
            #masks = torch.stack(masks,axis=0),
            #image_clips = torch.stack(image_clips,dim=0)
            #imgs = torch.stack(imgs,dim=0)
            #label = torch.ones(mask.shape[1], mask.shape[2]) * self.ignore_label
            #a = torch.sum(masks,dim=0)
            inference = True

            return (
                image_paths,
                torch.stack(imgs, dim=0),
                torch.stack(image_clips,dim=0),
                conversations,
                torch.stack(masks,dim=0),
                labels,
                resizes[0],
                None,
                None,
                inference
                )
            



def make_coco_transforms(current_epoch, image_set, max_size=640):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [288, 320, 352, 392, 416, 448, 480, 512]

    # CLIP at first to save time
    if image_set == 'train':
        return T.Compose([
            T.RandomSelect(
                T.Compose([
                    T.RandomResize(scales, max_size=max_size),
                    T.Check(),
                ]),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=max_size),
                    T.Check(),
                ])
            ),
            T.RandomHorizontalFlip(),
            T.PhotometricDistort(),
            normalize,
        ])
    
    # we do not use the 'val' set since the annotations are inaccessible
    if image_set == 'val':
        return T.Compose([
            T.RandomResize([360], max_size=640),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.ytvos_path)
    assert root.exists(), f'provided YTVOS path {root} does not exist'
    PATHS = {
        "train": (root / "train", root / "meta_expressions" / "train" / "meta_expressions.json"),
        "val": (root / "valid", root / "meta_expressions" / "val" / "meta_expressions.json"),    # not used actually
    }
    img_folder, ann_file = PATHS[image_set]
    dataset = YTVOSDataset(args, img_folder, ann_file, transforms=make_coco_transforms(args.current_epoch, image_set, max_size=args.max_size), return_masks=args.masks,
                           num_frames=args.num_frames, max_skip=args.max_skip)
    return dataset


'''root =  '/root/autodl-tmp/dataset/youtube'   
PATHS = {
    "train": (os.path.join(root , "train")),
    "val": (os.path.join(root , "valid")),
}

ytvos_dataset = YTVOSDataset(args, img_folder = '/root/autodl-tmp/dataset/youtube/train', 
                            ann_file='/root/autodl-tmp/dataset/youtube/train/meta_expressions.json',
                            transforms = make_coco_transforms,
                            return_masks=True,num_frames=3,max_skip=1)'''
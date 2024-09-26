# LISA
## Environment

instal the environment follow the LISA https://github.com/dvlab-research/LISA
'''pip install -r requirements.txt''' 

## Dataset

collect the dataset follow LISA https://github.com/dvlab-research/LISA

for the video segmentation, we ues the dataset ytvos https://github.com/skynbe/Refer-Youtube-VOS.

Download them from the above links, and organize them as follows.

├── dataset
│   ├── ade20k
│   │   ├── annotations
│   │   └── images
│   ├── coco
│   │   └── train2017
│   │       ├── 000000000009.jpg
│   │       └── ...
│   ├── cocostuff
│   │   └── train2017
│   │       ├── 000000000009.png
│   │       └── ...
│   ├── llava_dataset
│   │   └── llava_instruct_150k.json
│   ├── mapillary
│   │   ├── config_v2.0.json
│   │   ├── testing
│   │   ├── training
│   │   └── validation
│   ├── reason_seg
│   │   └── ReasonSeg
│   │       ├── train
│   │       ├── val
│   │       └── explanatory
│   ├── refer_seg
│   │   ├── images
│   │   |   ├── saiapr_tc-12 
│   │   |   └── mscoco
│   │   |       └── images
│   │   |           └── train2014
│   │   ├── refclef
│   │   ├── refcoco
│   │   ├── refcoco+
│   │   └── refcocog
│   └── vlpart
│       ├── paco
│       │   └── annotations
│       └── pascal_part
│           ├── train.json
│           └── VOCdevkit
|   |—— ytvos
|   |   |——meta_expressions
|   |   |——train
|   |   |   |——Annotations
|   |   |   |——JPEGImages
|   |   |   |——meta_expressions.json
|   |   |   |——meta.json
|   |   |——valid
|   |   |   |——Annotations
|   |   |   |——JPEGImages
|   |   |   |——meta_expressions.json
|   |   |   |——meta.json

## Model preparation
In order to run the code, you need to prepare the following models:

LLaVA-Lightning-7B-v1-1 and liuhaotian/LLaVA-Lightning-7B-delta-v1-1 for 7B model running

LLaVA-13B-v1-1 and liuhaotian/LLaVA-13B-delta-v1-1 for 13B model running

the model above can be found in the huggingface model hub. https://huggingface.curated.co/

after downloading the model, To train LISA-7B or 13B, you need to follow the instruction to merge the LLaVA delta weights. The instruction can be found in this link: https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md

sam_vit_h.pth for the vision encoder can be download in this link:  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth


## Prepare to run


run the code

you can run the code with the following command:

,,,
sh run.sh
,,,

the detail of the command is as follows:

//run for oringinal image dataset
'''
deepspeed --master_port=24999 train_ds.py \
  --version="/media/lcx/sia-vision/lisa/LISA/model/llava-v1.5-7b" \
  --dataset_dir='/media/lcx/sia-vision/lisa/LISA/dataset' \
  --vision_pretrained="/media/lcx/sia-vision/lisa/LISA/model/sam_vit_h.pth" \
  --dataset="sem_seg||refer_seg||vqa||reason_seg" \
  --sample_rates="9,3,3,1" \
  --exp_name="lisa-7b"
'''


//run for video dataset ytvos
'''
deepspeed --master_port=24999 train_ds.py \
  --version="/media/lcx/sia-vision/lisa/LISA/model/llava-v1.5-7b" \
  --dataset_dir='/media/lcx/sia-vision/lisa/LISA/dataset' \
  --vision_pretrained="/media/lcx/sia-vision/lisa/LISA/model/sam_vit_h.pth" \
  --dataset="ytvos" \
  --sample_rates="1" \
  --exp_name="lisa-7b"
'''


where the run.sh is the script file, you can modify the script file to run the code on different datasets and models.
the .sh file include the original image training and the video segmentation training. you can modify the script file to run the code on different datasets and models.



When training is finished, to get the full model weight:
```
cd ./runs/lisa-7b/ckpt_model && python zero_to_fp32.py . ../pytorch_model.bin
```

## Merge LoRA Weight
Merge the LoRA weights of `pytorch_model.bin`, save the resulting model into your desired path in the Hugging Face format:
```
CUDA_VISIBLE_DEVICES="0" python merge_lora_weights_and_save_hf_model.py \
  --version="PATH_TO_LLaVA" \
  --weight="PATH_TO_pytorch_model.bin" \
  --save_path="PATH_TO_SAVED_MODEL"
```

For example:
```
CUDA_VISIBLE_DEVICES="" python3 merge_lora_weights_and_save_hf_model.py \
  --version="./LLaVA/LLaVA-Lightning-7B-v1-1" \
  --weight="lisa-7b/pytorch_model.bin" \
  --save_path="./LISA-7B"
```

## Evaluation
for video segmentation, you can run the python file to evaluate the performance of the model on the test dataset:
'''
python inference_ytvos.py 
'''
some path in the code need to be modified according to your own environment.

after the evaluation, the result sould be upload to the competition website to abtain the final result: https://codalab.lisn.upsaclay.fr/competitions/3282
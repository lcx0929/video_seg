
# run for origin lisa
'''deepspeed --master_port=24999 train_ds.py \
  --version="/media/lcx/sia-vision/lisa/LISA/model/llava-v1.5-7b" \
  --dataset_dir='/media/lcx/sia-vision/lisa/LISA/dataset' \
  --vision_pretrained="/media/lcx/sia-vision/lisa/LISA/model/sam_vit_h.pth" \
  --dataset="sem_seg||refer_seg||vqa||reason_seg" \
  --sample_rates="9,3,3,1" \
  --exp_name="lisa-7b"'''

# run for video dataset ytvos
deepspeed --master_port=24999 train_ds.py \
  --version="/media/lcx/sia-vision/lisa/LISA/model/llava-v1.5-7b" \
  --dataset_dir='/media/lcx/sia-vision/lisa/LISA/dataset' \
  --vision_pretrained="/media/lcx/sia-vision/lisa/LISA/model/sam_vit_h.pth" \
  --dataset="ytvos" \
  --sample_rates="1" \
  --exp_name="lisa-7b"
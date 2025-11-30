IMAGES_PATH="xx"

#CODE TO CREATE THE DATASET
CUDA_VISIBLE_DEVICES=0 python execute.py --input_folder=$IMAGES_PATH --batch_size 64 --repo_id 'sd14'
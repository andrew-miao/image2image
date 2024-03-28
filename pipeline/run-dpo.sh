export MODEL_NAME="duongna/stable-diffusion-v1-4-flax"
export SAVE_DIR="path-to-save-model"
export INSTANCE_DIR="../../dreambooth/dataset/dog"
export GENERATE_DIR="generate/dog"

python train_dream_dpo.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --generated_data_dir=$GENERATE_DIR \
  --savepath=$SAVE_DIR \
  --dpo_beta=0.01 \
  --prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --learning_rate=5e-6 \
  --num_generated_images=100 \
  --max_train_steps=50 \
  --save_steps=10
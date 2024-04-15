export MODEL_NAME="duongna/stable-diffusion-v1-4-flax"
export SAVE_DIR="path-to-save-model"
export REFERENCE_DIR="../../dreambooth/dataset/dog"
export GENERATE_DIR="generate/dog"

python train_dream_dpo.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --reference_data_dir=$REFERENCE_DIR \
  --generated_data_dir=$GENERATE_DIR \
  --savepath=$SAVE_DIR \
  --dpo_beta=5.0 \
  --prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --learning_rate=5e-6 \
  --num_generated_images=64 \
  --max_train_steps=120 \
  --save_steps=40
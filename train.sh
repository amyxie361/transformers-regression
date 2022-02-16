#5e-5 9e-5
export CUDA_VISIBLE_DEVICES=0
for TASK_NAME in rte
do
for seed in 0
do
for lr in 3e-5
do
for BS in 32
do
for epoch in 5
do
#baseline_base_seed0_BS64_lr9e-5_epoch3
#MODEL_NAME="electra"
#MODEL_SIZE="base"
MODEL_NAME="bert"
MODEL_SIZE="base"
MODEL_TYPE="uncased"
# bert-base-uncased the right baseline
#--model_name_or_path google/${MODEL_NAME}-${MODEL_SIZE}-discriminator
python run_glue.py \
  --model_name_or_path ${MODEL_NAME}-${MODEL_SIZE}-${MODEL_TYPE} \
  --task_name $TASK_NAME \
  --do_train \
  --do_predict \
  --max_seq_length 128 \
  --per_device_train_batch_size ${BS} \
  --learning_rate ${lr} \
  --num_train_epochs ${epoch} \
  --seed ${seed} \
  --save_steps 5000 \
  --output_dir ./$TASK_NAME/baseline_${MODEL_NAME}_${MODEL_SIZE}_${MODEL_TYPE}_seed${seed}_BS${BS}_lr${lr}_epoch${epoch}
done
done
done
done
done

export CUDA_VISIBLE_DEVICES=0
for TASK_NAME in rte
do
for seed in 1 2 3 4 5
do
for lr in 1e-5 3e-5
do
for BS in 16
do
for epoch in 5
do
for temp in 1 2 3
do
for alpha in 0.5 1.0 2.0
do
#init_path=gated_v3/$TASK_NAME/gatedinit_base_seed${seed}_BS${BS}_lr${lr}_initepoch${init_epoch}
old_path=$TASK_NAME/baseline_base_seed0_BS32_lr9e-5_epoch5
save_path=distill/$TASK_NAME/bert-large_seed${seed}_BS${BS}_lr${lr}_epoch${epoch}_temp${temp}_alpha${alpha}

python run_glue_distill.py \
  --model_name_or_path bert-large-cased \
  --old_model_name_or_path ${old_path} \
  --task_name $TASK_NAME \
  --do_train \
  --do_predict \
  --tempreture ${temp} \
  --alpha ${alpha} \
  --max_seq_length 128 \
  --per_device_train_batch_size ${BS} \
  --learning_rate ${lr} \
  --num_train_epochs ${epoch} \
  --seed ${seed} \
  --save_steps 5000 \
  --output_dir ${save_path}



done
done
done
done
done
done
done

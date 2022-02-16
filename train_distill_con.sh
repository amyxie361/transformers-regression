export CUDA_VISIBLE_DEVICES=0
for TASK_NAME in mrpc
do
for seed in 1 2 3 4 5
do
for lr in 1e-4
do
for BS in 32
do
for epoch in 1 
do
for temp in 3
do
for alpha in 1.0
do

# gated_v3/mrpc/gatedfinal_base_seed3_BS32_lr4e-5_initepoch3_continueepoch5_lr24e-5/predict_results_mrpc.txt

#init_path=gated_v3/$TASK_NAME/gatedinit_base_seed${seed}_BS${BS}_lr${lr}_initepoch${init_epoch}
old_path=mrpc/baseline_base_seed0_BS64_lr9e-5_epoch3
save_path=distill_continue/$TASK_NAME/bert-base_seed${seed}_BS${BS}_lr${lr}_epoch${epoch}_temp${temp}_alpha${alpha}
new_path=mrpc/baseline_base_seed${seed}_BS64_lr9e-5_epoch3

python run_glue_distill.py \
  --model_name_or_path ${new_path} \
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

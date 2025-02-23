train_batch_size=8
test_batch_size=20
num_epochs=3
learning_rate=1.5e-4
lora_r=64
random_seed=2
dataset=$1
# dataset in [GSM8K, BoolQ, OBQA]
al=$2
# al in [random, Entropy, PE, Entropy_subtract, Entropy_dynamic, PE_subtract, PE_dynamic]
output_dir=output/${dataset}-bz_${train_batch_size}-epoch_${num_epochs}-lr_${learning_rate}-seed_${random_seed}_${al}
if [ ! -d ${output_dir} ];then
  mkdir -p ${output_dir}
fi


export CUDA_VISIBLE_DEVICES=0
accelerate launch --config_file accelerate_ds_zero3_cpu_offload.yaml \
    finetuning.py \
    --dataset_name dataset/${dataset} \
    --seed ${random_seed} \
    --train_batch_size ${train_batch_size} \
    --test_batch_size ${test_batch_size} \
    --output_dir ${output_dir} \
    --num_epochs ${num_epochs} \
    --learning_rate ${learning_rate} \
    --lora_r ${lora_r} \
    --iter_dataset dataset/${dataset}_init_0 \
    --al ${al} 2>&1 | tee $output_dir/training.log

for i in {1..9}
do
echo $i
   accelerate launch --config_file accelerate_ds_zero3_cpu_offload.yaml \
    finetuning.py \
    --dataset_name dataset/${dataset} \
    --seed ${random_seed} \
    --train_batch_size ${train_batch_size} \
    --test_batch_size ${test_batch_size} \
    --output_dir ${output_dir} \
    --num_epochs ${num_epochs} \
    --learning_rate ${learning_rate} \
    --lora_r ${lora_r} \
    --iter_dataset dataset/${dataset}_${al}_iter${i} \
    --al ${al} 2>&1 | tee $output_dir/training${i}.log
done
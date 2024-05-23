./main.py \
    --model_name_or_path xlnet-base-cased \
    --data_dir /data/papers/midas/d3/ \
    --output_dir /data/papers/midas/output \
    --overwrite_output_dir \
    --evaluate_during_training \
    --do_train \
    --do_eval \
    --max_seq_length 512 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --learning_rate 3e-5 \
    --warmup_steps 10000 \
    --num_train_epochs 3 \
    --eval_steps 10000 \
    --save_steps 40000 \
    --save_total_limit 3 \


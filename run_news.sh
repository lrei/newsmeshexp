echo 'EUS'
./main.py \
    --data_dir /data/papers/midas/news/EUS \
    --model_name_or_path /data/papers/midas/output \
    --overwrite_output_dir \
    --do_eval \
    --max_seq_length 512 \
    --per_device_eval_batch_size 1 \
    --output_dir /data/papers/midas/res/EUS \
    --overwrite_output_dir \
     --overwrite_cache \

echo 'FIN'
./main.py \
    --data_dir /data/papers/midas/news/FIN \
    --model_name_or_path /data/papers/midas/output \
    --overwrite_output_dir \
    --do_eval \
    --max_seq_length 512 \
    --per_device_eval_batch_size 1 \
    --output_dir /data/papers/midas/res/FIN \
    --overwrite_output_dir \
     --overwrite_cache \


echo 'INF'
./main.py \
    --data_dir /data/papers/midas/news/INF \
    --model_name_or_path /data/papers/midas/output \
    --overwrite_output_dir \
    --do_eval \
    --max_seq_length 512 \
    --per_device_eval_batch_size 1 \
    --output_dir /data/papers/midas/res/INF \
    --overwrite_output_dir \
     --overwrite_cache \

echo 'IRE'
./main.py \
    --data_dir /data/papers/midas/news/IRE \
    --model_name_or_path /data/papers/midas/output \
    --overwrite_output_dir \
    --do_eval \
    --max_seq_length 512 \
    --per_device_eval_batch_size 1 \
    --output_dir /data/papers/midas/res/IRE \
    --overwrite_output_dir \
     --overwrite_cache \

echo 'NIR'
./main.py \
    --data_dir /data/papers/midas/news/NIR \
    --model_name_or_path /data/papers/midas/output \
    --overwrite_output_dir \
    --do_eval \
    --max_seq_length 512 \
    --per_device_eval_batch_size 1 \
    --output_dir /data/papers/midas/res/NIR \
    --overwrite_output_dir \
     --overwrite_cache \

if [ ! -d "./run_log" ]; then
    mkdir ./run_log
fi
if [ ! -d "./run_log/log_test_win" ]; then
    mkdir ./run_log/log_test_win
fi
if [ ! -d "./run_log/log_test_win/ETTm1" ]; then
    mkdir ./run_log/log_test_win/ETTm1
fi
if [ ! -d "./run_log/log_test_win/ETTh1" ]; then
    mkdir ./run_log/log_test_win/ETTh1
fi
if [ ! -d "./run_log/log_test_win/ETTm2" ]; then
    mkdir ./run_log/log_test_win/ETTm2
fi

if [ ! -d "./run_log/log_test_win/ETTh2" ]; then
    mkdir ./run_log/log_test_win/ETTh2
fi
if [ ! -d "./run_log/log_test_win/electricity" ]; then
    mkdir ./run_log/log_test_win/electricity
fi

if [ ! -d "./run_log/log_test_win/Exchange" ]; then
    mkdir ./run_log/log_test_win/Exchange
fi

if [ ! -d "./run_log/log_test_win/Solar" ]; then
    mkdir ./run_log/log_test_win/Solar
fi

if [ ! -d "./run_log/log_test_win/weather" ]; then
    mkdir ./run_log/log_test_win/weather
fi

if [ ! -d "./run_log/log_test_win/Traffic" ]; then
    mkdir ./run_log/log_test_win/Traffic
fi

if [ ! -d "./run_log/log_test_win/PEMS03" ]; then
    mkdir ./run_log/log_test_win/PEMS03
fi

if [ ! -d "./run_log/log_test_win/PEMS04" ]; then
    mkdir ./run_log/log_test_win/PEMS04
fi

if [ ! -d "./run_log/log_test_win/PEMS07" ]; then
    mkdir ./run_log/log_test_win/PEMS07
fi
if [ ! -d "./run_log/log_test_win/PEMS08" ]; then
    mkdir ./run_log/log_test_win/PEMS08
fi


#ETTh1


python -u run.py \
    --task_name pretrain \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1 \
    --model SimMTM \
    --data ETTh1 \
    --features M \
    --seq_len 336 \
    --e_layers 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --n_heads 16 \
    --d_model 32 \
    --d_ff 64 \
    --positive_nums 2 \
    --mask_rate 0.5 \
    --learning_rate 0.001 \
    --batch_size 8 \
    --train_epochs 50 \
> ./run_log/log_test_win/ETTh1/'SimMTM_pretrain'0.01.log 2>&1


for pred_len in 96; do
    python -u run.py \
        --task_name finetune \
        --is_training 1 \
        --root_path ./dataset/ETT-small/ \
        --data_path ETTh1.csv \
        --model_id ETTh1 \
        --model SimMTM \
        --data ETTh1 \
        --features M \
        --seq_len 336 \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 3 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --n_heads 16 \
        --d_model 32 \
        --d_ff 64 \
        --learning_rate 0.0001 \
        --dropout 0.2 \
        --batch_size 8 \
> ./run_log/log_test_win/ETTh1/'SimMTM_finetune'$pred_len_0.01.log 2>&1

done

#ETTh2

python -u run.py \
    --task_name pretrain \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh2.csv \
    --model_id ETTh2 \
    --model SimMTM \
    --data ETTh2 \
    --features M \
    --input_len 336 \
    --seq_len 336 \
    --e_layers 2 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --n_heads 8 \
    --d_model 8 \
    --d_ff 32 \
    --positive_nums 2 \
    --mask_rate 0.5 \
    --learning_rate 0.001 \
    --batch_size 8 \
    --train_epochs 50 \
> ./run_log/log_test_win/ETTh2/'SimMTM_pretrain'0.01.log 2>&1


for pred_len in 96; do
    python -u run.py \
        --task_name finetune \
        --is_training 1 \
        --root_path ./dataset/ETT-small/ \
        --data_path ETTh2.csv \
        --model_id ETTh2 \
        --model SimMTM \
        --data ETTh2 \
        --input_len 336 \
        --features M \
        --seq_len 336 \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 2 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --n_heads 8 \
        --d_model 8 \
        --d_ff 32 \
        --dropout 0.4 \
        --head_dropout 0.2 \
        --batch_size 8 \
> ./run_log/log_test_win/ETTh2/'SimMTM_finetune'$pred_len_0.01.log 2>&1

done





#
#
#python -u run.py \
#    --task_name pretrain \
#    --root_path ./dataset/ETT-small/ \
#    --data_path ETTm1.csv \
#    --model_id ETTm1 \
#    --model SimMTM \
#    --data ETTm1 \
#    --features M \
#    --seq_len 336 \
#    --e_layers 2 \
#    --enc_in 7 \
#    --dec_in 7 \
#    --c_out 7 \
#    --n_heads 8 \
#    --d_model 32 \
#    --d_ff 64 \
#    --positive_nums 2 \
#    --mask_rate 0.5 \
#    --learning_rate 0.001 \
#    --batch_size 8 \
#    --train_epochs 50 \
#> 'TimeDART_my_SimMTM_ETTm1_pretrain'0.01.log 2>&1
#
#
#
#for pred_len in 96 192 336 720; do
#    python -u run.py \
#        --task_name finetune \
#        --is_training 1 \
#        --root_path ./dataset/ETT-small/ \
#        --data_path ETTm1.csv \
#        --model_id ETTm1 \
#        --model SimMTM \
#        --data ETTm1 \
#        --features M \
#        --seq_len 336 \
#        --label_len 48 \
#        --pred_len $pred_len \
#        --e_layers 2 \
#        --enc_in 7 \
#        --dec_in 7 \
#        --c_out 7 \
#        --n_heads 8 \
#        --d_model 32 \
#        --d_ff 64 \
#        --dropout 0 \
#> 'TimeDART_my_SimMTM_ETTm1_finetune'$pred_len_0.01.log 2>&1
#
#done
#
#
#python -u run.py \
#    --task_name pretrain \
#    --root_path ./dataset/ETT-small/ \
#    --data_path ETTm2.csv \
#    --model_id ETTm2 \
#    --model SimMTM \
#    --data ETTm2 \
#    --features M \
#    --seq_len 336 \
#    --e_layers 3 \
#    --enc_in 7 \
#    --dec_in 7 \
#    --c_out 7 \
#    --n_heads 8 \
#    --d_model 8 \
#    --d_ff 16 \
#    --positive_nums 2 \
#    --mask_rate 0.5 \
#    --learning_rate 0.001 \
#    --batch_size 8 \
#    --train_epochs 50 \
#> 'TimeDART_my_SimMTM_ETTm2_pretrain'0.01.log 2>&1
#
#
#
#for pred_len in 96 ; do
#    python -u run.py \
#        --task_name finetune \
#        --is_training 1 \
#        --root_path ./dataset/ETT-small/ \
#        --data_path ETTm2.csv \
#        --model_id ETTm2 \
#        --model SimMTM \
#        --data ETTm2 \
#        --features M \
#        --seq_len 336 \
#        --label_len 48 \
#        --pred_len $pred_len \
#        --e_layers 3 \
#        --enc_in 7 \
#        --dec_in 7 \
#        --c_out 7 \
#        --n_heads 8 \
#        --d_model 8 \
#        --d_ff 16 \
#        --dropout 0 \
#        --batch_size 64 \
#> 'TimeDART_my_SimMTM_ETTm2_finetune'$pred_len_0.01.log 2>&1
#
#done
#
#

#!/usr/bin/bash
module load cuda
module load cudnn
source activate xlm
EMB=832
RELOAD=dumped/train_lm_enfr_emb832_pmi2_60k/9910492/best-valid_mlm_ppl.pth
METHOD=pmi2_60k
MAX_BATCH=32
TOKS=1000
export NGPU=1; python -m torch.distributed.launch --nproc_per_node=$NGPU train.py --exp_name unsupMT_enfr_emb${EMB}_${METHOD}_mb${MAX_BATCH}_tok${TOKS} --dump_path ./dumped/ --reload_model ${RELOAD},${RELOAD} --data_path ./data/processed/en-fr_${METHOD}/ --lgs 'en-fr' --bt_steps 'en-fr-en,fr-en-fr' --encoder_only false --emb_dim ${EMB} --n_layers 6 --n_heads 8 --dropout 0.1 --attention_dropout 0.1 --gelu_activation true --tokens_per_batch ${TOKS} --max_batch_size ${MAX_BATCH} --bptt 256 --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001 --epoch_size 200000  --eval_bleu true --stopping_criterion 'valid_en-fr_mt_bleu,10' --validation_metrics 'valid_en-fr_mt_bleu' --debug_slurm True --rttae
#--ae_steps 'en,fr' --word_shuffle 3 --word_dropout 0.1 --word_blank 0.1 --lambda_ae '0:1,100000:0.1,300000:0'

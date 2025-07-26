export CUDA_VISIBLE_DEVICES="3" 


python main_dynn.py --data-set NaSC \
--arch GFNet-Dynn --num_epoch 200 \
--input-size 128 \
--ce_ic_tradeoff 0.75  \
--max_warmup_epoch 15 \
--resume \
--eval

# python main_dynn.py --data-set NWPU \
# --arch GFNet-Dynn --num_epoch 100 \
# --input-size 256 \
# --ce_ic_tradeoff 0.75  \
# --eval

# python main_dynn.py --data-set UCM \
# --arch GFNet-Dynn --num_epoch 20 \
# --input-size 256 \
# --ce_ic_tradeoff 0.75  \
# --resume \
# --eval

# python main_dynn.py --data-set PatternNet \
# --arch GFNet-Dynn --num_epoch 100 \
# --input-size 256 \
# --ce_ic_tradeoff 0.75  \
# --resume \
# --eval

# python main_dynn.py --data-set AID \
# --arch GFNet-Dynn --num_epoch 100 \
# --input-size 600 \
# --ce_ic_tradeoff 0.75  \
# --eval

SELECT="subset"
SAMPLE_SIZE=50

python sample_mdn.py \
  --flagfile=configs/custom_sample_mdn-mel-32seq-512.cfg \
  --sample_seed=42 \
  --sample_size=$SAMPLE_SIZE \
  --sampling_dir=dataset/samples_"$SELECT"_2


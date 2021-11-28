SELECT="subset"
SAMPLE_SIZE=50

python sample_ncsn.py \
  --flagfile=configs/custom_ddpm-mel-32seq-512.cfg \
  --compute_metrics=True \
  --sample_seed=42 \
  --sample_size="$SAMPLE_SIZE" \
  --sampling_dir=dataset/samples_"$SELECT"_sdedit1
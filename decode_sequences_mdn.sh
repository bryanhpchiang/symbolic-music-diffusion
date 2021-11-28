SELECT="subset_2"

python sample_audio.py \
  --input=dataset/samples_"$SELECT"/mdn \
  --output=dataset/samples_"$SELECT"/midi \
  --n_synth=50 \
  --include_wav=True \
  --include_plots=True \
  --gen_only=False \

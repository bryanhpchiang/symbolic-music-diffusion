SELECT="subset"

python sample_audio.py \
  --input=dataset/samples_"$SELECT"/ncsn \
  --output=dataset/samples_"$SELECT"/midi \
  --n_synth=50 \
  --include_wav=True \
  --include_plots=True \
  --gen_only=True \

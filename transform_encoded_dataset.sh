SELECT=subset
ENCODED_TFRECORD=dataset/encoded_note_sequences_"$SELECT"
PREPROCESSED_TFRECORD=dataset/preprocessed_note_sequences_"$SELECT"

python transform_encoded_data.py\
  --encoded_data=$ENCODED_TFRECORD\
  --output_path=$PREPROCESSED_TFRECORD\
  --mode=sequences \
  --context_length=32
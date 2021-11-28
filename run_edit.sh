# convert input midi files to tfrecord NoteSequences
SELECT=mario_bros
INPUT_DIRECTORY=lmd_full/"$SELECT"/

# TFRecord file that will contain NoteSequence protocol buffers.
SEQUENCES_TFRECORD=dataset/note_sequences_"$SELECT".tfrecord

convert_dir_to_note_sequences \
  --input_dir=$INPUT_DIRECTORY \
  --output_file=$SEQUENCES_TFRECORD \
  --recursive

# exit 1
# run NoteSequences through MusicVAE 2bar to encode every 2-measures as latents
MUSICVAE_CKPT=cat-mel_2bar_big/cat-mel_2bar_big.ckpt
# SELECT=subset
# SEQUENCES_TFRECORD=dataset/note_sequences_"$SELECT".tfrecord
ENCODED_TFRECORD=dataset/encoded_note_sequences_"$SELECT".tfrecord


python generate_song_data_beam.py \
  --checkpoint=$MUSICVAE_CKPT \
  --input=$SEQUENCES_TFRECORD \
  --output=$ENCODED_TFRECORD \
  --mode=multitrack



# SELECT=subset
ENCODED_TFRECORD_FOLDER=dataset/encoded_note_sequences_"$SELECT"
mkdir -p "$ENCODED_TFRECORD_FOLDER"
cp "$ENCODED_TFRECORD-00000-of-00001" "$ENCODED_TFRECORD_FOLDER/training_seqs.tfrecord-00000-of-00001"
cp "$ENCODED_TFRECORD-00000-of-00001" "$ENCODED_TFRECORD_FOLDER/eval_seqs.tfrecord-00000-of-00001"
# PREPROCESSED_TFRECORD=dataset/preprocessed_note_sequences_"$SELECT"

python transform_encoded_data.py\
  --encoded_data=$ENCODED_TFRECORD_FOLDER \
  --output_path=$PREPROCESSED_TFRECORD\
  --mode=sequences \
  --context_length=32

exit 1
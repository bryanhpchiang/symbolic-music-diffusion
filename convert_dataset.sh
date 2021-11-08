# convert input midi files to tfrecord NoteSequences
SELECT=subset
INPUT_DIRECTORY=lmd_full/"$SELECT"/

# TFRecord file that will contain NoteSequence protocol buffers.
SEQUENCES_TFRECORD=dataset/note_sequences_"$SELECT".tfrecord

convert_dir_to_note_sequences \
  --input_dir=$INPUT_DIRECTORY \
  --output_file=$SEQUENCES_TFRECORD \
  --recursive

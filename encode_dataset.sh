# run NoteSequences through MusicVAE 2bar to encode every 2-measures as latents
MUSICVAE_CKPT=cat-mel_2bar_big/cat-mel_2bar_big.ckpt
SELECT=subset
SEQUENCES_TFRECORD=dataset/note_sequences_"$SELECT".tfrecord
ENCODED_TFRECORD=dataset/encoded_note_sequences_"$SELECT".tfrecord

python generate_song_data_beam.py \
  --checkpoint=$MUSICVAE_CKPT \
  --input=$SEQUENCES_TFRECORD \
  --output=$ENCODED_TFRECORD
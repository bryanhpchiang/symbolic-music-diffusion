LONG_TRACK=mario_bros/c9d0cec33b96c45f56475fd1aaa85048.mid
SHORT_TRACK=midis/twinkle.mid
MUSICVAE_CKPT=cat-mel_2bar_big/cat-mel_2bar_big.ckpt
FUNKY_TRACK=c/ca8824e02b7f540721c29c25cc8a5fd2.mid

python sd_edit.py --long_track "lmd_full/$FUNKY_TRACK" \
                --editing_space "latent" \
                --short_track "$SHORT_TRACK" \
                --latent_checkpoint "$MUSICVAE_CKPT" \
                --sampling "ddpm" \
                --flagfile=configs/custom_ddpm-mel-32seq-512.cfg \
                --sample_seed=42 \
                --output test_sd_edit14
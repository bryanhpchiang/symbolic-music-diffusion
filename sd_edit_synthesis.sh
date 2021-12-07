space="latent"
mode="composition"
melody_id="repeated_c"
LONG_TRACK=mario_bros/c9d0cec33b96c45f56475fd1aaa85048.mid
# SHORT_TRACK=midis/twinkle.mid
SHORT_TRACK=custom_midis/"$melody_id".mid
# SHORT_TRACK=custom_midis/scale.mid
# SHORT_TRACK=custom_midis/minor_scale.mid
MUSICVAE_CKPT=cat-mel_2bar_big/cat-mel_2bar_big.ckpt
# FUNKY_TRACK=c/ca8824e02b7f540721c29c25cc8a5fd2.mid

python sd_edit.py --long_track "lmd_full/edit_splits" \
                --editing_space "$space" \
                --mode "$mode" \
                --short_track "$SHORT_TRACK" \
                --latent_checkpoint "$MUSICVAE_CKPT" \
                --sampling "ddpm" \
                --flagfile=configs/custom_ddpm-mel-32seq-512.cfg \
                --sample_seed=42 \
                --num_sigmas=1000 \
                --output "$space"_"$mode"_"$melody_id"_trimmed
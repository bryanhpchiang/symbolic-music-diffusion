from magenta.models.music_vae import TrainedModel
from IPython import embed
import config
MUSICVAE_CKPT="cat-mel_2bar_big/cat-mel_2bar_big.ckpt"
MODEL="melody-2-big"
model_config = config.MUSIC_VAE_CONFIG[MODEL]
model = TrainedModel(model_config,    batch_size=1,checkpoint_dir_or_path=MUSICVAE_CKPT)
model2 = TrainedModel(model_config,    batch_size=1,checkpoint_dir_or_path=MUSICVAE_CKPT)
model3 = TrainedModel(model_config,    batch_size=1,checkpoint_dir_or_path=MUSICVAE_CKPT)
embed()
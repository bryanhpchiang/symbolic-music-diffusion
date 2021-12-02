"""
"""
import glob
import os
from pathlib import Path

import jax
import note_seq
import numpy as np
from scipy.sparse import data
import tensorflow as tf
from absl import app, flags, logging
from flax.training import checkpoints
from IPython import embed
from magenta.models.music_vae import TrainedModel

import config
import input_pipeline
import sample_ncsn
import train_ncsn
from utils import data_utils, ebm_utils, song_utils, train_utils

FLAGS = flags.FLAGS


flags.DEFINE_string(
    "long_track",
    "mario.mid",
    "File path to MIDI file of original long track that we will embed another shorter melody (either existing or user-generated)",
)

flags.DEFINE_string(
    "short_track",
    "twinkle.mid",
    "Path to MIDI file of short track to embed within the longer one.",
)

flags.DEFINE_enum(
    "mode",
    "composition",
    ["composition", "editing", "synthesis"],
    """Editing mode.
    Composition: existing melody inserted within longer melody
    Editing:  user sequences of notes inserted into longer melody
    Synthesis: only user sequence of notes
    """,
)

flags.DEFINE_enum(
    "editing_space",
    "output",
    ["output", "latent"],
    "Space to perform the editing in (latent or output MIDI).",
)
flags.DEFINE_integer(
    "latent_edit_start_index",
    10,
    "We choose the start position within long_track to embed our short_track. Only applicable when editing_space is 'latent'.",
)

flags.DEFINE_string(
    "latent_model",
    "melody-2-big",
    "Model configuration to use for embedding in the latent space.",
)
flags.DEFINE_string(
    "latent_checkpoint", "fb512_0trackmin/model.ckpt-99967", "Latent model checkpoint."
)
flags.DEFINE_string("output", "sd_edit_test/", "Output folder to save results to.")


def _extract_songs(model, model_config, ns, chunk_length=2):
    melodies = song_utils.extract_melodies(ns)
    songs = [
        song_utils.Song(melody, model_config.data_converter, chunk_length)
        for melody in melodies
    ]
    return songs


def _save_ns_as_midi(ns, path):
    note_seq.note_sequence_to_midi_file(ns, path)


def _sample_edits(samples, masks, rng_seed=1):
    """Fork of sample_ncsn.infill_samples."""
    rng = jax.random.PRNGKey(rng_seed)
    rng, model_rng = jax.random.split(rng)

    # Create a model with dummy parameters and a dummy optimizer
    model_kwargs = {
        "num_layers": FLAGS.num_layers,
        "num_heads": FLAGS.num_heads,
        "num_mlp_layers": FLAGS.num_mlp_layers,
        "mlp_dims": FLAGS.mlp_dims,
    }
    model = train_ncsn.create_model(
        rng, samples.shape[1:], model_kwargs, batch_size=1, verbose=True
    )
    optimizer = train_ncsn.create_optimizer(model, 0)
    ema = train_utils.EMAHelper(mu=0, params=model.params)
    early_stop = train_utils.EarlyStopping()

    # Load learned parameters
    optimizer, ema, early_stop = checkpoints.restore_checkpoint(
        FLAGS.model_dir,
        (optimizer, ema, early_stop),
    )

    # Create noise schedule
    sigmas = ebm_utils.create_noise_schedule(
        FLAGS.sigma_begin,
        FLAGS.sigma_end,
        FLAGS.num_sigmas,
        schedule=FLAGS.schedule_type,
    )

    if FLAGS.sampling == "ald":
        sampling_algorithm = ebm_utils.annealed_langevin_dynamics
    elif FLAGS.sampling == "cas":
        sampling_algorithm = ebm_utils.consistent_langevin_dynamics
    elif FLAGS.sampling == "ddpm":
        sampling_algorithm = ebm_utils.diffusion_dynamics
    else:
        raise ValueError(f"Unknown sampling algorithm: {FLAGS.sampling}")

    init_rng, ld_rng = jax.random.split(rng)
    init = jax.random.uniform(key=init_rng, shape=samples.shape)
    generated, collection, ld_metrics = sampling_algorithm(
        ld_rng,
        optimizer.target,
        sigmas,
        init,
        FLAGS.ld_epsilon,
        FLAGS.ld_steps,
        FLAGS.denoise,
        True,
        infill_samples=samples,
        infill_masks=masks,
    )

    ld_metrics = ebm_utils.collate_sampling_metrics(ld_metrics)
    return generated, collection, ld_metrics


def _preprocess_batch(batch):
    # preprocess data according to flags of sample_ncsn
    # goal is to reuse as much of existing data preproc infra as possible to avoid errors
    """
    passed into input_pipeline.get_dataset

    dim_weights_ckpt, pca_ckpt = None
    problem = 'vae'

    things to actually look out for
    normalize = True, slice_ckpt = './checkpoints/slice-mel-512.pkl'
    """
    # extract relevant indices (42 from 512)
    pca = (
        data_utils.load(os.path.expanduser(FLAGS.pca_ckpt)) if FLAGS.pca_ckpt else None
    )
    slice_idx = (
        data_utils.load(os.path.expanduser(FLAGS.slice_ckpt))
        if FLAGS.slice_ckpt
        else None
    )
    dim_weights = (
        data_utils.load(os.path.expanduser(FLAGS.dim_weights_ckpt))
        if FLAGS.dim_weights_ckpt
        else None
    )

    # pca, dim_weights should be None
    batch = input_pipeline.slice_transform(
        batch, problem=FLAGS.problem, slice_idx=slice_idx, dim_weights=dim_weights
    )

    # normalize to [-1, 1]
    ds_min, ds_max = (
        tf.math.reduce_min(batch),
        tf.math.reduce_max(batch),
    )  # custom to replace data_utils.compute_dataset_min_max
    batch = input_pipeline.normalize_dataset(batch, ds_min, ds_max)

    # this would usually be taken care of by tfds.as_numpy(...)
    batch = np.array(batch)

    print(f"{batch.shape = }, {batch.max() = }, {batch.min() = }")
    return batch, ds_min, ds_max


def _extract_latents(model, songs):
    encoding_matrices = song_utils.encode_songs(model, songs)
    encoding_matrices = np.array([batch[0] for batch in encoding_matrices])
    # TODO: concatenate into a single matrix with all latents

    # for now, just take the first one
    batch = encoding_matrices[0:1]
    processed, ds_min, ds_max = _preprocess_batch(batch)
    return processed, ds_min, ds_max


def _decode_emb(emb, model, data_converter, chunks_only=False):
    """Generates NoteSequence objects from set of embeddings.

    Args:
      emb: Embeddings of shape (n_seqs, seq_length, 512).
      model: Pre-trained MusicVAE model used for decoding.
      data_converter: Corresponding data converter for model.
      chunks_only: If True, assumes embeddings are of the shape (n_seqs, 512)
          where each generated NoteSequence corresponds to one embedding.

    Returns:
      A list of decoded NoteSequence objects.
    """
    if chunks_only:
        assert len(emb.shape) == 2
        samples = song_utils.embeddings_to_chunks(emb, model)
        samples = [
            song_utils.Song(sample, data_converter, reconstructed=True)
            for sample in samples
        ]
    else:
        samples = []
        count = 0
        for emb_sample in emb:
            if count % 100 == 0:
                logging.info(f"Decoded {count} sequences.")
            count += 1
            recon = song_utils.embeddings_to_song(emb_sample, model, data_converter)
            samples.append(recon)

    return samples


def main(argv):
    del argv

    # setup
    model_config = config.MUSIC_VAE_CONFIG[FLAGS.latent_model]
    model = TrainedModel(
        model_config, batch_size=1, checkpoint_dir_or_path=FLAGS.latent_checkpoint
    )

    output_dir = FLAGS.output
    os.makedirs(output_dir)

    # load note_seqs
    long_ns = note_seq.midi_file_to_note_sequence(FLAGS.long_track)
    short_ns = note_seq.midi_file_to_note_sequence(FLAGS.short_track)

    # load in latents for melodies
    long_songs = _extract_songs(model, model_config, long_ns)
    short_songs = _extract_songs(model, model_config, short_ns)

    # save all the melodies to disk
    print(f"{len(long_songs) = }, {len(short_songs) = }")
    for i, song in enumerate(long_songs):
        _save_ns_as_midi(song.note_sequence, f"{output_dir}/long_melody_{i}.mid")
    for i, song in enumerate(short_songs):
        _save_ns_as_midi(song.note_sequence, f"{output_dir}/short_melody_{i}.mid")

    # edit the melodies
    edited_samples = None
    masks = None
    ds_min = ds_max = None

    if FLAGS.mode in ["composition", "editing"]:
        if FLAGS.editing_space == "latent":
            # select which song we want to use
            # TODO: modify if needed later
            long_latent, ds_min, ds_max = _extract_latents(model, [long_songs[3]])
            short_latent, _, _ = _extract_latents(model, [short_songs[0]])

            edited = long_latent
            start = FLAGS.latent_edit_start_index

            edit_idx = list(range(start, start + short_latent.shape[1]))
            edited[:, edit_idx, :] = short_latent

            edited_samples = edited

            masks = np.ones(edited.shape)  # 1 to hold fixed
            masks[:, edit_idx, :] = 0  # 0 to edit
    elif FLAGS.mode in ["synthesis"]:
        pass
    else:
        raise NotImplementedError(f"{FLAGS.mode} is not supported.")

    # do the sampling
    generated, collection, ld_metrics = _sample_edits(
        edited_samples, masks, rng_seed=FLAGS.sample_seed
    )
    # embed()

    # decode
    pca_ckpt, slice_ckpt, dim_weights_ckpt, = (
        FLAGS.pca_ckpt,
        FLAGS.slice_ckpt,
        FLAGS.dim_weights_ckpt,
    )
    pca = data_utils.load(os.path.expanduser(pca_ckpt)) if pca_ckpt else None
    slice_idx = data_utils.load(os.path.expanduser(slice_ckpt)) if slice_ckpt else None
    dim_weights = (
        data_utils.load(os.path.expanduser(dim_weights_ckpt))
        if dim_weights_ckpt
        else None
    )
    gen = input_pipeline.inverse_data_transform(
        generated,
        normalize=FLAGS.normalize,
        pca=pca,
        data_min=ds_min,
        data_max=ds_max,
        slice_idx=slice_idx,
        dim_weights=dim_weights,
        out_channels=512,
    )
    print(f"{gen.shape = }")

    # save everything to disk
    id2emb = {
        "gen": gen,
        "baseline": edited_samples,
    }

    for id, emb in id2emb.items():
        ns_dir = os.path.join(output_dir, id, "ns")
        Path(ns_dir).mkdir(parents=True, exist_ok=True)
        midi_dir = os.path.join(output_dir, id, "midi")
        Path(midi_dir).mkdir(parents=True, exist_ok=True)

        # audio_dir = os.path.join(FLAGS.output, sample_split, "audio")
        # image_dir = os.path.join(FLAGS.output, sample_split, "images")
        # ns_dir = os.path.join(FLAGS.output, sample_split, "ns")
        # Path(audio_dir).mkdir(parents=True, exist_ok=True)
        # Path(image_dir).mkdir(parents=True, exist_ok=True)

        decoded = _decode_emb(emb, model, model_config.data_converter)
        for i, song in enumerate(decoded):
            ns = song.note_sequence
            # ns
            data_utils.save(ns, os.path.join(ns_dir, f"{i+1}.pkl"))
            # midi
            note_seq.note_sequence_to_midi_file(
                ns, os.path.join(midi_dir, f"{i+1}.mid")
            )

    embed()


if __name__ == "__main__":
    app.run(main)

"""
Compute list of splits for "gt" and "edited"
"""
import os

import note_seq
import numpy as np
from absl import app, flags, logging
from IPython import embed
from tqdm import tqdm

from utils import data_utils, ebm_utils, song_utils, train_utils

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "midi_list", "lmd_full/subset_list", "List of MIDI files to sample."
)
flags.DEFINE_integer("num_samples", 50, "Number of samples in each split.")
flags.DEFINE_integer(
    "min_length_sec", 90, "Minimum length of MIDI files in seconds to keep."
)
flags.DEFINE_string(
    "output", "lmd_full/edit_splits", "Output folder to write split note sequences to."
)


def _filter(l, idx):
    result = []
    for i in idx:
        result.append(l[i])
    return result


def main(argv):
    del argv
    with open(FLAGS.midi_list, "r") as f:
        lines = [x.strip() for x in f.readlines()]

    note_seqs = []

    for l in lines:
        print(l)
        try:
            ns = note_seq.midi_file_to_note_sequence(l)
            if ns.total_time >= FLAGS.min_length_sec:
                note_seqs.append((l, ns))
            # embed()
        except Exception as e:
            print(f"Failed to load MIDI file: {e}.")

        if len(note_seqs) >= 2 * FLAGS.num_samples:
            break

    print(f"{len(lines) = }, {len(note_seqs) = }")
    idxs = np.arange(len(note_seqs))

    np.random.seed(0)
    np.random.shuffle(idxs)

    real_idxs = idxs[: FLAGS.num_samples]
    edit_idxs = idxs[FLAGS.num_samples :]

    assert len(real_idxs) == len(edit_idxs)

    real_note_seqs = _filter(note_seqs, real_idxs)
    edit_note_seqs = _filter(note_seqs, edit_idxs)

    # save both to disk
    output_dir = FLAGS.output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data_utils.save(real_note_seqs, os.path.join(output_dir, "real_note_seqs.pkl"))
    data_utils.save(edit_note_seqs, os.path.join(output_dir, "edit_note_seqs.pkl"))
    embed()


if __name__ == "__main__":
    app.run(main)

"""Calculates OA metrics for """
import glob
import os
import pickle

from absl import app, flags
from IPython import embed

from utils import metrics

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "input", "dataset/subset/midi", "Location of MIDI output from sample_audio."
)


def _load_pkl(fname):
    with open(fname, "rb") as f:
        return pickle.load(f)


def main(argv):
    # load real and generated note_seqs
    id2note_seqs = {}
    ids = ["real", "gen"]
    for id in ids:
        path = os.path.join(FLAGS.input, id, "ns")
        # load all the pickles
        id2note_seqs[id] = [_load_pkl(fname) for fname in glob.glob(f"{path}/*.pkl")]

    # calculate metrics
    (
        pitch_consistency,
        pitch_variance,
        duration_consistency,
        duration_variance,
    ) = metrics.framewise_self_sim(id2note_seqs["real"], id2note_seqs["gen"])

    results = f"{pitch_consistency = }, {pitch_variance = }, {duration_consistency =}, {duration_variance = }"
    print(results)

    # write to disk
    output = os.path.join(FLAGS.input, "self_sim.txt")
    with open(output, "w") as f:
        f.write(f"{results}\n")


if __name__ == "__main__":
    app.run(main)

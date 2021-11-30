"""
"""
import glob

import tensorflow as tf
from absl import app, flags, logging
from IPython import embed
import note_seq

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
    "editing_output",
    "output",
    ["output", "latent"],
    "Space to perform the editing in (latent or output MIDI).",
)


def main(argv):
    del argv
    long_track = note_seq.midi_file_to_note_sequence(FLAGS.long_track)
    short_track = note_seq.midi_file_to_note_sequence(FLAGS.short_track)

    # load in melodies

    # edit the melodies

    # create the mask

    # load the model

    # do the sampling

    # save the sampled output
    embed()


if __name__ == "__main__":
    app.run(main)

import os
from pathlib import Path

import note_seq
from absl import app, flags, logging
from bokeh.io import export_png
from IPython import embed

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "midi_path",
    "lmd_full/",
    "Path to MIDI file to visualize",
)
flags.DEFINE_string("output", "twinkle.png", "Place to write visualization to.")


def main(argv):
    del argv
    # load in midi file
    ns = note_seq.midi_file_to_note_sequence(FLAGS.midi_path)
    # embed()
    fig = note_seq.plot_sequence(ns, show_figure=False)
    plot_path = FLAGS.output
    # Path(plot_path).mkdir(parents=True, exist_ok=True)
    export_png(fig, filename=plot_path)


if __name__ == "__main__":
    app.run(main)

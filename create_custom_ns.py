from math import e
import os
from pathlib import Path

import note_seq
from absl import app, flags, logging
from note_seq.protobuf import music_pb2
from pathlib import Path
from bokeh.io import export_png
import pretty_midi
from IPython import embed

FLAGS = flags.FLAGS
flags.DEFINE_string("id", "repeated_c", "Melody to generate.")
flags.DEFINE_string("output", "custom_midis/", "Output MIDI directory.")


def main(argv):
    del argv

    ns = music_pb2.NoteSequence()

    if FLAGS.id == "repeated_c":
        for i in range(0, 10, 2):
            ns.notes.add(
                pitch=60, start_time=float(i), end_time=float(i + 1), velocity=80
            )
        embed()
    elif FLAGS.id == "scale":
        pitches = [60, 62, 64, 65, 67, 69, 71, 72]
        note_names = ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"]
        offset = 1
        duration = 0.5
        for i, note_name in enumerate(note_names):
            pitch = pretty_midi.note_name_to_number(note_name)
            ns.notes.add(
                pitch=pitch,
                start_time=i * duration + offset,
                end_time=(i + 1) * duration + offset,
                velocity=80,
            )
    elif FLAGS.id == "minor_scale":
        note_names = ["E4", "F#4", "G4", "A4", "B4", "C5", "D#5", "E5"]
        offset = 0.5
        duration = 0.5
        repeats = 5
        for j in range(repeats):
            for i, note_name in enumerate(note_names):
                pitch = pretty_midi.note_name_to_number(note_name)
                start_time = i * duration + offset
                end_time = (i + 1) * duration + offset
                ns.notes.add(
                    pitch=pitch,
                    start_time=start_time,
                    end_time=end_time,
                    velocity=80,
                )
            offset = end_time

    # save ns
    Path(FLAGS.output).mkdir(parents=True, exist_ok=True)
    midi_path = os.path.join(FLAGS.output, f"{FLAGS.id}.mid")
    note_seq.note_sequence_to_midi_file(ns, midi_path)


if __name__ == "__main__":
    app.run(main)

import json

import music21 as m21
import numpy as np
import tensorflow.keras as keras
from preprocess import MAPPING_PATH, SEQUENCE_LENGTH


class MelodyMaker:
    def __init__(self, model_path='model.h5'):

        self.model_path = model_path
        self.model = keras.models.load_model(model_path)

        with open(MAPPING_PATH, 'r') as fp:
            self._mappings = json.load(fp)

        self._start_symbols = ['/'] * SEQUENCE_LENGTH

    def generate_melody(self, seed, num_steps, max_sequence_length, temperature):

        # create seed with start symbol
        seed = seed.split()
        melody = seed
        seed = self._start_symbols + seed

        # map seed to int
        seed = [self._mappings[symbol] for symbol in seed]

        for _ in range(num_steps):
            # limit seed to max sequence length
            seed = seed[-max_sequence_length:]

            # one-hot encode seed
            onehot_seed = keras.utils.to_categorical(
                seed, num_classes=len(self._mappings))
            onehot_seed = onehot_seed[np.newaxis, ...]

            # make a prediction
            probabilities = self.model.predict(onehot_seed)[0]
            # [0.1, 0.2, 0.1, 0.6] -> 1
            output_int = self._sample_with_temperature(
                probabilities, temperature)

            # update seed
            seed.append(output_int)

            # map int to our encoding
            output_symbol = [
                k for k, v in self._mappings.items() if v == output_int][0]

            # check if end of melody
            if output_symbol == '/':
                break

            # update the melody
            melody.append(output_symbol)

        return melody

    def _sample_with_temperature(self, probabilities, temperature):
        predictions = np.log(probabilities) / temperature
        # softmax
        probabilities = np.exp(predictions) / np.sum(np.exp(predictions))

        choices = range(len(probabilities))  # [0, 1, 2, 3]
        index = np.random.choice(choices, p=probabilities)

        return index

    def save_melody(self, melody, step_duration=0.25, format='midi', file_name='mel.midi'):
        # create music21 stream
        stream = m21.stream.Stream()

        # parse all symbols in melody and create note/rests
        start_symbol = None
        step_counter = 1

        for i, symbol in enumerate(melody):
            # handle case where we have a note/rest
            if symbol != '_' or i + 1 == len(melody):
                # ensure note/rest beyond the first one
                if start_symbol is not None:
                    quarter_length_duration = step_duration * step_counter
                    # handle rest
                    if start_symbol == 'r':
                        m21_event = m21.note.Rest(
                            quarterLength=quarter_length_duration)

                    # handle note
                    else:
                        m21_event = m21.note.Note(
                            int(start_symbol),
                            quarterLength=quarter_length_duration)

                    stream.append(m21_event)

                    # reset the step counter
                    step_counter = 1
                start_symbol = symbol

            # handle case where we have a prolongation sign
            else:
                step_counter += 1

        # write the m21 stream to a midi file
        stream.write(format, file_name)


if __name__ == '__main__':
    mg = MelodyMaker()
    seed = '55 _ 57 _ 59 _ 60 _ _ _ '
    melody = mg.generate_melody(seed, 500, SEQUENCE_LENGTH, 0.7)
    print(melody)
    mg.save_melody(melody)

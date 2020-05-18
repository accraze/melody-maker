
import os

import music21 as m21

KERN_DATASET_PATH = "deutschl/test"
ACCEPTABLE_DURATIONS = [
	0.25,
	0.5,
	0.75,
	1,
	1.5,
	2,
	3,
	4
]


def load_songs_in_kern(dataset_path):
	songs = []
	# go through all files in dataset and load w/ music21
	for path, subdir, files in os.walk(dataset_path):
		for file in files:
			if file[-3:] == "krn":
				song = m21.converter.parse(os.path.join(path, file))
				songs.append(song)
	return songs

def has_acceptable_durations(song, acceptable_durations):
	for note in song.flat.notesAndRests:
		if note.duration.quarterLength not in acceptable_durations:
			return False
	return True

def transpose(song):
	"""
	Transpose to Cmaj or Aminor.
	"""
	# get key from the song
	parts = song.getElementsByClass(m21.stream.Part)
	measures_part0 = parts[0].getElementsByClass(m21.stream.Measure)
	key = measures_part0[0][4]

	# estimate key using m21
	if not isinstance(key, m21.key.Key):
		key = song.analyze('key')
	
	# get interval for transposition
	if key.mode == 'major':
		interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
	elif key.mode == 'minor':
		interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch('A'))

	# transpose song by calculated interval
	transposed_song = song.transpose(interval)
	return transposed_song

def preprocess(dataset_path):
	# load folksongs
	print('loading songs')
	songs = load_songs_in_kern(dataset_path)
	print(f'loaded {len(songs)} songs')

	for song in songs:
		# filter out songs with non-acceptable durations
		if not has_acceptable_durations(song, ACCEPTABLE_DURATIONS):
			continue


		# transpose to Cmaj/Amin
		song = transpose(song)

		# encode songs with music time series representation

		# save songs to text file

if __name__ == '__main__':
	songs = load_songs_in_kern(KERN_DATASET_PATH)
	print(f'loaded {len(songs)} songs')
	song = songs[0]
	transpose(song)

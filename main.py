from os import listdir
from os.path import isfile, join

from madmom.features import RNNBeatProcessor, MultiModelSelectionProcessor, BeatTrackingProcessor, BeatDetectionProcessor,\
                            CRFBeatDetectionProcessor, DBNBeatTrackingProcessor, RNNDownBeatProcessor,\
                            DBNDownBeatTrackingProcessor


audio_dir = '../data/music_speech/music_wav/'
audio_files = [f for f in listdir(audio_dir) if isfile(join(audio_dir, f))]

beat_base_algo = RNNBeatProcessor()
downbeat_base_algo = RNNDownBeatProcessor()

beat_algos = [BeatTrackingProcessor(fps=100), BeatDetectionProcessor(fps=100), CRFBeatDetectionProcessor(fps=100),
                        DBNBeatTrackingProcessor(fps=100)]
# MultiModelSelectionProcessor(num_ref_predictions=None)

# downbeat_algos = [DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)]
# RNNBarProcessor(), DBNBarTrackingProcessor() # for given beat positions

beats = []
downbeats = []
for audio in audio_files:
    for algo in beat_algos:
        act = RNNBeatProcessor()(audio_dir+audio)
        beats.append(algo(act))
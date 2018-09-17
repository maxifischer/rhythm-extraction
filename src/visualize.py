from matplotlib import pyplot as plt
import numpy as np
from keras.models import Model
import pdb

"""
In this file, music and speach always mean single audio files and are of the shape: X.shape[1:]
- only the batch dimension is cut off
"""

def linear_transition(music, speech, start, end, model, result_dir=None):
    """
    Creates an audio with a linear transition from music to speech in the interval [start, end]
    """
    start, end = int(start), int(end)
    timeframes = music.shape[1]

    print("music", music.shape)
    w_speech = np.zeros(timeframes)
    w_speech[start:] = 1
    w_speech[start:end] = np.arange(0, 1, 1/(end-start))
    w_speech = w_speech[None, :, None]

    print("w_speech", w_speech.shape)

    audio = w_speech*speech + (1-w_speech)*music

    # flat w_speech
    w_speech = w_speech[0,:,0]

    print("audio", audio.shape)

    y_time = model.predict(audio[None, ...])[0,0,:,0]

    print("y", y_time.shape)

    # compute correlation
    w = 0*y_time
    start_w, end_w = int(start*len(y_time)/timeframes), int(end*len(y_time)/timeframes)
    w[end_w:] = 1
    w[start_w:end_w] = np.arange(0, 1, 1/(end_w-start_w))

    corr = np.corrcoef(-w, y_time)[0,1]

    timesteps = np.array(list(range(len(y_time))))
    plt.plot(timesteps, y_time, label="p(Music | audio)")
    plt.plot(timesteps, 1-w, "--", label="t(Music | audio)")
    plt.title("Linear Transition - music to speech\ncorr(p,t) = {}".format(corr))

    plt.legend()

    if result is None:
        plt.show()
    else:
        plt.savefig(result_dir + "/score_over_time_transition.png".format(name))

    plt.clf()


def prediction_over_time(music, speech, model, result_dir=None):
    """
    Plots for some audio files the prediction over time
    Also plots the prediction over time for mixed signals 
    """
    model = Model(inputs=model.input,
                  outputs=model.layers[-2].output)
    pdb.set_trace()
    Y_p = model.predict(np.array([music, speech]))[:,0,:,0]
    print("Y_p", Y_p.shape)

    for name, y_time in zip(["music", "speech"], Y_p):
        print("y_time", y_time.shape)
        plt.plot(list(range(len(y_time))), y_time)
        plt.plot(list(range(len(y_time))), [np.mean(y_time)]*len(y_time), "--")
        plt.title(name)
        if result is None:
            plt.show()
        else:
            plt.savefig(result_dir+"/score_over_time_{}.png".format(name))
        plt.clf()


    timeframes = min(music.shape[1], speech.shape[1])
    music = music[:,:timeframes]
    speech = speech[:,:timeframes]

    linear_transition(music, speech, timeframes*0.2, timeframes*0.8, model)

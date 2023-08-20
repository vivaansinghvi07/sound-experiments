import librosa
import numpy as np
import matplotlib.pyplot as plt

# define types
type_Song = np.ndarray
type_DFT = np.ndarray

# pipeline for fft manipulation
class DFT_Pipeline:

    # add optional starting functions
    def __init__(self, *funcs):
        self.__funcs = [*map(lambda f: [f, []], funcs)]

    # visualize pipeline as string
    def __str__(self):

        # return if empty
        if not self.__funcs:
            return "Empty Pipeline"
        
        # create string from functions and arguments
        output = "Pipeline:\n"
        for i, [func, args] in enumerate(self.__funcs):

            # check if its the end of the pipeline
            if i == len(self.__funcs) - 1:
                output += f"└─"
            else:
                output += f"├─"

            # add function name, and arguments if necessary
            output += f" {func.__name__}"
            if args:
                output += f", args: {' '.join(map(str, args))}"
            output += "\n"
        return output
        
    # add a function to the pipeline with optional arguments
    def add(self, other_func, *args):
        self.__funcs.append([other_func, args])

    # remove a function from the end
    def remove(self, other_func):
        remove_idx = -1
        for idx, [func, _] in self.__funcs:
            if func.__name__ == other_func.__name__:
                remove_idx = idx
        if remove_idx != -1:
            del self.__funcs[remove_idx]

    # go through the pipeline
    def __call__(self, s: type_DFT):
        for [func, args] in self.__funcs:
            s = func(s, *args)
        return s

# loads a song and returns data, rate
def load(songname: str, instr: bool = True) -> tuple[type_Song, float]:
    return librosa.load(f"songs/{songname}{['', '_instr'][instr]}.wav")

# displays a spectograph given stft song data
def disp_specto(s: type_DFT) -> None:
    fig, ax = plt.subplots()
    img = librosa.display.specshow(
        librosa.amplitude_to_db(np.abs(s), ref=np.max),
        y_axis='log', x_axis='time', ax=ax
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.show()

# plot for model
def disp_model_specto(x, sample_rate, show_black_and_white=False):
    x_stft = np.abs(librosa.stft(x, n_fft=2048))
    fig, _ = plt.subplots()
    fig.set_size_inches(20, 10)
    x_stft_db = librosa.amplitude_to_db(x_stft, ref=np.max)
    if(show_black_and_white):
        librosa.display.specshow(
            data=x_stft_db, y_axis='log', 
            sr=sample_rate, cmap='gray_r'
        )
    else:
        librosa.display.specshow(data=x_stft_db, y_axis='log', sr=sample_rate)

    plt.colorbar(format='%+2.0f dB')

# displays the waveform 
def disp_wav(song: type_Song, rate: int, slice: slice) -> None:
    time = np.arange(len(song[slice]))/rate
    fig = plt.gcf()
    fig.set_size_inches(10, 2)
    plt.plot(time, song[slice], lw=1)
    plt.show()
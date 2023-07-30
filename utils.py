import librosa
import numpy as np
import matplotlib.pyplot as plt

# define functions
stft = lambda song: librosa.stft(song)
istft = lambda song: librosa.istft(song)

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
                output += f"  └─"
            else:
                output += f"  ├─"

            # add function name, and arguments if necessary
            output += f" {func.__name__}"
            if args:
                output += f", args: {' '.join(map(str, args))}"
            output += "\n"
        return output
        
    # add a function to the pipeline with optional arguments
    def add(self, other_func, *args):
        self.__funcs.append([other_func, args])

    # go through the pipeline
    def run(self, s: type_DFT):
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

# displays the waveform 
def disp_wav(song: type_Song, rate: int, slice: slice) -> None:
    time = np.arange(len(song[slice]))/rate
    fig = plt.gcf()
    fig.set_size_inches(10, 2)
    plt.plot(time, song[slice], lw=1)
    plt.show()
"""Changing the pace of a finished reading without changing its pitch, for the engines that can't
set a pace themselves.

This used to be librosa's time_stretch, a phase vocoder: it rebuilds the sound from its frequency
content, which smears speech into a metallic echo even at 5 to 10 percent. change_speed() is WSOLA
(waveform similarity overlap-add) instead: the reading is cut into short overlapping pieces that
are laid down closer together or further apart, and each piece is taken from the spot, within a
few milliseconds of where it should come from, that lines up best with the sound already written.
Every sample of the result is a sample of the original, so a voice keeps its own sound.
"""
import numpy as np

WINDOW_SECONDS = 0.032  # one piece: long enough for two or three pitch periods of a low voice
SEARCH_SECONDS = 0.012  # how far a piece may move to line up; a 85 Hz voice has a 12 ms period
MIN_SPEED, MAX_SPEED = 0.5, 2.0


def change_speed(audio: np.ndarray, sample_rate: int, speed: float) -> np.ndarray:
    """audio (mono float32) played `speed` times as fast: 1.1 is 10% faster and shorter."""
    speed = float(min(max(speed, MIN_SPEED), MAX_SPEED))
    window_size = int(sample_rate * WINDOW_SECONDS) // 2 * 2
    search = int(sample_rate * SEARCH_SECONDS)
    audio = np.asarray(audio, dtype=np.float32)
    if abs(speed - 1.0) < 1e-3 or len(audio) < 4 * window_size:
        return audio
    from scipy.signal import correlate

    step = window_size // 2                       # pieces overlap by half in the result
    window = np.hanning(window_size + 1)[:-1].astype(np.float32)  # periodic: halves add up to 1
    source = np.concatenate([np.zeros(search, np.float32), audio, np.zeros(search + 2 * window_size, np.float32)])
    pieces = int(np.ceil(len(audio) / speed / step))
    out = np.zeros(pieces * step + window_size, dtype=np.float32)

    start = search                                # the first piece is the start of the reading
    out[:window_size] = source[start:start + window_size] * window
    for k in range(1, pieces):
        wanted = search + int(round(k * step * speed))   # where this piece should come from
        if wanted + window_size + search > len(source):
            break
        # What would follow the previous piece if nothing were changed: the best piece continues like it.
        follows = source[start + step:start + step + window_size]
        region = source[wanted - search:wanted + search + window_size]
        match = correlate(region, follows, mode="valid", method="fft")
        squares = np.concatenate([[0.0], np.cumsum(region.astype(np.float64) ** 2)])
        energy = np.sqrt(np.maximum(squares[window_size:] - squares[:-window_size], 1e-9))  # of each candidate piece
        start = wanted - search + int(np.argmax(match / energy))
        out[k * step:k * step + window_size] += source[start:start + window_size] * window
    return out[:int(round(len(audio) / speed))]

"""Regenerates docs/pipeline.png (needs matplotlib)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sps

from heartbeam.rppg import bandpass, estimate_bpm, pos
from heartbeam.synth import skin_trace

FS = 30.0
TRUE_BPM = 75
OUT = Path(__file__).resolve().parent.parent / "docs" / "pipeline.png"

rgb = skin_trace(TRUE_BPM, seconds=20, fs=FS, flicker=0.05, noise=0.1, seed=2)
t = np.arange(len(rgb)) / FS
green = bandpass(rgb[:, 1] - rgb[:, 1].mean(), FS)
pulse = bandpass(pos(rgb, FS), FS)

fig, axes = plt.subplots(2, 2, figsize=(11, 6), constrained_layout=True)
fig.suptitle(f"Synthetic face, true heart rate {TRUE_BPM} BPM, lamp flicker 5% (pulse is only 0.3%)")

for ax, wave, title, colour in [
    (axes[0, 0], green, "Green channel only", "tab:green"),
    (axes[0, 1], pulse, "POS projection", "tab:red"),
]:
    ax.plot(t, wave, color=colour, lw=1)
    ax.set_title(title)
    ax.set_xlabel("seconds")
    ax.set_yticks([])

for ax, wave, colour in [(axes[1, 0], green, "tab:green"), (axes[1, 1], pulse, "tab:red")]:
    freqs, psd = sps.periodogram(wave, FS, window="hann", nfft=1 << 14)
    bpm = freqs * 60
    keep = (bpm >= 40) & (bpm <= 200)
    ax.plot(bpm[keep], psd[keep] / psd[keep].max(), color=colour)
    ax.axvline(TRUE_BPM, color="black", ls="--", lw=1, label="true")
    found = estimate_bpm(wave, FS).bpm
    ax.axvline(found, color=colour, ls=":", lw=2, label=f"estimated {found:.0f}")
    ax.set_xlabel("beats per minute")
    ax.set_yticks([])
    ax.legend(loc="upper right")

fig.savefig(OUT, dpi=110)
print(f"wrote {OUT}")

# Data

No audio is committed to this repository: the experiments run on a synthetic
rotor-noise source by default, so everything here is reproducible without any
download.

## Recorded drone audio (optional)

The experiments pick up a recorded sample automatically if one is present at:

    data/drone_bebop_sample.wav

Any mono or multichannel WAV of drone rotor noise works -- it is resampled to
48 kHz and band-passed to 1-3.5 kHz on load. The samples used during the
internship came from the public **drone-audio-detection-samples** corpus
(Parrot Bebop recordings, among others). To use it:

```bash
# The corpus is distributed via git-LFS and is ~2.3 GB
git clone https://github.com/saraalemadi/DroneAudioDataset data/drone-audio-detection-samples
cp data/drone-audio-detection-samples/<some>/<file>.wav data/drone_bebop_sample.wav
```

`data/drone-audio-detection-samples/` is listed in `.gitignore`. Check the
corpus's own licence before redistributing any of its audio; it is not covered
by this repository's MIT licence.

## Why the synthetic source is the default

`arraylib.sources.harmonic_drone` generates a 24-harmonic stack with slow
amplitude modulation, which reproduces the two features of rotor noise that
matter for direction finding: strong energy at blade-passing harmonics, and
beating between rotors. It is deterministic given a seed, so experiment
numbers are repeatable, and it keeps a 2.3 GB download off the critical path
for anyone reading the code.

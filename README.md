# Random Image Sequence (PsychoPy scaffold)

Files added
- `bin/`:
  - `__init__.py`
  - `utils.py` (image discovery, preloading, window and stim helpers)
  - `logger.py` (TSV logger)
  - `generate_sample_images.py` (create sample PNGs)
- `task/`:
  - `random_image_sequence.py` (task runner)
- `requirements.txt`

Generate sample images (for quick testing)
```bash
python bin/generate_sample_images.py --out_dir ./sample_images --num 6 --size 512 512
```

Run the task
```bash
python task/random_image_sequence.py \
  --images_dir ./sample_images \
  --n 6 \
  --duration 0.5 \
  --bg 128 128 128 \
  --output_dir ./logs \
  --seed 42 \
  --fullscreen
```

Notes
- Images are preloaded into RAM with `load_image_arrays()` before any flips, then converted into `ImageStim`s tied to the active `Window`. This minimizes disk I/O during timing-critical presentation.
- Logs are TSV (tab-delimited) at `./logs/image_sequence_log.tsv`. Each row includes:
  - event name (image_on, image_off, fixation_pre_start, fixation_post_start, abort)
  - image name
  - requested duration
  - `flip_time_psychopy_s` (value returned by `win.flip()`)
  - `flip_time_perf_s` (high-resolution `time.perf_counter()`)
  - `end_time_perf_s` (high-resolution `time.perf_counter()` at end event)
- If `--n` is greater than available images, sampling is done with replacement.
- For more precise control (frame-locked scheduling), the code can be extended to use frame counting and scheduled flips instead of `core.wait()`.

Configuration via JSON (required for tasks)
-----------------------------------------
All tasks in this repository must support loading a JSON configuration file as an alternative to specifying parameters via command-line arguments. The config file should allow you to set experiment-level parameters such as:

- `images_dir` (string): path to image resources
- `output_dir` (string): path where logs and metadata will be saved
- `n` (int): number of stimuli to display
- `duration` (number): stimulus presentation duration in seconds
- `isi` (number): inter-stimulus interval in seconds (time between trials)
- `bg` (array of 3 ints): background RGB values in 0-255
- `seed` (int, optional): random seed
- `fullscreen` (bool, optional)
- `win_size` (array of 2 ints, optional)
- `fixation_size` (int, optional)
- `image_size` (array of 2 ints, optional)

Tasks must validate the config when loaded and raise a helpful error if required keys are missing or types are invalid. Command-line arguments should override values in the config file when both are provided.

Example JSON config (`example_config.json`):

```json
{
  "images_dir": "./sample_images",
  "output_dir": "./logs",
  "n": 10,
  "duration": 0.5,
  "isi": 0.2,
  "bg": [128, 128, 128],
  "seed": 42,
  "fullscreen": false,
  "image_size": [512, 512]
}
```

Usage with config file:

```bash
python task/random_image_sequence.py --config example_config.json
```

Or override a config value from CLI:

```bash
python task/random_image_sequence.py --config example_config.json --n 20 --duration 0.4
```

Next steps
- Add subject/run CLI args and include them in the TSV header and meta file.
- Add a small unit test that verifies the logger writes TSV rows.
- Add optional preloading progress reporting for large image sets.

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
- In `active_foraging` and the current image-sequence presentation paths, timing-critical visual sections already use frame-counted `win.flip()` loops rather than `core.wait()`. Remaining `core.wait()` usage is limited to non-visual polling or housekeeping paths and is not used to schedule stimulus onsets/offsets.

CPU Affinity for `active_foraging`
----------------------------------
The `active_foraging` task treats CPU core `0` as the timing-critical presentation core.

- The main `active_foraging` process, including stimulus presentation and touch-event detection, pins itself to CPU `0` before entering the trial loop.
- Non-timing-critical child processes such as the background trial-generation worker and the experimenter preview process inherit the remaining CPU cores.
- This is necessary because `multiprocessing` children inherit the parent's CPU affinity by default. To prevent workers from inheriting CPU `0`, the parent process is first moved onto the non-zero worker-core pool, the child processes are spawned, and then the parent is pinned back to CPU `0`.
- For the intended timing behavior on Linux or Raspberry Pi, CPU `0` should also be isolated from normal OS scheduling at the kernel level, for example with `isolcpus`, `nohz_full`, `rcu_nocbs`, or an equivalent cpuset-based setup.
- Launch the task from a shell or service whose affinity mask still includes CPU `0` and the worker cores. If the launcher has already removed CPU `0` from the process affinity mask, the task cannot pin the main presentation process onto that core.
- Event and message logs for `active_foraging` are buffered during the timing-critical portion of a trial and flushed only in the between-trial gap after `block_end`, so synchronous disk flushes do not run while the initiation cue, stimulus presentation, touch detection, and reward delivery are active.

Active Foraging Timing
----------------------
The main visual timing parameters in `active_foraging` are interpreted by the presentation mode, not as abstract global delays. Those visual timings are quantized to display frames before use. `pump_delay_time` is separate: it is a post-choice reward delay applied in wall-clock seconds before reward delivery begins.

`active_foraging` now validates requested visual timings against the active frame rate before the task starts. If `duration`, `isi`, `choice_time`, or `ibi` is not an exact multiple of the frame duration, the task logs an error and exits instead of silently rounding. It also enforces minimum visible durations: `choice_time` must be at least 1 frame, and when `sequential=true`, `duration` must be at least 1 frame. If you want nominal frame-based timings such as `0.050` at `120 Hz`, set `refresh_rate` explicitly to the intended rate.

- `duration`: stimulus display duration for sequential presentation. When `sequential=false`, `duration` must be exactly `0`; there is no separate timed stimulus-display phase in that mode. Instead, the full array appears on the first choice frame and remains visible for `choice_time` only. In sequential memory mode (`sequential=true`, `is_memory=true`), `duration` is the on-screen time for each individual stimulus in the sequence and choices are not accepted yet.
- `isi`: pre-stimulus cue interval, not a between-trial delay. In simultaneous non-memory mode it shows dots at all candidate locations before the full array appears. In sequential memory mode it shows the dot cue for each item before that item is shown.
- `choice_time`: response-window extension after the stimulus display phase defined by the active mode. In simultaneous non-memory mode the response window starts on the first frame of the full array and lasts `choice_time`, with the full array remaining visible throughout. In sequential memory mode the response window begins only after the full sequence has finished and lasts `choice_time`, with only the remembered dot locations visible.
- `ibi`: inter-block interval after choice handling. This begins only after reward delivery or timeout handling completes; it is not inserted between stimuli within a block.
- `pump_delay_time`: delay in seconds between a rewarded choice being made and the first pump pulse. It applies only on rewarded trials with at least one configured pump pulse, and defaults to `0.0`.

Two common `active_foraging` configurations:

- Config A: `sequential=false`, `is_memory=false`
  - `isi`: all choice-location dots are shown together before the stimuli.
  - `duration`: must be `0`.
  - `choice_time`: the full array appears on the first choice frame and remains selectable for this long.
  - Total selectable time: `choice_time`.

- Config B: `sequential=true`, `is_memory=true`
  - For each option in the block: show that option's dot for `isi`, then show that stimulus for `duration`.
  - After each stimulus disappears, its location remains as a memory dot.
  - After the final stimulus, the task enters a dot-only choice period for `choice_time`.
  - `ibi` starts only after the resulting reward or timeout has finished.

Implementation note: in the current sequential-memory branch, the memory dots are created inside the `isi > 0` path. If `sequential=true`, `is_memory=true`, and `isi=0`, the current code does not create the usual pre-item dot cue / persistent memory dots for that block.

Active Foraging Color TSV
-------------------------
`active_foraging` expects `colors_tsv` to be a tab-delimited file with four columns: `id`, `r`, `g`, `b` (column name case is flexible, for example `ID R G B` also works).

- Include a header row.
- The first data row is treated as the background gray and is not used as a selectable stimulus color.
- Every later row is one displayable color definition with a unique integer ID and integer RGB values.
- Row order matters. After the background row, colors must be ordered by luminance blocks: all `n_colors` base colors for luminance level 1, then all `n_colors` base colors for luminance level 2, and so on.
- The number of non-background color rows must equal `n_colors * n_lum_levels`.

Example:

```tsv
id	r	g	b
0	168	169	166
1	143	115	120
2	142	116	114
3	141	117	108
```

Configuration via JSON (required for tasks)
-----------------------------------------
All tasks in this repository must support loading a JSON configuration file as an alternative to specifying parameters via command-line arguments. The config file should allow you to set experiment-level parameters such as:

- `images_dir` (string): path to image resources
- `output_dir` (string): path where logs and metadata will be saved
- `n` (int): number of stimuli to display
- `duration` (number): stimulus presentation duration in seconds. For `active_foraging`, when `sequential=false`, this must be `0`.
- `isi` (number): pre-stimulus / inter-stimulus interval in seconds; exact meaning is task-specific
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

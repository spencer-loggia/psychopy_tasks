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
- If `--n` is greater than available images, sampling is done with replacement.
- In `active_foraging` and the current image-sequence presentation paths, timing-critical visual sections use frame-counted `win.flip()` loops rather than `core.wait()`. Remaining `core.wait()` usage is limited to non-visual polling or housekeeping paths and is not used to schedule stimulus onsets/offsets.

Logging Output
--------------
All tasks now write session outputs into a dedicated run directory under the configured `output_dir` (normally `./logs`):

`L_[YYYYMMDDHHMMSS]_[task_name]_[config_name]`

Each session directory contains:

- `event_log.tsv`
- `message_log.tsv`
- `behavior_log.tsv`
- `event_code_library.json`

The repo also includes a shared checked-in `event_name_library.json`. It is the repo-wide source of truth for event names, codes, event types, and descriptions across tasks. Not every event in that file is used by every task.

`event_log.tsv` is the deployment-facing timing log shared across tasks. Columns:

- `trial_num`
- `time_since_session_start`
- `event`
- `event_code`
- `event_type`
- `requested_duration`

Common event-log rules:

- `time_since_session_start` is measured from session start using high-resolution real time (`time.perf_counter()`), not separate PsychoPy and perf-counter columns.
- `event_type` is one of `frame_flip`, `interaction`, or `signal`.
- Frame-flip events are logged at the real flip time for the frame that changed the main display.
- Interaction events are logged as close as possible to the touch / click / key / eye-tracker event itself.
- Signal events are logged as close as possible to the GPIO or external signal send.
- Events that have no visible effect or a programmatic zero-duration no-op should not be logged.
- `requested_duration` is filled only when the code requested a fixed duration for that state or signal. Variable windows such as `choice_start` leave it blank.

`event_code_library.json` is generated per session as the minimal subset of `event_name_library.json` that was actually used in that run. It includes:

- integer event code
- event name
- event type
- verbose description

`message_log.tsv` is the non-timing log shared across tasks. Columns:

- `time_since_session_start`
- `level`
- `message`

Allowed levels are `INFO`, `WARN`, and `ERROR`.

`behavior_log.tsv` is task-specific. Every behavior log includes `trial_num` so behavior rows can be aligned with the event log.

Task-Specific Logging
---------------------
`active_foraging` is the most fully specified task and defines the current repo-wide target for event semantics.

Active-foraging event naming:

- Sequential runs (`sequential=true`) log option-specific frame-flip events such as `option_1_dot` and `option_1_on`.
- Simultaneous runs (`sequential=false`) log combined frame-flip events such as `options_dot` and `options_on`.
- `choice_start` marks the first frame where a choice can be made.
- `grey_inter_trial_interval` marks the flip to the gray post-choice / inter-trial screen.
- `cue_touch` and `option_touch` are the current touch interaction events.
- `trial_start_signal_on/off`, `pump_on/off`, and `buzzer_on/off` are the current signal events.
- `event_name_library.json` is shared across tasks. For active-foraging sequential runs it defines `option_{n}_dot` and `option_{n}_on` through templates rather than duplicating a separate static entry for each possible option index.

Active-foraging behavior log columns:

- `trial_num`
- `initiation_time`
- `reaction_time`
- `shape_0 ... shape_(k-1)`
- `color_0 ... color_(k-1)`
- `lum_0 ... lum_(k-1)`
- `choice_made_index`
- `choice_made_color`
- `choice_made_shape`
- `choice_made_lum`
- `reward_level`
- `choice_touch_x`
- `choice_touch_y`
- `choice_reaction_time`

For `active_foraging`:

- `choice_made_index` is zero-based.
- `reaction_time` is the time from `choice_start` until `option_touch`.
- `choice_reaction_time` is currently the same quantity as `reaction_time`, retained because it is part of the requested task-specific schema.

Other tasks use the same session packaging and shared schemas but simpler task-specific behavior rows:

- `random_image_sequence` logs one behavior row per presented image.
- `afc_block_sequence` logs one behavior row per block, including the selected item list and any choice touch.
- `play_video` logs one behavior row per played clip and only logs video clip start / expected duration / end in the event log, not every frame.

Screen Selection
----------------
Multi-screen tasks use `screens.main` for the subject display and `screens.experimenter` for the secondary display.
Each value can be a detected screen index or an output name such as `HDMI-1` or `DSI-1`.
Set either value to `null` to inherit the process environment defaults: `screens.main` reads `MAIN_SCREEN`, and
`screens.experimenter` reads `SECONDARY_SCREEN`. The touch launcher exports its resolved global `screens` values
to those environment variables for launched tasks.
For `active_foraging`, setting `screens.main` and `screens.experimenter` to the same display is allowed and disables
the experimenter preview, so only the main task content is shown.

Eye Tracker Calibration
-----------------------
`task/calibrate_eye_tracker.py` calibrates two analog eye-position voltages from a Pi-Plates DAQC2plate.
The reusable implementation lives in `bin/eye_tracking.py` so other tasks can consume the same smoothed,
calibrated eye position.

Run with a config:

```bash
python task/calibrate_eye_tracker.py --config test_configs/eye_calibration_config.json
```

Important config keys:

- `screens.main` and `screens.experimenter`: main subject display and experimenter display, using the same selector rules as `active_foraging`.
- `daq.address`, `daq.x_channel`, `daq.y_channel`: DAQC2plate address and analog input channels. The bundled DAQC2plate guide documents `piplates.DAQC2plate.getADC(addr, channel)` for channels `0` through `7`; channel `8` is the board supply readback and is not used for eye position.
- `daq.sample_rate_hz`: analog sampling rate, default `240`.
- `daq.voltage_min` and `daq.voltage_max`: expected valid eye-position voltage range, default `-10.0` to `10.0`.
- `eye_filter.ema_gamma`: exponential moving-average gamma, default `0.98`.
- `eye_filter.max_voltage_step`: optional per-sample jump threshold for blink/artifact rejection. Set to `null` to disable step rejection.
- `initial_x_scale`, `initial_y_scale`, `initial_x_offset`, `initial_y_offset`: starting voltage-to-screen mapping parameters.
- `fix_diameter`: fixation acceptance diameter, as a fraction of the shorter main-screen dimension. Default `0.05`.
- `fix_accept_percent`: proportion of recent frame samples that must be inside the fixation window before automatic reward. Defaults to `0.95`; values like `95` are also accepted.
- `fix_accept_time`: rolling acceptance-window duration in seconds. Default `2.0`.
- `pump_pin` and `pump_pulse_time_seconds`: manual reward output controlled by the green experimenter-screen button.

The eye tracker reports centered screen fractions relative to the real main screen dimensions:
`x=-0.5` is the left edge, `x=0.5` is the right edge, `y=-0.5` is the bottom edge, and `y=0.5` is the top edge.
The experimenter screen draws a gray preview box with the same aspect ratio as the main screen; the blue eye dot
and fixation cross are mapped into that box. Clicking inside the box moves the fixation cross on both screens.
The bottom slider changes `x_scale`, the left slider changes `y_scale`, and the lower-left `x` button sets the
offsets so the current smoothed eye position maps to the current fixation position.
When the smoothed eye position stays within `fix_diameter` of the fixation cross for at least `fix_accept_percent`
of the past `fix_accept_time`, the task delivers one automatic `pump_pulse_time_seconds` reward. That automatic
reward re-arms only after fixation is broken or the fixation/calibration target is changed.

On exit, the task writes `[YYYYMMDDHHMMSS]_eye_calibration.json` directly under `output_dir` with `x_scale`,
`y_scale`, `x_offset`, and `y_offset`, plus DAQ/filter metadata. A normal session log directory is also created
under `output_dir` for messages and pump signal events.

CPU Affinity for `active_foraging`
----------------------------------
The `active_foraging` task treats CPU core `0` as the timing-critical presentation core.

- The main `active_foraging` process, including stimulus presentation and touch-event detection, pins itself to CPU `0` before entering the trial loop.
- Non-timing-critical child processes such as the background trial-generation worker and the experimenter preview process inherit the remaining CPU cores.
- This is necessary because `multiprocessing` children inherit the parent's CPU affinity by default. To prevent workers from inheriting CPU `0`, the parent process is first moved onto the non-zero worker-core pool, the child processes are spawned, and then the parent is pinned back to CPU `0`.
- For the intended timing behavior on Linux or Raspberry Pi, CPU `0` should also be isolated from normal OS scheduling at the kernel level, for example with `isolcpus`, `nohz_full`, `rcu_nocbs`, or an equivalent cpuset-based setup.
- Launch the task from a shell or service whose affinity mask still includes CPU `0` and the worker cores. If the launcher has already removed CPU `0` from the process affinity mask, the task cannot pin the main presentation process onto that core.
- Event, message, and behavior logs for `active_foraging` are buffered during the timing-critical portion of a trial and flushed only in the between-trial gap, so synchronous disk flushes do not run while the initiation cue, stimulus presentation, touch detection, and reward delivery are active.

Active Foraging Timing
----------------------
The main visual timing parameters in `active_foraging` are interpreted by the presentation mode, not as abstract global delays. Those visual timings are quantized to display frames before use. `pump_delay_time` is separate: it is a post-choice reward delay applied in wall-clock seconds before reward delivery begins.

`active_foraging` now validates requested visual timings against the active frame rate before the task starts. If `duration`, `isi`, `choice_time`, or `ibi` is not an exact multiple of the frame duration, the task logs an error and exits instead of silently rounding. It also enforces minimum visible durations: `choice_time` must be at least 1 frame, and when `sequential=true`, `duration` must be at least 1 frame. If you want nominal frame-based timings such as `0.050` at `120 Hz`, set `refresh_rate` explicitly to the intended rate.

- `duration`: stimulus display duration for sequential presentation. When `sequential=false`, `duration` must be exactly `0`; there is no separate timed stimulus-display phase in that mode. Instead, the full array appears on the first choice frame and remains visible for `choice_time` only. In sequential memory mode (`sequential=true`, `is_memory=true`), `duration` is the on-screen time for each individual stimulus in the sequence and choices are not accepted yet.
- `isi`: pre-stimulus cue interval, not a between-trial delay. In simultaneous non-memory mode it shows dots at all candidate locations before the full array appears. In sequential memory mode it shows the dot cue for each item before that item is shown.
- `choice_time`: response-window extension after the stimulus display phase defined by the active mode. In simultaneous non-memory mode the response window starts on the first frame of the full array and lasts `choice_time`, with the full array remaining visible throughout. In sequential memory mode the response window begins only after the full sequence has finished and lasts `choice_time`, with only the remembered dot locations visible.
- `ibi`: inter-block interval after choice handling. This begins only after reward delivery or timeout handling completes; it is not inserted between stimuli within a block.
- `pump_delay_time`: delay in seconds between a rewarded choice being made and the first pump pulse. It applies only on rewarded trials with at least one configured pump pulse, and defaults to `0.0`.
- `pump_pulse_time_seconds`: duration in seconds that the pump output remains on for each reward pulse.
- `inter_pump_interval`: delay in seconds between repeated pump pulses. When omitted, it defaults to `pump_pulse_time_seconds`, preserving the previous behavior.

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

Active Foraging Positioning
---------------------------
`active_foraging` places every stimulus center on a stimulus circle in main-screen pixel coordinates. `center_point` is `[x, y]` with origin at the upper-left of the main screen. When `center_point` is `null`, it defaults to the exact middle of the main screen. `stim_range_radius` is the circle radius in pixels. When it is `null`, it defaults to half the distance from `center_point` to the closest screen edge.

- `fixed_positions=true`: locations are evenly spaced around the circle. The spacing angle is `2*pi / num_afc`, and the first location is offset by half that spacing from the point directly below `center_point`.
- `fixed_positions=false`: locations are random points on the circle, with rejected draws when stimulus bounding boxes would overlap.
- Custom `center_point` and `stim_range_radius` values can be provided in JSON or as `--center_point X Y --stim_range_radius R`.

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
- `center_point` (array of 2 ints or null, optional): `active_foraging` stimulus circle center in main-screen pixels
- `stim_range_radius` (int or null, optional): `active_foraging` stimulus circle radius in pixels

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

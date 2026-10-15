# Neuro Tasks

Repository layout

- `bin/`: shared presentation, timing, hardware, screen, and TSV logging helpers.
- `task/`: runnable tasks, including `active_foraging.py`, `match2cue.py`, `afc_csc1.py`, `afc_trial_sequence.py`, standalone image/video stimulus tasks, and hardware setup utilities.
- `config_files/`: deployment configurations.
- `test_configs/` and `tests/`: development configurations and automated tests.

Experiment Manager
------------------

Launch the touch interface with an interface configuration:

```bash
python interface/touch_interface.py --config config_files/interface/rpi_launch_config.json
```

The interface opens on a root menu with Start Experiment, Run System Diagnostic, Shell, the context-dependent Rig
Mode/Portable Mode switch, Desktop, and Shutdown actions. Start Experiment opens subject selection. The launch
config's `subjects` object maps each displayed full name to the short subject code used in directory names.
Selecting a subject starts a new experiment under `logs`:

`exp_[subject_code]_[YYYYMMDD]_[incrementing_id]`

For example, Yuri's first experiment on August 13, 2026 is `logs/exp_Y_20260813_001`. The experiment directory
contains:

- `launch_config.json`: a snapshot of the interface launch config.
- `event_name_library.json`: a snapshot of the shared event-name/code library used by its blocks.
- `state.json`: the current mutable experiment state.
- `blocks.tsv`: blocks in launch order, including start and end times in milliseconds from experiment start.
- `blocks/[block_num]_[block_name]/`: the generated config and outputs for each block.

`blocks.tsv` has the fields `block_name`, `block_num`, `subject`, `start_time`, `end_time`, and `out_dir`.
`block_name` is the task config's `config_name`, falling back to the interface button name for a task with no
source config. A row is written with a blank `end_time` before the subprocess starts and is updated when it exits,
so an interrupted or failed block remains visible in the experiment record.

Both Raspberry Pi launch configs also synchronize experiment boundaries with the tracker Pi:

```json
"eye_tracker_experiment": {
  "enabled": true,
  "uart": {"device": "/dev/ttyAMA10", "baud": 460800}
}
```

Selecting a subject queues UART `START_EXPERIMENT` (code 6) with the selected full name
(for example, `Yuri`, not its directory code `Y`) and the experiment start datetime converted
to Unix UTC nanoseconds. End Experiment or closing the launcher queues `STOP_EXPERIMENT`
(code 7), which asks the tracker to flush recording and close its experiment. The tracker
must run its `pi_tracker.py CONFIG` interface with permission to set its system clock;
standalone `run`/`preview` services do not implement these experiment commands. Subject
names must match `[A-Za-z0-9][A-Za-z0-9_.-]{0,31}`; unsupported names produce a warning
and are never silently rewritten.

These commands run in order on a background worker, with the existing 10-second state
timeout and one identical-packet retry by default (`uart.state_timeout_s` / `state_retries`).
The UART is closed after each command. Task eye acquisition waits for the experiment-start
handoff in its own worker, so the launcher and task rendering remain responsive.
Missing hardware, rejected commands, unexpected status flags, or recording failures show
nonmodal warning dialogs and write warnings to stderr; experiments and tasks continue.
Repeated warnings of the same kind are suppressed within each session. Missing recording
does not invalidate otherwise valid gaze, but stopped, unconfigured, calibrating, stale,
or error telemetry is unusable. Local eye CSV failures also warn without aborting the task.

Experiment synchronization defaults to disabled when this launcher section is absent;
set `enabled` to `false` (or `uart.simulate` to `true`) for hardware-free operation.
With synchronization enabled, the interface's **Shutdown** button also sends binary UART
`SHUTDOWN` (code 8), even after End Experiment has returned to the root menu. It runs
after data cleanup is accepted and after any queued experiment commands. The interface
stays responsive while awaiting the matching `SHUTTING_DOWN` acknowledgement or the
configured timeout/retries, then requests local power-off. Tracker failures show a warning
and still allow local shutdown; cancelling data cleanup sends no shutdown command.
The tracker acknowledgement confirms application cleanup and acceptance of the request,
not that its operating system has finished powering off. The tracker process must have
permission to request system power-off.
The launcher UART device/baud must match the task's UART settings and physical wiring.
Experiment start opens the tracker session; each eye-enabled task still controls tracking
through `START_TRACKING` / `STOP_TRACKING`.

### `config_name` convention

`config_name` is a stable analysis-family identifier, not a unique name for one configuration or run. Assign the
same value as broadly as possible to tasks whose trial records have compatible meanings and would normally be
pooled for analysis. For example, all active-foraging configurations in the same analysis family should retain
one `config_name` across sequential and simultaneous presentation, memory and non-memory variants, different AFC
counts, stimulus durations, other timing values, and display changes. Downstream analysis should distinguish
those variants with their specific config fields (for example `sequential`, `is_memory`, `num_afc`, or
`duration`) from the block's generated `config.json`. Subject names, dates, rig names, and labels such as
`test`, `memory`, or `delay` should likewise not be added merely to make the name unique. Use a different
`config_name` only when the task's trial meaning or fields are incompatible enough that its trials should not be
combined with the existing family. Config filenames and interface menu labels can remain specific and descriptive.

The launch config must contain `subjects`, `tasks`, and `initial_state` objects. `initial_state` is copied into
`state.json`, then its `subject` field is set to the selected full subject name. Eye calibration uses this state
shape:

```json
{
  "subject": null,
  "eye_tracker_calibration": [
    {
      "x_scale": null,
      "y_scale": null,
      "x_offset": null,
      "y_offset": null,
      "set_time": 0.0
    }
  ]
}
```

After every block, the manager checks its directory for `calibration.json`. For each top-level field that also
names a list-valued state field, the calibration object must contain all fields in that state's most recent entry
except `set_time`. The manager then appends those values and supplies `set_time` in milliseconds from experiment
start. Extra calibration metadata is ignored. This generic matching rule allows future calibration histories to be
added to `initial_state` without adding task-specific manager code.

Before each block, the manager creates its output directory and writes `config.json` there. It copies the task
config from its original source at block launch time, sets `config_name`, `output_dir`, and the full-name `subject`, copies
other non-list state fields, and flattens the newest entry from every list-valued state field into top-level config
fields, excluding `set_time`. It never injects the calibration-history lists themselves. The task subprocess is
then launched with `--config` pointing to this generated file.

Interface task entries can be nested into menu pages. A normal leaf has one launch script and an optional config:

```json
{
  "launch": "task/active_foraging.py",
  "config": "config_files/csc2/example.json"
}
```

A loop leaf uses equal-length `launch` and `config` lists:

```json
{
  "launch": ["task/active_foraging.py", "task/active_foraging.py"],
  "config": ["config_files/csc2/classic.json", "config_files/csc2/memory.json"],
  "order_mode": "sequential",
  "n_iters": null
}
```

`n_iters` counts launched blocks, not complete passes through the list. `sequential` wraps to the first item after
the last; `random` samples one item independently for each block. `null` runs indefinitely, while a non-negative
integer limits the number of blocks. A natural block completion (exit `0`) advances the loop. A user exit stops the
current block and its loop immediately.

The subject and task option area scrolls with a mouse wheel (including X11 Button-4/5 wheel events) or by dragging
vertically with one finger. A touch drag must move at least 12 pixels before it becomes a scroll, so a normal tap
still activates its button while a swipe does not accidentally launch it.
All menus also support the Up and Down arrow keys to move the blue selection outline and Enter to activate the
selected button. Selection wraps at the ends of the menu and scrolls into view automatically, including on subject
selection and nested task pages. Mouse and touch controls remain available.

The root menu and top-level task menu both have a Shell button. On the Raspberry Pi it opens a new LXTerminal on
the experimenter display, starts in `environment.working_dir`, and activates the virtual environment containing
`environment.python`. The top-level task menu also has an End Experiment button. It closes the current experiment
and returns to the root menu; Desktop, Shutdown, and mode switching are available only from that root menu.

Data synchronization
--------------------

The interface config's `remote_data_url` is the NFS-backed storage root. The contents of the local `logs`
directory are copied into `remote_data_url/experiments/`, not directly into the storage root. Cleanup runs when
**End Experiment** is pressed and before shutdown; it does not run at startup or after individual tasks or blocks.
Cleanup checks both `remote_git_url` and `remote_data_url`, including that the data root is a mounted, non-empty NFS
filesystem. An unavailable remote, failed operation, or timeout is logged and normal interface operation continues,
with local experiment data left untouched unless synchronization completes successfully.

When storage is available, the launcher uses `rsync --archive --timeout=30 --progress --no-owner --no-group`
to copy the contents of `logs`. In addition to rsync's own I/O timeout, the launcher stops a transfer after 60 seconds
without progress; active transfers have no total-duration limit. Owner and group metadata are intentionally not
preserved because NFS root-squashing commonly rejects those changes; experiment contents, timestamps, permissions,
directories, and symlinks are still preserved. A modal progress window prevents tasks from launching during the copy
and offers **Cancel Sync**. Cancelling aborts the current operation without removing local data; during shutdown, it
also cancels shutdown. Any rsync failure retains every local experiment. After a fully successful copy, all but the
most recently modified experiment are removed from the local cache. Generated data directories are excluded by
`.gitignore`.

Code updates use `remote_git_url` from the same interface config (normally an NFS-mounted repository path). At
startup the launcher makes a bounded update attempt; a failure or timeout is logged without preventing the interface
from opening. Each later cleanup also runs `git reset --hard` and pulls from that explicit URL before data
synchronization starts.

Only shutdown turns a network-cleanup failure into a blocking prompt: **Data network not available. Try again?**
**Yes** retries cleanup, while **No** proceeds with shutdown.

Run System Diagnostic uses the configured `environment.python` interpreter and does not create an experiment or
block. It checks that PsychoPy can import, `lgpio` can open GPIO chip 0, and the Pi-Plates DAQC2 driver can read the
board supply voltage on ADC channel 8. It creates the GL/window stack on worker cores, waits until the launcher has
uncovered and released the main window, then pins the timing process to CPU 0. The diagnostic reads the resolved
main output's starred active-mode refresh rate from `xrandr`, requests blocking vsync, confirms that the X11/GLX
driver stored swap interval 1, and records 120 low-overhead PsychoPy flip intervals. PsychoPy's implicit
constructor timing test is disabled; the reported observed rate comes from this same controlled flip population
and is never substituted for the main output's xrandr hardware rate. When `GLX_OML_sync_control` is available, the
report also includes GLX retrace and completed-swap deltas. The flip-lock check passes when at least 90% of those
intervals match one hardware refresh and the median interval also matches the expected frame period. The
completion dialog always includes the monitor refresh rate (or `unavailable`) and a
per-check list, followed by explicit errors for failed checks. If native fullscreen placement fails but a window
exists, vsync and flip timing still run. Results on an identified different output are labeled
`WRONG-OUTPUT TIMING ONLY` and use that output's independently queried xrandr rate. An unmatched window uses the
main output's rate only as an explicitly labeled fallback reference. Only window-creation failure skips timing.

The DAQC2 check uses board address 0 by default. A rig with a different address or module name can set optional
top-level launcher settings:

```json
{
  "diagnostic": {
    "daq_address": 0,
    "daq_module": "piplates.DAQC2plate"
  }
}
```

Experiment-managed task/subprocess policies:

- Every launchable script must accept `--config`. This includes setup utilities, even if they use only a few fields.
- Natural completion returns process exit code `0`. Escape, an exit button, Ctrl-C, or another explicit user stop
  returns `130`. Other nonzero codes mean failure. The manager stops a loop for either a user stop or failure.
- Tasks using `SessionLogBundle` automatically write directly into the manager-created block directory when the
  manager sets `NEURO_TASK_SESSION_DIR`; they must not add another session directory.
- The manager sets `NEURO_EVENT_NAME_LIBRARY` so block event-code exports are resolved from the experiment's
  snapshot rather than a subsequently edited repository file.
- Manager-generated block configs force `fullscreen=true`; portable windowed runs remain a standalone debugging
  option and are not used by the Raspberry Pi interface.
- A state-producing task writes `calibration.json` in its block directory. Its top-level keys are state-field names,
  and each value is an object containing at least the subfields declared by that state history.
- When main-input masking is enabled, a task must signal readiness after its subject window exists but before
  timing-sensitive work. The call waits for the launcher to remove the curtain; the task then activates the
  window. `bin.utils.setup_task_window()` is the privileged and mandatory entry point for every subject-facing
  task window. Task code must never construct `visual.Window`, resolve its own main output, or call
  `setup_window()`, `initialize_psychopy_window()`, or `open_psychopy_window()` for the subject window. This one
  path owns canonical selector resolution, launcher readiness, Pyglet output binding, refresh synchronization,
  native activation, and final realized-rectangle verification. On Linux it also makes the resolved subject
  output XRandR-primary, confirms swap interval 1 on that drawable, and restores the previous primary output when
  `bin.utils.close_task_window()` closes it. Every subject task must use this matching setup/close pair.
  Secondary calibration controls use the shared `ExperimenterPreview` process.
  The diagnostic and preview process are the only other low-level window-factory users. `tests/test_screen_path_policy.py` enforces this invariant for every current task and
  rejects future direct window-factory bypasses.

X11 Main-Screen Idle Masking
----------------------------

On a dual-screen Linux/X11 rig, the experiment manager can prevent touches on the monkey-facing main screen from
moving the desktop pointer or taking focus while the interface is idle. Configure these top-level launch-config
fields:

```json
{
  "screens": {
    "main": "HDMI-1",
    "experimenter": "DSI-1"
  },
  "mask_main_inputs": true,
  "main_touchscreen_xinput": "Exact XInput Touchscreen Name"
}
```

- `mask_main_inputs` defaults to `false` and must be a JSON boolean.
- When it is `true` and the resolved main and experimenter displays differ, `main_touchscreen_xinput` is required.
  It must be the exact stable device name reported by `xinput list --name-only`; numeric XInput IDs are rejected
  because they can change after reboot.
- When both screen selectors resolve to the same display, masking is skipped completely for debugging and
  `main_touchscreen_xinput` is not required.
- Masking requires Linux with a real X11 `DISPLAY`. It intentionally does not support Xwayland.

While the interface is idle, the manager runs `xinput disable` only for that named touchscreen and covers the main
display with a fullscreen black window. The experimenter touchscreen and physical mouse are not modified. A task
subprocess creates its main window, then emits a private readiness marker. The manager removes the black window,
maps the touchscreen to the resolved main RandR output with `xinput map-to-output`, enables it, and acknowledges
release. The task waits for that acknowledgment and then activates its window before beginning timing-sensitive
work.
After every block—including errors and user exits—the manager disables the touchscreen again, restores the black
window, and returns focus to the interface. Choosing **Desktop**, closing the interface window, or otherwise
leaving the interface normally re-enables the touchscreen before exit.

Experiment Quiet Mode
---------------------

The Raspberry Pi launcher enables `experiment_quiet_mode` when a subject is selected and restores normal activity
when **End Experiment** is pressed or the interface exits. While active, the launcher does not perform its network
time check or `git pull`, hides itself while a task owns the displays, and blocks without polling once the task
window is ready. It also stops active systemd maintenance timers for APT updates, man-db, log rotation, filesystem
trimming/scrubbing, cron/anacron, and time synchronization. Only units that were active are restarted.
Task-owned workers, preview processes, hardware services, networking, input, logging, and the window manager are not
stopped.

Set `"experiment_quiet_mode": true` in the launcher config to use the maintained unit list. A rig can replace it
with an explicit list:

```json
{
  "experiment_quiet_mode": {
    "systemd_units": ["cron.service", "apt-daily.timer"]
  }
}
```

The interface user must be root or have non-interactive `sudo` permission to start and stop the listed units. If
quiet mode cannot be entered, experiment creation fails instead of silently running without the requested guard.
An already-running APT update is not terminated; experiment creation waits by failing clearly until that package
operation finishes.

If input remapping fails after a task starts, the manager stops that task, restores the guarded idle state, records
the block end, and reports the launch failure. An exact touchscreen name is rig-specific; enable masking only after
copying the name from that rig's `xinput list --name-only` output.

Behavioral Terminology
----------------------

- A `block` is a contiguous chunk of an experiment dedicated to one task and stimulus type. Blocks are experiment-level units and are not repeated choice cycles inside a task.
- A `trial` is one complete behavioral cycle. In active foraging this is initiation cue, option presentation, choice, reward or timeout, and the inter-trial interval. In match-to-cue it is initiation cue, match cue, delay, option presentation, choice, and inter-trial interval. In standalone image and video tasks, one presented image or clip is one trial.
- An option presentation within an AFC trial is represented by its frame-flip event. It does not receive a separate numeric identifier.
- `trial_num` identifies the enclosing trial for every task.

Logging Output
--------------
When run directly, tasks write session outputs into a dedicated run directory under the configured `output_dir`
(normally `./logs`):

`L_[YYYYMMDDHHMMSS]_[task_name]_[config_name]`

When run by the experiment manager, the same files are written directly into
`exp_.../blocks/[block_num]_[block_name]`; no `L_...` directory is added.

Each session directory contains the applicable files from this set:

- `event_log.tsv`
- `message_log.tsv`
- `behavior_log.tsv` when the task records trial-level behavior
- `event_code_library.json`
- `calibration.json` when the task produces runtime state that can affect a future block or task

`calibration.json` is the project-wide filename for optional runtime task output that may affect future experiment
state. It must be written inside the task's session directory. Tasks that do not produce such state do not create
the file.

The repo also includes a shared checked-in `event_name_library.json`. It is the repo-wide source of truth for event names, codes, event types, and descriptions across tasks. Not every event in that file is used by every task.

`event_log.tsv` is the deployment-facing timing log shared across tasks. Columns:

- `trial_num`
- `requested_time_since_session_start`
- `time_since_session_start`
- `event`
- `event_code`
- `event_type`
- `requested_duration`

Common event-log rules:

- `requested_time_since_session_start` records when a refresh-locked flip or hardware change was submitted; it is blank for events that have no separate request time.
- `time_since_session_start` records when that flip or hardware operation actually completed, measured from session start using high-resolution real time (`time.perf_counter()`).
- `event_type` is one of `frame_flip`, `interaction`, or `signal`.
- Frame-flip events are logged at the real flip time for the frame that changed the main display.
- Interaction events are logged as close as possible to the touch / click / key / eye-tracker event itself.
- Signal events are logged as close as possible to the GPIO, DAQC2, or external signal send.
- Events that have no visible effect or a programmatic zero-duration no-op should not be logged.
- `requested_duration` always retains the exact configured request, not the nearest frame-achievable duration. It is blank only when no duration was requested.

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

`behavior_log.tsv` is task-specific. Every behavior log includes `trial_num` so behavior rows can be aligned with `event_log.tsv`.

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

For `match2cue`, the behavior log additionally records the match cue, whether that displayed match cue was tapped,
the number of matching options, the tie mode, whether the choice was correct, the resulting choice-reward
probability and pulse count, and whether match-cue and choice rewards were delivered. A no-response trial is left
blank for correctness and does not increment either experimenter-screen correctness counter.

Other tasks use the same session packaging and shared schemas but simpler task-specific behavior rows:

- `random_image_sequence` treats each image presentation as a trial and logs one behavior row per image.
- `afc_trial_sequence` logs one behavior row per trial, including the option list and any choice touch.
- `play_video` takes an explicit `video_files` list, a required
  `clip_duration_seconds`, and a positive `num_clips`. Subject-screen taps do
  not interrupt playback; the task completes after `num_clips` clips. Each
  trial randomly selects one source, uniformly selects a valid temporal start
  aligned to that source's PTS origin and configured video-frame timebase, and
  presents the fixed-duration clip without extracting a temporary file. A
  dedicated `multiprocessing` `spawn` worker uses ffpyplayer directly to seek,
  decode, center-crop, and prepare that clip. `frame_rate` defaults to `30` and must
  match the probed source rate; rate conversion belongs in offline
  preprocessing so runtime playback never synthesizes or discards a source
  frame. The requested duration is rounded only to the nearest count of video
  frames; both the original request and scheduled duration are logged.
- Sources must be HEVC Main/yuv420p. The task probes each unique path once and
  refuses incompatible media or sources shorter than the requested clip.
- The behavior row for each completed or user-aborted trial records the full
  source path, requested and actual source timestamps, first-frame
  flip-completion time, last-frame end flip-completion time,
  requested and scheduled duration, configured rate, scheduled/displayed frame
  counts, and displayed duration. The corresponding event-log start/end records
  are main-display flips; the end flip removes the last frame. A fatal timing
  or decoder error writes an invalid attempted-trial behavior row and preserves
  exact onset, clear, and diagnostic records in the event/message logs. These
  timestamps describe flip completion near vertical blanking; use the existing
  GPIO pulse and a photodiode when physical pixel illumination time is needed.
- After the clear flip, `video_frame_timing.tsv` records every source-frame
  index and media PTS, requested/ideal/actual flip-completion times, signed
  timing error, and planned/realized refresh hold. No timing-row file I/O occurs
  during playback. `video_frame_validation` adds a one-line `PASS`, `WARN`, or
  `FAIL` summary to `message_log.tsv`, including PTS continuity, frame counts,
  median/p95/maximum boundary error, planned and realized hold histograms, and
  cadence mismatches. The same summary status and key error values are present
  in `behavior_log.tsv`.
- The shared subject-window lifecycle owns Linux/Pi XRandR-primary selection,
  main-drawable swap-interval validation, and restoration for every task.
  `play_video` revalidates that common contract after decoder/preview startup.
  Each completed clip additionally logs
  `video_main_vblank_validation`, comparing the drawable's actual GLX OML media
  stream counter (MSC) and swap buffer counter (SBC) deltas with the exact
  planned subject refreshes and source-frame swaps. These values are also saved
  in `behavior_log.tsv`. The experimenter preview runs in another process and
  synchronizes only its own drawable to its own output; it cannot become the
  subject window's swap reference. If post-preview main-output validation fails,
  playback never becomes ready.
- Playback uploads, draws, and calls blocking PsychoPy `Window.flip()` only at
  source-frame boundaries. Between boundaries the front buffer is left
  untouched, so the display naturally repeats that image at each hardware
  refresh without more Python or GL work. Every boundary is scheduled from
  the frame-zero onset plus its absolute source time; the blocking swap maps
  it to the nearest VBL. The measured monitor rate supplies the submission
  margin and cadence diagnostics, but its small measurement error is never
  multiplied across the clip. Thus 59.94/30 uses mostly two-refresh
  holds with an occasional one-refresh correction, while 60/24 uses a balanced
  three/two cadence. The phase error remains bounded to half a refresh instead
  of accumulating over the clip, and every source frame is still displayed at
  least once. A combination that would assign zero refreshes to any source
  frame is rejected. One coarse sleep positions each submission after the
  preceding VBL; the blocking flip then synchronizes to its absolute target
  VBL. Every realized boundary is checked against the onset-anchored schedule.
  A boundary that quantizes to an adjacent VBL is recorded as a timing miss,
  after which playback continues toward the next absolute boundary without
  skipping or substituting a source frame. An unavailable prepared chunk
  remains fatal. The exact hold histogram, total refresh count, realized
  boundary errors, and maximum/final phase errors are logged.
- A noninteger display/video ratio is valid but is not perceptually uniform:
  for example, 23.976/24 fps on a 60 Hz output alternates two- and
  three-refresh holds (3:2 pulldown). This produces visible judder in fast pans
  even when `video_frame_validation` passes. The task emits a separate
  `video_motion_cadence` warning with exact integer-multiple refresh-rate
  candidates; use a supported mode such as 119.88 Hz for 23.976 fps when
  uniform motion is required.
- The ffpyplayer worker writes prepared RGB24 frames into parent-owned shared
  memory. The prepared-frame budget is configured by
  `video_buffer_megabytes` and defaults to 512 MiB. If the complete prepared
  clip is no longer than one 0.25-second chunk, it is ready before playback in
  one slot. Otherwise, the pool has exactly three approximately 0.25-second
  chunks; two (about 0.5 seconds of media) are ready before playback, and the
  remaining slot lets the worker stay ahead by refilling released slots as the
  main process consumes prepared chunks. This overlaps preparation with display
  even when a longer clip would fit entirely in RAM, substantially reducing
  onset latency and memory footprint. No pixel data
  crosses a queue; queues carry only ownership, timestamp, status, and error
  messages. The main process reads each frame in place across chunk boundaries
  and uploads it once into a persistent GL texture; hardware refreshes between
  source boundaries require no redraw, copy, or pixel re-upload.
- `play_video` decodes each clip once. No faster than once every 0.1 seconds,
  the main process copies the currently displayed RGB24 frame into a four-slot,
  latest-frame-wins shared-memory preview ring. The experimenter process copies
  the newest committed slot into a private snapshot before its own GL upload;
  these consistency copies do not create a second decoder or a delayed frame
  queue. Each preview frame includes its sequence, source frame/media time,
  main-display flip time, and trial number.
- Each completed attempt emits a `video_pipeline_performance` message with
  mean/maximum shared-buffer fetch, RGB texture upload, draw submission, and
  preview-copy costs plus the minimum source-boundary submission headroom.
  `video_playback` separately reports seek/preload time and its effective
  preparation rate. These measurements distinguish decoder starvation from a
  slow GL upload or an overloaded experimenter mirror on the deployed Pi.
- The preview is a subject-view mirror: it transposes the complete native main
  framebuffer geometry (background, positions, stimuli, hit boxes, and video
  crop), rather than merely undoing each stimulus orientation. Scene commands
  use a bounded, sequenced latest-snapshot mailbox, and shared video reads the
  newest committed ring frame, so a slow preview drops stale work instead of
  accumulating lag.
- During movie playback the experimenter overlay shows a separate `Video:`
  elapsed timer. Its zero is the actual main-display first-frame flip callback,
  independent of decoder preparation time and preview transport latency.
- A fresh ffpyplayer worker owns each selected clip and exits when preparation
  is complete. For efficient random access over a mounted network filesystem,
  preprocess sources with `bin/preprocess_videos.py --frame_rate RATE`, where
  `RATE` is the source's exact native CFR and the task's configured
  `frame_rate` (for example, use `23.976023976` for `24000/1001`).
  Preprocessing center-crops and Lanczos-scales the video, automatically
  deinterlaces an interlaced input, rebases PTS to zero, encodes progressive
  HEVC Main/yuv420p, enables MP4 fast-start, disables B-frames, and uses a
  default 0.5-second maximum keyframe interval. The quality-oriented defaults
  are libx265 preset `medium`, CRF 18, and no `fastdecode` tune because that tune
  disables HEVC loop filters and can expose blocking/ringing in fast motion.
  The script refuses a source/output frame-rate mismatch by default, since the
  FFmpeg `fps` filter must duplicate or drop frames and can create periodic
  motion blips. `--allow_frame_rate_conversion` is the explicit opt-in when
  that motion change is intentional. Runtime playback performs no frame-rate
  conversion. Pass the subject-view dimensions (after output rotation) to
  `--screen_size`; runtime cropping cannot avoid decoding pixels that should
  have been removed offline. Existing outputs must be regenerated with
  `--overwrite` from the original master files (not a previously rate-converted
  output) to acquire the newer scale/quality/interlace guarantees.
- Each main-window RGB upload explicitly unbinds any pixel-unpack buffer,
  restores packed-client-memory pixel-store state, and checks the resulting GL
  error. This prevents stale context-global OpenGL state from turning an
  otherwise correctly timed upload into a shifted, partial, or transiently
  corrupted frame.
- On Raspberry Pi, `play_video` sends one-video-presentation-frame sync pulses on BCM GPIO
  `sync_pin` (default `18`). Pulse onsets are frame locked and their successive
  intervals are sampled inclusively from `sync_interval_frames` (default
  `[100, 300]`) for each interval. `sync_pulse_frames` controls pulse width and
  defaults to `1`.
- On Raspberry Pi, `play_video` also drives the reward pump through Pi-Plates
  DAQC2 `DOUT pump_pin`. Both `pump_interval` and
  `pump_pulse_time_seconds` are required. The first onset occurs one
  `pump_interval` after task startup and subsequent onsets remain on one
  absolute session schedule across clip boundaries. Pump writes run on a
  worker thread so DAQC2 I/O cannot stall display flips; delayed intervals are
  skipped rather than replayed. `pump_on` and `pump_off` log rows contain both
  requested and actual edge timestamps. Optional `daq.address`/`daq_address`
  and `daq.module_name`/`daq_module_name` select the board and driver.
- `play_video` treats each randomly selected temporal clip as a trial. It logs
  clip start/end and video sync on/off edges, but does not log every displayed
  video frame.

A minimal video-source portion of the configuration is:

```json
{
  "video_files": [
    "/mnt/experiment-videos/source_01.mp4",
    "/mnt/experiment-videos/source_02.mp4"
  ],
  "clip_duration_seconds": 5.0,
  "num_clips": 10,
  "seek_timeout_seconds": 30.0,
  "frame_rate": 30,
  "video_buffer_megabytes": 512,
  "pump_pin": 0,
  "pump_pulse_time_seconds": 0.25,
  "pump_interval": 60.0
}
```

Screen Selection
----------------
**Invariant: a subject window is opened by `bin.utils.setup_task_window()` and
closed by `bin.utils.close_task_window()`.**
This is a privileged task boundary, not a convenience wrapper. Any change to
screen discovery, synchronization, task startup, window placement, or cleanup
must retain that pair in every subject-facing task and run
`tests/test_screen_path_policy.py`. Do not add a task-local fallback path.

Multi-screen tasks use `screens.main` for the subject display and `screens.experimenter` for the secondary display.
Each value can be a detected screen index or an output name such as `HDMI-1` or `DSI-1`.
Use output names on X11 rigs because numeric monitor order is not stable across display changes.
Set either value to `null` to inherit the process environment defaults: `screens.main` reads `MAIN_SCREEN`, and
`screens.experimenter` reads `SECONDARY_SCREEN`. The touch launcher exports the canonical names of the physical
outputs it resolved, marks those values authoritative for every diagnostic and child task, and persists them into
each manager-generated block config. CLI selectors remain highest priority. Outside the launcher, config selectors
retain priority over ordinary environment defaults.
Output names such as `HDMI-2` are rig configuration values, not constants in the display code.
PsychoPy presentation windows default to true fullscreen. Main X11 windows use PsychoPy's native,
window-manager-controlled fullscreen path without post-creation moves or resizing. On dual-screen launcher runs
with main-input masking, the interface is withdrawn and the already-visible subject-screen curtain becomes a
verified X11 focus anchor before launching a task or diagnostic. A child is not started unless that focus transfer
completes, so the window manager cannot inherit the experimenter output as its active display. Pyglet preselection
is best-effort during PsychoPy's Linux `screen=0` workaround; the realized native rectangle is the authoritative
check. The experimenter preview synchronizes its separate drawable to its own output and uses the
same native fullscreen creation with scoped override-redirect so it remains on its configured output without
altering the subject window's presentation path. Every realized native rectangle is verified; tasks, diagnostics,
and experimenter previews all reject incorrect placement before timing begins. A preview reports ready only after
its controls and initial subject-view scene complete their first successful flip.
The shared window factory disables PsychoPy's implicit constructor timing benchmark; timing begins only after the
launcher has released and the task has reactivated the subject window.
The launcher, main-screen curtain, and experimenter controls likewise request true fullscreen after first being
positioned on their assigned output.
Display positions and sizes are read from the OS when each task starts. If display enumeration is unavailable, only
the main display is created from the native `2560x1600` fallback; detected OS dimensions always take precedence.
For `active_foraging`, setting `screens.main` and `screens.experimenter` to the same display is allowed and disables
the experimenter preview, so only the main task content is shown.

The `active_foraging` experimenter display shows the config name, current subject, and current trial as
`Trial: current / total`, in addition to the current system time and elapsed task timer. An indefinite task
(`n <= 0`) displays `∞` as its total. While stimuli are visible, their selectable hit boxes are outlined by
reward level: red for `0`, gray for `1`, yellow for `2`, and green for `3`. Clicking `rew.` or pressing `r`
delivers the configured manual juice-pump pulse. The keyboard command works when either the main task window or
experimenter window has keyboard focus.

The `match2cue` display uses the same preview, subject/trial indicator, manual reward control, and exit control.
Its running counts are `Correct`, `Incorrect`, and `Rewards delivered`; it does not show reward-level outlines.

Main-display sizes and `win_size` overrides always use native framebuffer order. On the current rig that is
`2560x1600`, with the OS output set to `normal`. `active_foraging`, `match2cue`, and `play_video` compensate by
turning subject-facing stimuli 90 degrees clockwise. If the OS output is returned to RandR `right`, the software
compensation becomes zero to avoid a double rotation. Positions and touch coordinates remain in native framebuffer
coordinates. The experimenter process transposes the entire scene to the subject-facing `1600x2560` view.

Eye Tracker Calibration
-----------------------
`task/calibrate_eye_tracker.py` calibrates **Eye 0 (the first UART slot)** from the digital eye tracker.
`bin/eye_tracking.py` owns acquisition, validity checks, smoothing and calibration. The controller-side wire
codec and UART routines are adapted from the companion `eye_tracker` repository; provenance and license are
in `bin/eye_tracker_wire/SOURCE.md`. No separate checkout or camera package is needed on the task Pi.

```bash
python task/calibrate_eye_tracker.py --config config_files/eye_tracking/calibration_rpi.json
```

Both Pi launchers use this config, which enables DAQC2 reward output. It uses the shared task defaults for
calibration and filtering. For a desktop control/display check, use
`--config test_configs/eye_calibration_config.json --simulate_eye` (reward output disabled). Two displays are required.
The subject window uses the same placement, realized dimensions, rotation and vblank verification as
`active_foraging`. Calibration controls run in the same fullscreen, non-vsync `ExperimenterPreview` process
as other tasks. `experimenter_win_size` is no longer used; the secondary window fills its selected display.

Connection and protocol:

- `uart.device` defaults to `/dev/ttyAMA10`, the Pi 5 dedicated **3-pin UART connector**. This differs from
  `/dev/ttyAMA0`, which selects UART0 on the 40-pin GPIO header. The current tracker profiles
  also default to `/dev/ttyAMA10`.
  `/dev/serial0` normally aliases the dedicated connector on Pi 5; an explicit device avoids alias ambiguity.
  See [Raspberry Pi UART documentation](https://github.com/raspberrypi/documentation/blob/master/documentation/asciidoc/computers/configuration/interfaces.adoc).
- Both ends must select the UART actually wired: cross TX to RX and share ground, using 3.3 V logic.
  Disable the serial login console/getty on that device and ensure the task user has serial-port access.
  If the tracker Pi also uses its 3-pin connector, set its `transport.uart_device` to `/dev/ttyAMA10` too.
- Use `uart.baud: 460800`, raw 8-N-1 without flow control, matching the tracker. `uart.poll_rate_hz` defaults
  to 100; this is a request/response protocol, not an unsolicited byte stream. Each exchange consists of a
  64-byte command and 64-byte status with CRC-32C and echoed command sequence.
- `uart.manage_tracking: true` sends acknowledged START_TRACKING on entry and STOP_TRACKING on exit.
  START can also start recording if the tracker config enables `record_on_tracking`; STOP flushes it.
  Set this option to `false` to poll an already-running tracker without changing its state.
  State commands wait up to `uart.state_timeout_s` (10 s) and retry the identical command
  `uart.state_retries` times (default 1). Close may wait for these acknowledgements.
- `uart.expect_recording` defaults to `true`: a missing `RECORDING` status flag shows a
  warning while allowing the task and valid gaze acquisition to continue. Set it to `false`
  when recording on the tracker Pi is intentionally disabled. Error, stopped/unconfigured,
  calibration, and restarted-service states also produce warnings without aborting tasks.

Eye data and calibration:

- Only slot 0 is used, whether one or two eyes are configured. Its x/y are signed Q16.16 **crop-local pixels**;
  pupil diameter is Q12.4 and confidence is 0–255. Input x points right and input y points down.
- CRC/flags/acknowledgements are checked before use. Blink, lost, stopped, unconfigured, stale or error states
  cannot earn automatic rewards or set offsets. `uart.max_sample_age_s` defaults to 0.15 s; age includes the
  reported result age, UART round trip and time since receipt. Camera timestamps are not compared across Pis.
  Duplicate frames do not update smoothing or count as additional fixation samples. A service restart
  invalidates the session and requires restarting calibration; its ROI coordinates may have changed.
- `eye_filter.ema_gamma` defaults to 0 (no added smoothing); higher values below 1 retain more of the previous
  sample. Filtering occurs at the camera frame rate, not the UART poll rate.
- The affine calibration maps camera crop pixels to subject-relative fractions:
  `subject_x = x_scale * eye_x + x_cross * eye_y + x_offset` and
  `subject_y = y_cross * eye_x + y_scale * eye_y + y_offset`.
  The corresponding `initial_` coefficient settings initialize calibration; non-null runtime coefficients
  supplied by the experiment manager take precedence. Defaults are +0.005 for `x_scale`, -0.005 for
  `y_scale`, and zero for cross terms and offsets. Existing four-coefficient files load with zero cross terms.
  These are starting values to calibrate, not a measured mapping. **Old analog-voltage calibration values must
  be cleared/replaced in the experiment state before use.** Changing camera ROI or analysis resolution also
  requires recalibration. Analog DAQ input configuration is rejected rather than silently reused.
- Output fractions refer to the oriented subject scene, including the actual window size in windowed mode:
  x=-0.5 is left, x=0.5 right, y=-0.5 bottom and y=0.5 top. The preview preserves that scene's aspect ratio.
  With X11 `normal`, the mounted screen is compensated by 90° clockwise stimulus rotation. Subject up
  therefore maps to native-screen right, and subject left maps to native-screen up. Camera/UART pixels
  remain in their original coordinates; the display rotation is applied once, after calibration.

Click the preview to move the fixation target, or use **number-pad 8/6/2/4** for up/right/down/left
with either calibration window focused. Each keypress moves it by 2.5% of the subject-scene width or
height, stopping at the screen edges. Plain 8/6/2/4 also work. Moving the target resets auto-reward
fixation history just as clicking does. When the subject looks at the target, **Z** (or **Add point**)
captures the current valid, smoothed eye position and refits the affine transform using all points collected
in this calibration session. A persistent red dot marks each captured target on both displays. One point
changes only offsets; two points adjust the constrained direction while preserving the remaining sensitivity.
Three noncollinear points determine a full affine transform. Additional points use a least-squares fit;
collinear or repeated points preserve directions that the observations cannot determine.
**Delete** (or Backspace) removes the last point and restores the transform and anchor from before that capture.
Removing every point restores the pre-collection mapping. Invalid/stale eye samples cannot add points.

The bottom and left sliders show signed subject-X/Y gains from the latest fit and update automatically
after capture or undo. Each gain is the signed length of its affine row (the largest-magnitude coefficient
sets its sign; the diagonal wins ties), so it controls that subject axis even when the camera axes are mixed.
Manual changes scale the entire row around the last captured eye position, keeping that position fixed.
Adding another point refits the collected data and snaps the sliders to the resulting gains. Slider ranges
expand when necessary to display an automatic fit. Before the first capture, the first valid sample provides
the gain anchor; moving the target does not change it. The anchor and axis directions are saved with the
calibration, including when a gain is zero. Invalid gaze is hidden
and its status is displayed. The green reward button or `r` delivers a manual pulse; Exit or Escape ends the task.
The `f`, `z`, Delete, `r`, `a`, and `v` shortcuts work with either calibration window focused.
Arrow keys also work in either window: Right/Left increase/decrease subject-X gain, and Up/Down increase/decrease
subject-Y gain by 0.0001 per keypress, within the slider limits. The corresponding slider and its readout update
together. These are signed adjustments: Up moves a negative Y gain toward zero. The gain anchor stays fixed.
Press `f` with either calibration window focused to double the fixation cross (including stroke width) and
flash red, orange, yellow, green, blue, indigo, and violet at 7 Hz over one second.
It then returns to its original color, normal size and steady display. Pressing `f` again
restarts the cue; the target position, calibration and fixation acceptance area stay unchanged.

Press **V** to toggle attention videos (initially off). Video replaces the fixation cross in a
**100×100 subject-screen-pixel box**, centered on the current fixation location. Clicking or using
the number pad moves the video with the target, including on rotated displays. The experimenter
preview mirrors the video and retains the gaze marker. V off restores the cross; F animates the
cross when it is visible. Video toggles do not change calibration or reset auto-reward fixation history.

Calibration accepts the same media settings as `play_video.py`: `video_files`, `clip_duration_seconds`,
`frame_rate`, `seek_timeout_seconds`, `video_buffer_megabytes`, `ffprobe`, and `seed`. Set `video_size`
to `[width, height]` in pixels to change the box. The supplied calibration configs use the same
`monkey_kingdom.mp4` path as `test_configs/play_video_config.json`, with 5-second clips at 24 fps;
edit the paths and rate for your preprocessed HEVC files. Without a config, clip duration defaults to
5 seconds and `frame_rate` to 30 Hz. Sources must match the configured rate, as in `play_video`.

Random files and start points are selected repeatedly until V is toggled off; no `num_clips` limit
applies. Clips are center-cropped to the box without stretching. The shared decoder prepares the next
clip while the current clip plays and scales pixels before transfer to the display. The decoder budget
is split between these two clips. A delayed seek holds the last video frame; initial loading shows a
black box. File/decoder errors switch video off and appear in the status readout without aborting
calibration. Attention clips follow elapsed time, and a slow display uses the latest frame; the full-screen
task retains its exact-frame presentation policy. UART acquisition and calibration controls continue
during playback, and the movie continues updating during reward pulses.

Reward and fixation settings:

- `raspi: true` enables the shared `DAQC2DigitalOutputs` adapter. `daq.address` (default 0) and
  `daq.module_name` select the plate. `pump_pin` means **DAQC2 DOUT bit 0–7**, default 0, as in active foraging;
  it is not a BCM GPIO pin. Logical on uses `setDOUTbit`, off uses `clrDOUTbit`. With `raspi: false`, reward
  events are logged without hardware writes. An enabled output that fails to initialize aborts the task.
  The preview indicates when reward output is disabled; `--raspi` enables it for a direct launch with a test config.
- `pump_pulse_time_seconds` defaults to 0.25. Pulses use software timing and log actual on/off times.
  The pump is cleared on exit or exceptions, before UART/camera shutdown.
- `fix_diameter` is a fraction of the shorter subject-scene dimension (default 0.05).
- Press **A** to toggle automatic rewards; the mode starts **off**. The preview shows ON/OFF and both
  fixation percentages. `reward_interval` defaults to **2 seconds** and must exceed the pump pulse duration.
  Each automatic reward requires **more than 95% fixation during the past 100 ms** and **more than 85%
  during the past `reward_interval` seconds**, plus a currently valid, in-target eye position.
- Percentages measure time inside the fixation area, using distinct camera samples and clipping their
  held durations to each window. Blink/loss count as non-fixation; stale/error states and long observation
  gaps invalidate history. Repeated UART replies cannot manufacture new camera observations.
  Acquisition continues during pump pulses, so sustained fixation can earn repeated rewards without a
  fixation break. Rewards are spaced by at least `reward_interval` between pulse onsets; if fixation fails
  at the due time, the task waits until both criteria pass and never queues catch-up pulses.
- Toggling the mode or changing gain, zero, or target clears history and requires a fresh full interval.
  Manual **R** rewards remain available and postpone the next automatic reward by one interval.
  `fix_accept_percent` is superseded by the two fixed thresholds; an older positive `fix_accept_time` is
  used as the interval only when `reward_interval` is absent.

On normal exit, `calibration.json` retains the top-level `eye_tracker_calibration` object with all six
affine coefficients and the collected `calibration_points` (smoothed camera pixels and subject-fraction targets).
The experiment manager preserves all six coefficients in state and later block configs, including when its
initial state declared only the older four coefficients. Additional metadata records pixel input units, `eye_id: 0`,
UART/filter settings, gain anchor, oriented/native screen dimensions, rotation and latest sample validity. The session
also contains message and pump event logs.

Calibration and task previews reuse their text layouts for changing readouts and disable automatic per-frame
stimulus logging. This avoids the label-allocation leak documented in PsychoPy 2025.1.1's `TextStim` setter;
the calibration gaze marker and controls still update at 20 Hz.
If calibration fails after receiving a sample (including a preview-process crash), the main process turns
the pump off and saves its applied settings to `calibration_recovery.json` before shutting down.
This file is marked interrupted and is not automatically imported by the experiment manager. To resume from
it, explicitly set `eye_tracking.calibration_file` to its path in the calibration task config, check the
mapping on the rig, and exit normally to write a completed `calibration.json`.

Eye tracking in presentation tasks
---------------------------------
`active_foraging`, `play_video`, `match2cue`, `afc_csc1`, `afc_trial_sequence`, and `random_image_sequence`
use the same Eye 0 UART reader and calibration loader. Enable it in a task config with
`"eye_tracking": {"enabled": true}`. The experiment manager passes the latest calibration coefficients and
references the calibration artifact so UART/filter settings and pixel-coordinate metadata also carry forward.
Within an experiment, a newly completed calibration supersedes an older `calibration_file` reference.
For a standalone task, reference a saved file instead:

```json
"eye_tracking": {
  "enabled": true,
  "calibration_file": "../calibrations/calibration.json"
}
```

Relative paths are resolved against the source config before the manager copies it into a block. Embedded
`eye_tracker_calibration` objects or histories are also accepted. Non-null flat calibration coefficients take
precedence; null placeholders fall back to saved values. All four coefficients are required for enabled
presentation tasks. Explicit `eye_tracking.uart` and `eye_tracking.filter` settings override saved settings
(for example, `"uart": {"simulate": true}` for a hardware-free check). Eye tracking defaults to disabled and
is independent of `raspi`, which controls reward hardware. Saved UART settings include `simulate`; override
it to `false` when switching a simulated calibration to hardware, and recalibrate on the rig.

Tasks with an experimenter preview show a blue gaze marker in the subject-oriented scene, including video.
Invalid, stale and offscreen gaze is hidden. The two sequence tasks currently have no experimenter preview,
but provide the same main-thread gaze access. No choices or stimulus timing depend on gaze at present.

For future gaze-dependent task logic, use `session_logs.eye_tracking.get_position()` in the main task thread:
it returns an unclamped `(x, y)` pair in centered subject-scene fractions, or `None` when invalid/stale.
`get_state()` provides raw/smoothed crop pixels, validity, timestamps and frame metadata. Check that
`session_logs.eye_tracking` is not `None` when eye tracking is optional. Both methods read the worker's latest
state without serial I/O on the presentation thread. `"show_preview": false` disables the marker while keeping
that API available. The session owns tracker startup/shutdown; gaze updates do not alter stimulus commands.

Every enabled presentation block also writes **`eye_tracking.csv` directly in that block's directory**,
alongside `event_log.tsv` and `behavior_log.tsv`, for example:
`logs/exp_Y_20260813_001/blocks/1_active_foraging/eye_tracking.csv`.
Standalone runs put it in their usual session directory. Disabled eye tracking creates no eye CSV.

Recording runs at **60 Hz** on its own worker, independently of stimulus changes, screen flips, reward waits,
or whether the experimenter preview is enabled. Set `eye_tracking.record_rate_hz` to change that cadence
(for example, `50`). UART acquisition keeps its separate `uart.poll_rate_hz` schedule. Each recording tick
observes the reader's latest state; a camera slower than the recording rate produces repeated source frames.
Missed recording deadlines leave actual timestamp gaps, without backfilled rows or invented measurements.

The CSV columns are:

| Column | Meaning |
| --- | --- |
| `timestamp_ms` | Observation time in milliseconds since the block's session clock started; same origin as `event_log.tsv` (which uses seconds). Independent of file-flush time and flips. |
| `x_screen_px`, `y_screen_px` | Calibrated, filtered position in native PsychoPy pixel coordinates: `x = screen_x - W/2`, `y = H/2 - screen_y`, where screen coordinates start at the top left. The full affine transform, scene size and display rotation are applied, matching stimulus positions. Offscreen positions are retained without clamping. |
| `pupil_diameter_px` | Eye 0 pupil diameter in camera pixels, as supplied by the tracker. |
| `eye_valid`, `rejection_reason` | Eye validity (`1`/`0`) and reason such as `blink`, `lost`, `stale`, or `uart_error`. Invalid samples write `nan` for both position coordinates and pupil diameter. |
| `screen_ready` | `1` once the task has supplied its screen geometry. During setup, `0` means screen coordinates are still `nan`; a valid pupil measurement can already be recorded. |
| `sample_timestamp_ms` | Estimated source-sample time on the same block clock, accounting for reported age and UART round-trip time; `nan` before a source timestamp is available. This is not the unsynchronized remote camera clock. |
| `sample_index`, `frame_sequence` | Source sample/frame identifiers, allowing repeated camera measurements to be distinguished from new ones. |

Thus, after accurate calibration, looking at a stimulus positioned at `(x, y)` should record that same
centered pixel position. The existing main-thread `get_position()` API continues returning subject-scene
fractions. Applied calibration, UART/filter settings, recording rate and screen geometry are recorded in
the block's message log. CSV storage uses a fixed-size buffer, flushes about once per second, and closes
before UART shutdown; writer failures are reported by task log flushing/cleanup instead of silently losing data.

CPU Affinity for Timing-Critical Tasks
--------------------------------------
The `active_foraging` and `play_video` tasks treat CPU core `0` as the timing-critical presentation core.

- The main `active_foraging` process, including stimulus presentation and touch-event detection, pins itself to CPU `0` before entering the trial loop.
- `play_video` measures the main display and starts the experimenter preview on worker cores. For each clip, its ffpyplayer `spawn` worker also starts on the worker-core pool and prepares either the whole clip or the first two bounded chunks before playback. Only the main presentation thread moves to CPU `0` for the exact, refresh-locked frame loop; it returns to worker cores between clips while the preparation worker remains off CPU `0`.
- Non-timing-critical child processes such as the background trial-generation worker and the experimenter preview process inherit the remaining CPU cores. This keeps the `play_video` experimenter preview off CPU `0` as well.
- This is necessary because `multiprocessing` children inherit the parent's CPU affinity by default. To prevent workers from inheriting CPU `0`, the parent process is first moved onto the non-zero worker-core pool, the child processes are spawned, and then the parent is pinned back to CPU `0`.
- For the intended timing behavior on Linux or Raspberry Pi, CPU `0` should also be isolated from normal OS scheduling at the kernel level, for example with `isolcpus`, `nohz_full`, `rcu_nocbs`, or an equivalent cpuset-based setup.
- Launch the task from a shell or service whose affinity mask still includes CPU `0` and the worker cores. If the launcher has already removed CPU `0` from the process affinity mask, the task cannot pin the main presentation process onto that core.
- Event, message, and behavior logs for `active_foraging` are buffered during the timing-critical portion of a trial and flushed only in the between-trial gap, so synchronous disk flushes do not run while the initiation cue, stimulus presentation, touch detection, and reward delivery are active.
- The root-menu system diagnostic uses the same CPU 0 placement for its refresh-rate and flip-lock measurements and reports affinity failure as a diagnostic error.

Active Foraging Hardware Signals
--------------------------------
When `active_foraging` runs with `raspi=true`, the hardware outputs are split across Raspberry Pi GPIO and the Pi-Plates DAQC2plate:

- `trial_start_pin`: Raspberry Pi BCM GPIO pin used for the timing-critical trial-start pulse. This still uses `lgpio` and is scheduled on the PsychoPy flip path for precise timing.
- `daq.address` or top-level `daq_address`: DAQC2plate address for pump and buzzer output, default `0`.
- `pump_pin`: DAQC2 DOUT bit for pump reward delivery, in the range `0` through `7`.
- `buzz_pin`: DAQC2 DOUT bit for the timeout buzzer, in the range `0` through `7`.

Pump and buzzer writes use `piplates.DAQC2plate.setDOUTbit(addr, bit)` for logical on and `clrDOUTbit(addr, bit)` for logical off. DAQC2 DOUTs are open-drain outputs, so the terminal voltage is inverted for a pulled-up digital signal: setting a DOUT bit pulls the terminal near `0 V`, while clearing it turns the transistor off and lets the terminal return to `5 VDC`. The checked-in active-foraging configs use `pump_pin=0` and `buzz_pin=1` as DOUT defaults.

The launcher setup scripts use the same mapping by default: `task/pulse_pump.py` primes the pump on DAQC2 `DOUT0`, and `task/pulse_buzzer.py` tests the buzzer on DAQC2 `DOUT1`.

Active Foraging Timing
----------------------
The main visual timing parameters in `active_foraging` are interpreted by the presentation mode, not as abstract global delays. Any finite duration is accepted; it does not need to be an exact refresh interval. Each visual phase is planned for the nearest achievable refresh transition using the measured synchronized display rate, while the exact configured request is retained in the event log. `requested_time_since_session_start` and `time_since_session_start` expose flip-submission and realized-flip timing. Minimum semantic rules remain: `choice_time` must be positive, and when `sequential=true` or `is_memory=true`, `duration` must be positive. `pump_delay_time` remains a wall-clock delay before reward delivery.

- `duration`: stimulus display duration. When `sequential=true`, this is the on-screen time for each individual stimulus in the sequence. When `sequential=false` and `is_memory=true`, the full array remains visible for `duration`, then the task switches to dot-only choice for `choice_time`. When `sequential=false` and `is_memory=false`, `duration` must be exactly `0`; the full array appears on the first choice frame and remains visible for `choice_time` only.
- `isi`: pre-stimulus cue interval, not a between-trial delay. In simultaneous non-memory mode it shows dots at all candidate locations before the full array appears. In sequential mode, for every task using the shared option-presentation backend (including `active_foraging`, `match2cue`, and `afc_trial_sequence`), it shows the dot cue for each item before that item is shown and, between consecutive stimuli, also holds the post-stimulus dot scene unchanged for the same duration.
- `choice_time`: response-window extension after the stimulus display phase defined by the active mode. In simultaneous non-memory mode the response window starts on the first frame of the full array and lasts `choice_time`, with the full array remaining visible throughout. In memory modes, the response window begins only after the stimulus display phase has finished and lasts `choice_time`, with only the remembered dot locations visible.
- `iti`: inter-trial interval after choice handling. This begins only after reward delivery or timeout handling completes; it is not inserted between option presentations within a trial. The old `ibi` config/CLI name remains accepted only as a deprecated compatibility alias.
- `pump_delay_time`: delay in seconds between a rewarded choice being made and the first pump pulse. It applies only on rewarded trials with at least one configured pump pulse, and defaults to `0.0`.
- `pump_pulse_time_seconds`: duration in seconds that the pump output remains on for each reward pulse.
- `inter_pump_interval`: delay in seconds between repeated pump pulses. When omitted, it defaults to `pump_pulse_time_seconds`, preserving the previous behavior.

Common `active_foraging` configurations:

- Config A: `sequential=false`, `is_memory=false`
  - `isi`: all choice-location dots are shown together before the stimuli.
  - `duration`: must be `0`.
  - `choice_time`: the full array appears on the first choice frame and remains selectable for this long.
  - Total selectable time: `choice_time`.

- Config A-memory: `sequential=false`, `is_memory=true`
  - `isi`: all choice-location dots are shown together before the stimuli, when greater than zero.
  - `duration`: the full array appears together for this long and choices are not accepted yet.
  - `choice_time`: after `duration`, only memory dots remain visible and selectable for this long.

- Config B: `sequential=true`, `is_memory=true`
  - For each option in the trial: show that option's dot for `isi`, then show that stimulus for `duration`. After every stimulus except the last, hold the post-stimulus dot scene unchanged for `isi` before the next option's dot cue.
  - After each stimulus disappears, its location remains as a memory dot.
  - After the final stimulus, the task enters a dot-only choice period for `choice_time`.
  - `iti` starts only after the resulting reward or timeout has finished.

Active Foraging Positioning
---------------------------
`active_foraging` places every stimulus center on a stimulus circle in native main-framebuffer pixel coordinates. `center_point` is `[x, y]` with origin at the upper-left of the native `2560x1600` screen. When `center_point` is `null`, it defaults to the exact middle of the main screen. `stim_range_radius` is the circle radius in pixels. When it is `null`, it defaults to half the distance from `center_point` to the closest screen edge. Authored image sizes remain `[width, height]`; their native axis-aligned bounds swap after a 90-degree turn, which placement and overlap checks account for.

Active foraging and match-to-cue require four self-initiation settings. `initiation_cue_center_position` is either
`null` for the screen center or `[x, y]` in the same upper-left-origin native framebuffer coordinates.
`initiation_cue_style` is `"checker"` for the legacy Gaussian-masked checkerboard or `"blob"` for a
Gaussian-faded cue in `init_dot_color`; both styles have the same dimensions and darken when pressed.
`hold_cue_to_init_time_s` is the continuous cue-hold duration required to initiate a trial.
`hold_before_choice=true` keeps the pressed cue visible and requires that hold through the pre-choice phases.
A missing or out-of-cue touch lasting at most 100 ms is tolerated. A longer break aborts only the current trial,
logs `initiation_cue_release`, and starts the inter-trial interval. When a fixation cross is enabled, it is white
before and after choice and black while the choice window is open.

- `fixed_positions=true`: locations are evenly spaced around the circle. The spacing angle is `2*pi / num_afc`, and the first location is offset by half that spacing from the point directly below `center_point`.
- `fixed_positions=false`: locations are random points on the circle, with rejected draws when stimulus bounding boxes would overlap.
- Custom `center_point` and `stim_range_radius` values can be provided in JSON or as `--center_point X Y --stim_range_radius R`.

Active Foraging Subject Maps
----------------------------
`active_foraging` requires a JSON config and an exact, case-sensitive `subject` value. Both `freq_space_tsv` and
`reward_space_tsv` must be objects mapping subject names to paths; scalar path values and command-line path
overrides are not supported. The task resolves both maps before opening a window. If `subject` is unset, absent
from either map, or resolves to an empty/non-string path, the task exits with an error.

```json
{
  "subject": "Yuri",
  "freq_space_tsv": {
    "Yuri": "./task/resources/csc2/freq_space_TY.csv",
    "Buzz": "./task/resources/csc2/freq_space_SB.csv"
  },
  "reward_space_tsv": {
    "Yuri": "./task/resources/csc2/reward_space_TY.csv",
    "Buzz": "./task/resources/csc2/reward_space_SB.csv"
  }
}
```

When the experiment manager launches the task, its generated block config supplies the selected full subject name,
so the subject keys in these maps must match the names in the launch config's `subjects` object.

Active Foraging Color TSV
-------------------------
`active_foraging` expects `colors_tsv` to be a tab-delimited file with four columns: `id`, `r`, `g`, `b` (column name case is flexible, for example `ID R G B` also works).

- Include a header row.
- The first data row is treated as the background gray and is not used as a selectable stimulus color.
- Every later row is one displayable color definition with a unique integer ID and integer RGB values.
- Row order matters. After the background row, colors must be ordered by luminance groups: all `n_colors` base colors for luminance level 1, then all `n_colors` base colors for luminance level 2, and so on.
- The number of non-background color rows must equal `n_colors * n_lum_levels`.

Example:

```tsv
id	r	g	b
0	168	169	166
1	143	115	120
2	142	116	114
3	141	117	108
```

Match2Cue Task
--------------

Run the bundled native-SVG test configuration with:

```bash
python task/match2cue.py \
  --config test_configs/match2cue_test.json \
  --main_screen 1 \
  --experimenter_screen 0
```

Replace the screen selectors with the local display indices or RandR output names. The interface supplies them
automatically when launching the task as an experiment block.

Each trial runs `onset cue -> match cue/tap -> optional match-cue reward -> delay -> options -> choice -> optional
choice reward -> inter-trial interval`. The `match_cue_duration` and `delay_time` fields control the two added
visual phases. Option presentation uses the same `sequential`, `is_memory`, `fixed_positions`, `duration`, `isi`,
`choice_time`, `center_point`, `stim_range_radius`, and `num_afc` semantics as active foraging.

The cue is sampled uniformly from the complete configured shape/color/luminance stimulus space. One exact copy
is guaranteed among the options; every other option is an independent random draw with replacement from that
same space. Therefore the cue may occur more than once. Selecting an exact match is correct. With the default
`tie_mode="random"`, a selected match is rewarded with probability `1 / matching_option_count`, preserving the
legacy behavior. With `tie_mode="all"`, every matching option is rewarded if selected. An incorrect choice is
never rewarded.

`correct_num_pulse` is the positive number of pump pulses delivered for a rewarded choice and defaults to `1`.
`pump_delay_time` is applied once before that pulse train. `inter_pump_interval` is the delay between consecutive
pulses; when omitted, it defaults to `pump_pulse_time_seconds`. The pulse interval is not added after the final
pulse. `reward_match_cue_prob` is an independent probability from `0` to `1`, defaulting to `0`, that a fresh tap
on the displayed matching stimulus earns one pump pulse. The tap is accepted only while the matching stimulus is
visible; a held initiation press does not count. Reward delivery occurs after match-cue offset so the
configured `match_cue_duration` remains frame-accurate. Match-cue and manual rewards always use one pulse and are
not multiplied by `correct_num_pulse`.

The config has no frequency-space, reward-space, reward-level, timeout, or buzzer settings. Its `subject` field is
still required and is displayed and logged, but it does not alter task behavior.

Set `n_colors` to `0` to use SVG artwork exactly as authored rather than recoloring it. In this mode the color TSV
must contain exactly one data row—the background gray—and `n_lum_levels` should be `0`. With positive
`n_colors`, the task uses the same background-first palette layout as active foraging and requires exactly
`n_colors * n_lum_levels` non-background rows. `n_shapes` must match the number of rows in `shapes_tsv` in either
mode.

The match-to-cue event log adds `match_cue_on`, `match_cue_touch`, and `delay_start`. Both tasks log
`initiation_cue_release` when a required hold is broken for more than 100 ms. The behavior log records cue
and option feature indices, `matching_option_count`, `tie_mode`, `choice_correct`, `match_cue_tapped`,
`match_cue_reward_probability`, `match_cue_reward_delivered`, `reward_probability`, `reward_delivered`, and
`choice_reward_pulse_count`, in addition to the shared choice/touch timing fields. The existing
`reward_probability` and `reward_delivered` fields continue to describe the option-choice reward. The
experimenter preview shows the configured subject, current/total trial, and cumulative counts for correct choices,
incorrect choices, and individual reward pulses delivered. Correct and reward-pulse counts can differ because of
match-cue rewards, multi-pulse choice rewards, and random tie handling.

Configuration via JSON (required for tasks)
-----------------------------------------
All tasks in this repository must support loading a JSON configuration file as an alternative to specifying parameters via command-line arguments. The config file should allow you to set experiment-level parameters such as:

- `config_name` (string): stable, broad analysis-family identifier following the convention above; it is not a per-run or per-variant label
- `images_dir` (string): path to image resources
- `output_dir` (string): path where logs and metadata will be saved
- `n` (int): number of trials for tasks that use this field; `play_video` uses `num_clips` instead
- `duration` (number): stimulus presentation duration in seconds. For `active_foraging` and `match2cue`, this must be positive when `sequential=true` or `is_memory=true`, and must be `0` only when both are false.
- `isi` (number): pre-stimulus / inter-stimulus interval in seconds; exact meaning is task-specific
- `iti` (number): inter-trial interval in seconds for trial-based tasks
- `bg` (array of 3 ints): background RGB values in 0-255
- `seed` (int, optional): random seed
- `fullscreen` (bool, optional)
- `win_size` (array of 2 ints, optional)
- `fixation_size` (int, optional)
- `image_size` (array of 2 ints, optional)
- `center_point` (array of 2 ints or null, optional): AFC-style stimulus circle center in main-screen pixels
- `stim_range_radius` (int or null, optional): AFC-style stimulus circle radius in pixels
- `trial_start_pin` (int, optional): AFC-style BCM GPIO pin for the trial-start pulse
- `daq.address` or `daq_address` (int, optional): DAQC2plate address for task hardware outputs
- `pump_pin` and `buzz_pin` (int, optional): DAQC2 DOUT bits, not Raspberry Pi GPIO pins

Tasks must validate the config when loaded and raise a helpful error if required keys are missing or types are invalid. Command-line arguments should override values in the config file when both are provided.

Example JSON config (`example_config.json`):

```json
{
  "config_name": "random_image_sequence",
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

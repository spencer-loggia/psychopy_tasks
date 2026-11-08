"""Read-only analysis of the recorded csc2 classic block; outputs stay local.

Run with a Python environment containing numpy, pandas, scipy and matplotlib.
The event log, not requested durations, defines stimulus visibility. Eye task
timestamps are reception times; camera timestamps define within-visit durations.
"""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/csc2_mpl')
import hashlib
import json
import re
import subprocess
import html
from collections import Counter
import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
DATA = Path('/Volumes/lsr-conway/DATA/monkey_test_box_data/experiments/exp_Y_20261107_006/blocks/2_csc2_foraging_classic')
B = pd.read_csv(DATA / 'behavior_log.tsv', sep='\t')
E = pd.read_csv(DATA / 'event_log.tsv', sep='\t')
M = pd.read_csv(DATA / 'message_log.tsv', sep='\t')
X = pd.read_csv(DATA / 'eye_tracking.csv')
reward_path = ROOT / 'task/resources/csc2/reward_space_TY.csv'
freq_path = ROOT / 'task/resources/csc2/freq_space_TY.csv'
REWARD = pd.read_csv(reward_path).iloc[:, 0].to_numpy().reshape(36, 36)
FREQ = pd.read_csv(freq_path).iloc[:, 0].to_numpy()
FREQ = FREQ / FREQ.sum()
COLORS = ['#2673b8', '#de7627', '#8c50a0']
POS = {}
for msg in M.message:
    hit = re.search(r'position_assigned trial_num=(\d+) option_num=(\d+).*psychopy_pos=\(([^,]+), ([^)]+)\)', msg)
    if hit:
        POS.setdefault(int(hit[1]), {})[int(hit[2])] = (float(hit[3]), float(hit[4]))


def summary(x):
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return {'n': 0}
    return dict(n=len(a), min=float(a.min()), q25=float(np.quantile(a, .25)),
                median=float(np.median(a)), q75=float(np.quantile(a, .75)),
                p95=float(np.quantile(a, .95)), max=float(a.max()), mean=float(a.mean()))


def runs(labels, camera_s, task_s, max_gap=.0601):
    """Contiguous runs: invalid/outside samples and camera gaps >60 ms split visits.

    Duration is first-to-last camera sample, without extrapolating at either end.
    These are ROI visits, not velocity-defined physiological fixations.
    """
    out = []
    start = None
    for i in range(len(labels) + 1):
        split = (i == len(labels) or (start is not None and (
            labels[i] != labels[i-1] or camera_s[i] - camera_s[i-1] > max_gap)))
        if start is not None and split:
            out.append(dict(option=int(labels[start]), first_index=start, last_index=i-1,
                            start_s=float(task_s[start]), end_s=float(task_s[i-1]),
                            camera_duration_ms=float((camera_s[i-1] - camera_s[start]) * 1000),
                            reception_duration_ms=float((task_s[i-1] - task_s[start]) * 1000),
                            samples=i-start))
            start = None
        if i < len(labels) and start is None and labels[i] > 0:
            start = i
    return out


def distribution(values):
    """Exact distribution of a sum under independent uniform option choices."""
    d = np.array([1.])
    for row in values:
        p = np.bincount(np.asarray(row, int), minlength=int(max(row))+1) / len(row)
        d = np.convolve(d, p)
    return d / d.sum()


def chance(rows):
    vals = rows[['reward_1', 'reward_2', 'reward_3']].to_numpy(int)
    chosen = rows.reward_level.to_numpy(int)
    result = {'n': len(rows), 'chosen_reward_counts': dict(Counter(map(int, chosen))),
              'offered_reward_counts': dict(Counter(map(int, vals.ravel())))}
    definitions = {
        'rewarded': (vals >= 2, chosen >= 2),
        'best_with_ties': (vals == vals.max(axis=1)[:, None], chosen == vals.max(axis=1)),
        'reward_sum': (vals, chosen),
        'pump_pulses': (np.maximum(vals-1, 0), np.maximum(chosen-1, 0)),
        'level_3': (vals == 3, chosen == 3),
    }
    for name, (offered, observed) in definitions.items():
        d = distribution(offered)
        k = int(observed.sum())
        result[name] = dict(observed=k, expected=float(offered.mean(axis=1).sum()),
                            upper_tail_p=float(d[k:].sum()),
                            null_sum_95pct=[int(np.searchsorted(d.cumsum(), q)) for q in [.025, .975]])
    return result


trial_rows, reveal_rows, visit_rows, raw_runs, traces = [], [], [], [], {}
for row in B.to_dict('records'):
    n = int(row['trial_num'])
    ev = E[E.trial_num == n].sort_values('task_time_ns')
    t0 = int(ev.loc[ev.event == 'options_dot', 'task_time_ns'].iloc[0])
    touch = int(ev.loc[ev.event == 'option_touch', 'task_time_ns'].iloc[0])
    gray = int(ev.loc[ev.event.isin(['reward_interval', 'timeout_interval']), 'task_time_ns'].iloc[0])
    choice = int(row['choice_made_index']) + 1
    assert t0 == int(ev.loc[ev.event == 'choice_start', 'task_time_ns'].iloc[0])
    assert abs((touch-t0)/1e9-row['choice_reaction_time']) < 1e-8
    vals = [int(REWARD[int(row[f'shape_{j}']), int(row[f'color_{j}'])]) for j in range(3)]
    assert vals[choice-1] == row['reward_level']
    z = X[(X.task_time_ns >= t0) & (X.task_time_ns <= touch)].copy().reset_index(drop=True)
    time_s = (z.task_time_ns.to_numpy() - t0)/1e9
    camera_s = (z.camera_time_ns.to_numpy() - z.camera_time_ns.iloc[0])/1e9
    positions = np.array([POS[n][j] for j in [1, 2, 3]])
    dist = np.linalg.norm(z[['x_screen_px', 'y_screen_px']].to_numpy()[:, None, :] - positions[None, :, :], axis=2)
    valid = (z.eye_valid.to_numpy() == 1) & np.isfinite(dist).all(axis=1)
    safe_dist = np.where(np.isfinite(dist), dist, np.inf)
    in_roi = safe_dist <= 175
    nearest = safe_dist.argmin(axis=1) + 1
    nearest[~(valid & in_roi.any(axis=1))] = 0
    visits = runs(nearest, camera_s, time_s)
    # The camera period is 19.983 ms, so three nominal 20-ms intervals span
    # 59.949 ms. A 1-ms tolerance avoids excluding these at a 60-ms threshold.
    kept = [v for v in visits if v['camera_duration_ms'] >= 59]
    for v in visits:
        raw_runs.append(dict(trial_num=n, **v))
    for v in kept:
        visit_rows.append(dict(trial_num=n, **v))
    per_roi_runs = []
    for j in range(3):
        per_roi_runs += runs(np.where(in_roi[:, j] & valid, j+1, 0), camera_s, time_s)
    max_dwell = max([v['camera_duration_ms'] for v in per_roi_runs], default=0)
    accepted = ev[ev.event == 'fixation_accepted']
    on_events = ev[ev.event.str.match(r'option_[123]_on$')]
    intervals = []
    for _, on in on_events.iterrows():
        j = int(on.event.split('_')[1])
        ons = int(on.task_time_ns)
        off = ev[(ev.task_time_ns > ons) & (ev.event == f'option_{j}_off')]
        offs = min(gray, int(off.task_time_ns.iloc[0])) if len(off) else gray
        prior = accepted[accepted.task_time_ns <= ons]
        fixation_ns = int(prior.task_time_ns.iloc[-1])
        item = dict(trial_num=n, option=j, chosen_option=choice, chosen_revealed=(j == choice),
                    reward_level=int(row['reward_level']), onset_ms=(ons-t0)/1e6,
                    offset_ms=(offs-t0)/1e6, touch_ms=(touch-t0)/1e6,
                    visible_before_touch_ms=max(0, min(touch, offs)-ons)/1e6,
                    full_visible_ms=(offs-ons)/1e6, onset_to_touch_ms=(touch-ons)/1e6,
                    accepted_ms=(fixation_ns-t0)/1e6, accepted_to_flip_ms=(ons-fixation_ns)/1e6,
                    offset_kind='option_off' if offs < gray else 'post_choice_gray',
                    onset_task_time_ns=ons, offset_task_time_ns=offs)
        intervals.append(item)
        reveal_rows.append(item)
    transitions = sum(a['option'] != b['option'] for a, b in zip(kept, kept[1:]))
    pair_dist = np.linalg.norm(positions[:, None, :]-positions[None, :, :], axis=2)
    adjacent = [pair_dist[a['option']-1, b['option']-1] for a, b in zip(kept, kept[1:]) if a['option'] != b['option']]
    first_entry = time_s[np.where(in_roi.any(axis=1) & valid)[0][0]] if (in_roi.any(axis=1) & valid).any() else np.nan
    tr = dict(trial_num=n, chosen_option=choice, reward_level=int(row['reward_level']),
              **{f'reward_{j+1}': vals[j] for j in range(3)},
              rt_ms=(touch-t0)/1e6, touch_to_gray_ms=(gray-touch)/1e6,
              cue_touch_to_dots_ms=(t0-int(row['initiation_task_time_ns']))/1e6,
              n_reveals=len(intervals), n_unique_revealed=len(set(v['option'] for v in intervals)),
              any_reveal=bool(intervals), chosen_ever_revealed=any(v['chosen_revealed'] for v in intervals),
              total_visible_before_touch_ms=sum(v['visible_before_touch_ms'] for v in intervals),
              first_reveal_ms=intervals[0]['onset_ms'] if intervals else np.nan,
              n_samples=len(z), valid_samples=int(valid.sum()), valid_fraction=float(valid.mean()),
              first_roi_entry_ms=float(first_entry*1000), max_continuous_roi_dwell_ms=max_dwell,
              n_visits_60ms=len(kept), n_unique_visited_60ms=len(set(v['option'] for v in kept)),
              option_transitions_60ms=transitions,
              transitions_between_nonoverlapping_rois=sum(d > 350 for d in adjacent),
              chosen_option_visited_60ms=choice in set(v['option'] for v in kept),
              overlapping_window_samples=int((in_roi.sum(axis=1)>1).sum()),
              min_option_separation_px=float(pair_dist[np.triu_indices(3, 1)].min()),
              dot_task_time_ns=t0, touch_task_time_ns=touch, gray_task_time_ns=gray)
    for threshold in [40, 80, 100]:
        tr[f'n_unique_visited_{threshold}ms'] = len(set(v['option'] for v in visits if v['camera_duration_ms'] >= threshold-1))
    trial_rows.append(tr)
    traces[n] = (z, time_s, camera_s, dist, nearest, intervals)

T = pd.DataFrame(trial_rows)
R = pd.DataFrame(reveal_rows)
V = pd.DataFrame(visit_rows)
U = pd.DataFrame(raw_runs)
T.to_csv(OUT / 'trial_timing.csv', index=False)
R.to_csv(OUT / 'stimulus_reveals.csv', index=False)
V.to_csv(OUT / 'gaze_visits_60ms.csv', index=False)
U.to_csv(OUT / 'gaze_roi_runs_all.csv', index=False)

metrics = {'completed_trials': len(T), 'revealed_trials': T.loc[T.any_reveal, 'trial_num'].tolist(),
           'no_reveal_trials': int((~T.any_reveal).sum()), 'reveal_count': len(R),
           'trials_with_multiple_distinct_stimuli_revealed': int((T.n_unique_revealed > 1).sum()),
           'performance': {name: chance(rows) for name, rows in [
               ('all', T), ('no_reveal', T[~T.any_reveal]), ('any_reveal', T[T.any_reveal])]},
           'weighted_sampling_reward_probabilities': {str(k): float(FREQ[REWARD.ravel() == k].sum()) for k in range(4)},
           'timing': {k: summary(T[k]) for k in ['rt_ms', 'touch_to_gray_ms', 'cue_touch_to_dots_ms',
                       'first_roi_entry_ms', 'max_continuous_roi_dwell_ms']},
           'rt_by_visibility': {name: summary(rows.rt_ms) for name, rows in [('no_reveal', T[~T.any_reveal]), ('any_reveal', T[T.any_reveal])]},
           'reveals': {k: summary(R[k]) for k in ['onset_ms', 'visible_before_touch_ms', 'full_visible_ms', 'accepted_to_flip_ms']},
           'gaze': {'valid_fraction_during_choices': float(T.valid_samples.sum()/T.n_samples.sum()),
                    'samples_during_choices': int(T.n_samples.sum()),
                    'first_roi_entry_no_reveal_ms': summary(T[~T.any_reveal].first_roi_entry_ms),
                    'roi_visit_duration_60ms': summary(V.camera_duration_ms),
                    'all_roi_run_duration_ms': summary(U.camera_duration_ms),
                    'max_roi_dwell_no_reveal_ms': summary(T[~T.any_reveal].max_continuous_roi_dwell_ms),
                    'distinct_option_counts_60ms': dict(Counter(map(int, T.n_unique_visited_60ms))),
                    'transitions_counts_60ms': dict(Counter(map(int, T.option_transitions_60ms))),
                    'multiple_visits_no_reveal': int(((T.n_unique_visited_60ms >= 2) & ~T.any_reveal).sum()),
                    'trials_with_nonoverlapping_transitions': int((T.transitions_between_nonoverlapping_rois > 0).sum()),
                    'trials_with_roi_overlap': int((T.min_option_separation_px < 350).sum()),
                    'samples_inside_multiple_rois': int(T.overlapping_window_samples.sum()),
                    'chosen_option_visited_60ms': int(T.chosen_option_visited_60ms.sum()),
                    'multiple_distinct_visits_threshold_sensitivity': {
                        str(th): int((T[f'n_unique_visited_{th}ms'] >= 2).sum()) for th in [40, 60, 80, 100]}},
           'eye_recording': {'n_samples': len(X), 'valid_n': int(X.eye_valid.sum()),
                    'camera_interval_ms': summary(np.diff(X.camera_time_ns)/1e6),
                    'receipt_interval_ms': summary(np.diff(X.task_time_ns)/1e6),
                    'missing_camera_frames': int(np.maximum(0, np.diff(X.frame_sequence)-1).sum()),
                    'camera_frames_spanned': int(X.frame_sequence.iloc[-1] - X.frame_sequence.iloc[0] + 1),
                    'reason_counts': X.rejection_reason.value_counts().to_dict()},
           'rt_reward_spearman_exploratory': {
               name: dict(zip(['rho', 'p'], map(float, spearmanr(rows.rt_ms, rows.reward_level))))
               for name, rows in [('all', T), ('no_reveal', T[~T.any_reveal])]},
           'total_choice_time_s': float(T.rt_ms.sum()/1000),
           'total_prechoice_stimulus_visible_s': float(R.visible_before_touch_ms.sum()/1000)}
for event in ['onset_cue_on', 'options_dot', 'reward_interval', 'timeout_interval', 'sync_bar_flip']:
    ee = E[E.event == event]
    metrics.setdefault('flip_request_to_actual_ms', {})[event] = summary((ee.task_time_ns-ee.requested_task_time_ns)/1e6)
metrics['flip_request_to_actual_ms']['option_on'] = summary((E.loc[E.event.str.match(r'option_[123]_on$'), 'task_time_ns'] - E.loc[E.event.str.match(r'option_[123]_on$'), 'requested_task_time_ns'])/1e6)
metrics['rewarded_rate_exact_95_ci'] = list(map(float, binomtest(int((T.reward_level>=2).sum()), len(T)).proportion_ci()))
metrics['input_hashes_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [
    DATA/'behavior_log.tsv', DATA/'event_log.tsv', DATA/'eye_tracking.csv', DATA/'message_log.tsv',
    DATA/'config.json', reward_path, freq_path, ROOT/'bin/utils.py', ROOT/'bin/dot_fixation.py', ROOT/'bin/eye_tracking.py']}
metrics['local_git_head'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
(OUT/'metrics.json').write_text(json.dumps(metrics, indent=2))

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
fig, axes = plt.subplots(1, 3, figsize=(15, 11), gridspec_kw={'width_ratios':[1.7, 1, 1]})
ax = axes[0]
for r in T.itertuples():
    ax.plot([0, r.rt_ms], [r.trial_num, r.trial_num], color='#d1d8df', lw=3)
    for v in V[V.trial_num==r.trial_num].itertuples():
        ax.plot([v.start_s*1000, v.end_s*1000], [r.trial_num, r.trial_num], color=COLORS[v.option-1], lw=2)
    for v in R[R.trial_num==r.trial_num].itertuples():
        ax.plot([v.onset_ms, min(v.offset_ms, v.touch_ms)], [r.trial_num, r.trial_num], color='#151515', lw=6)
    ax.plot(r.rt_ms, r.trial_num, '|', color='#d12439', ms=6)
ax.set(xlabel='Milliseconds after dots / choice window open', ylabel='Trial', ylim=(73, 0), xlim=(0, 1150), title='Gaze visits and actual stimulus visibility')
ax.legend(handles=[Patch(color='#d1d8df', label='Response window to touch'),
                   *[Patch(color=c, label=f'Gaze visit: option {i+1}') for i,c in enumerate(COLORS)],
                   Patch(color='#151515',label='Stimulus visible before touch'), Patch(color='#d12439',label='Touch')], loc='upper left', bbox_to_anchor=(0,-.07), ncol=2, fontsize=8)
ax = axes[1]
bins=np.arange(350, 1151, 50)
ax.hist(T.loc[~T.any_reveal,'rt_ms'], bins=bins, color='#9caebd', label='No stimulus revealed (63)')
ax.hist(T.loc[T.any_reveal,'rt_ms'], bins=bins, color='#252e38', alpha=.85, label='Stimulus revealed (9)')
ax.axvline(T.rt_ms.median(), color='#d12439', ls='--', label=f'Median {T.rt_ms.median():.0f} ms')
ax.set(xlabel='Dot onset to touch (ms)', ylabel='Trials', title='Most choices precede any reveal')
ax.legend(fontsize=8)
ax = axes[2]
vals=T[['reward_1','reward_2','reward_3']].to_numpy()
d=distribution(vals>=2)
ax.bar(np.arange(len(d)),d,color='#9caebd')
obs=int((T.reward_level>=2).sum())
ax.axvline(obs,color='#d12439',lw=2,label=f'Observed {obs}/72')
expected=metrics['performance']['all']['rewarded']['expected']
p=metrics['performance']['all']['rewarded']['upper_tail_p']
ax.axvline(expected,color='#252e38',ls='--',label=f'Chance mean {expected:.2f}')
ax.set(xlim=(32,65),xlabel='Juice-rewarded choices out of 72',ylabel='Exact null probability',title=f'Random choice among offered options\nOne-sided p = {p:.3f}')
ax.legend(fontsize=8)
fig.suptitle('Yuri · 7 November 2026 · CSC2 classic block 2\n63/72 choices had zero logged stimulus exposure',fontsize=16,y=.98)
fig.tight_layout(rect=[0,.03,1,.94])
fig.savefig(OUT/'timing_overview.png',dpi=180)
plt.close(fig)

# A no-reveal trial with multiple robust, nonoverlapping ROI visits.
candidates=T[(~T.any_reveal)&(T.n_unique_visited_100ms>=2)&(T.transitions_between_nonoverlapping_rois>=1)].sort_values('n_unique_visited_100ms',ascending=False)
example=int(candidates.iloc[0].trial_num) if len(candidates) else 2
examples=[example,12,1,55]
fig,axes=plt.subplots(len(examples),1,figsize=(12,11))
for ax,n in zip(axes,examples):
    z,t,camera,d,near,intervals=traces[n]
    rr=T[T.trial_num==n].iloc[0]
    for j in range(3):
        ax.plot(t*1000,d[:,j],'.-',lw=1,ms=3,color=COLORS[j],label=f'Option {j+1}')
    ax.axhline(175,color='#555',ls='--',lw=1,label='175 px fixation radius')
    ax.axvline(rr.rt_ms,color='#c6233d',lw=1.8,label='Choice touch')
    for iv in intervals:
        ax.axvspan(iv['onset_ms'],iv['offset_ms'],color=COLORS[iv['option']-1],alpha=.18)
        ax.axvline(iv['accepted_ms'],color='#333',ls=':',lw=1)
        ax.text(iv['onset_ms']+4,1020,f"Option {iv['option']} visible",fontsize=8,rotation=90,va='top')
    ax.set(ylim=(0,1100),xlim=(0,max(650,rr.rt_ms+45)),ylabel='Gaze distance\nto option (px)',
           title=f'Trial {n}: chose option {rr.chosen_option}, reward {rr.reward_level}; '+
                 (f"{rr.total_visible_before_touch_ms:.1f} ms total stimulus exposure before touch" if intervals else 'no stimulus revealed'))
    ax.grid(axis='y',alpha=.15)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles,labels,ncol=5,fontsize=8,loc='upper center',bbox_to_anchor=(.5,.925))
axes[-1].set_xlabel('Milliseconds after dots / choice window open (eye timestamps: receipt time)')
fig.suptitle('Eye movements can visit dots without revealing their stimuli\nShading = logged stimulus visibility; dotted vertical line = fixation accepted',fontsize=14,y=.99)
fig.tight_layout(rect=[0,0,1,.95])
fig.savefig(OUT/'gaze_examples.png',dpi=180)
plt.close(fig)

print(json.dumps({k:v for k,v in metrics.items() if k!='input_hashes_sha256'},indent=2))
print('EXAMPLE_TRIAL',example)
print('REVEALS\n',R.drop(columns=['onset_task_time_ns','offset_task_time_ns']).round(2).to_string(index=False))

# Standalone report: image references and downloadable data are local relatives.
report_rows=[]
for r in T[T.any_reveal].itertuples():
    rr=R[R.trial_num==r.trial_num]
    report_rows.append({'Trial':r.trial_num,'Chosen option':r.chosen_option,'Reward':r.reward_level,
                       'Touch after dots (ms)':round(r.rt_ms,1),
                       'Visible intervals after dots (ms)':'; '.join(f'{v.onset_ms:.1f}–{v.offset_ms:.1f}' for v in rr.itertuples()),
                       'Exposure before touch (ms)':round(r.total_visible_before_touch_ms,1)})
reveal_table=pd.DataFrame(report_rows).to_html(index=False,border=0)
performance_rows=[]
for title,key in [('Juice rewarded','rewarded'),('Best available reward, ties count','best_with_ties'),
                  ('Total pump pulses','pump_pulses'),('Sum of reward levels','reward_sum')]:
    pp=metrics['performance']['all'][key]
    performance_rows.append({'Measure':title,'Observed':pp['observed'],'Random-choice expectation':round(pp['expected'],2),
                             'Exact one-sided p':round(pp['upper_tail_p'],4)})
performance_table=pd.DataFrame(performance_rows).to_html(index=False,border=0)
body=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>CSC2 classic: timing and choice analysis</title>
<style>body{{max-width:1120px;margin:40px auto;padding:0 24px;font:17px/1.6 system-ui,sans-serif;color:#213140;background:#fff}}
h1{{font-size:32px;line-height:1.2}}h2{{font-size:23px;margin-top:36px}}p{{max-width:1000px}}img{{width:100%;height:auto}}
table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:10px;text-align:left;border-bottom:1px solid #d9e1e7}}
th{{background:#eef3f7}}a{{color:#1262a0}}code{{font-size:.9em;background:#f1f4f6;padding:2px 4px}}
.lead{{font-size:21px;border-left:5px solid #de7627;padding-left:20px}}.note{{color:#506275;font-size:15px}}
@media print{{body{{margin:0;max-width:none}}img{{max-height:850px;object-fit:contain}}}}</style>
<h1>Most choices occurred before any stimulus was revealed</h1>
<p class="note">Yuri · 7 November 2026 · experiment exp_Y_20261107_006 · block 2_csc2_foraging_classic</p>
<p class="lead">The data support rapid selection of unrevealed dots on most trials, compatible with guessing or a spatial/motor strategy.
They do not support rapid visual comparison of multiple color–shape stimuli. There were 72 completed trials;
63 (87.5%) had zero logged stimulus exposure before the choice. Performance was not convincingly above the relevant chance baseline.</p>
<p>The remaining nine trials contained ten reveals. Every reveal was of the subsequently chosen option; trial 55 revealed that same option twice.
No trial revealed two different stimuli. Trial 73 was aborted during initiation and is excluded from completed-trial analyses.</p>
<h2>What this configuration actually does</h2>
<p>The saved block configuration has <code>sequential=false</code>, <code>is_memory=false</code>, <code>duration=0</code>,
<code>wait_for_saccade_on_dot=true</code>, a 175-pixel fixation radius and a 300-ms continuous dwell criterion.
All three dots appear together and the 2-second choice window opens on that same flip, on all 72 trials.
Touch targets for all options are already active, even before any image has appeared.</p>
<p>A stimulus is revealed only after continuous accepted gaze in its window; leaving that window restores its dot.
The 100-ms ISI is a minimum waiting period, not a guaranteed image presentation.
The 4-second saccade timeout cannot precede the 2-second choice deadline in this configuration.
A touch ends the trial even if no fixation was accepted and no stimulus was revealed.</p>
<p>This behavior follows <a href="../../bin/utils.py">bin/utils.py</a>:
<code>gaze_choice_at_dot_onset</code> and the simultaneous-dot branch (around line 3331),
<code>_match_choice_target</code> (2749), and <code>_run_gaze_contingent_options</code> (3039).
The fixation gate is in <a href="../../bin/dot_fixation.py">bin/dot_fixation.py</a> (61–102).</p>
<h2>Timing of choices, gaze and flips</h2>
<ul><li>Dot onset to touch: median <b>511.5 ms</b>, interquartile range 478.6–568.5 ms, range 418.6–1097.2 ms.</li>
<li>Trials with no reveal: median 506.9 ms. Trials with a reveal: median 901.6 ms.</li>
<li>First valid gaze sample inside any option window: median <b>238.1 ms</b> after dots appeared, using eye reception timestamps.</li>
<li>On no-reveal trials, the longest continuous in-window sample span was typically only <b>159.9 ms</b> (median); the maximum was 259.8 ms, below the 300-ms requirement.</li>
<li>Accepted fixation to image-on flip: median 12.1 ms, range 11.6–19.7 ms. The rare reveals were not being held back by hundreds of milliseconds of flip latency.</li>
<li>Touch to the post-choice gray flip: median 14.3 ms, range 10.0–42.6 ms.</li>
<li>The nominal display rate was 59.46 Hz (16.82 ms per refresh). Dot-flip submission to callback took a median 14.6 ms;
image-on submission to callback took 10.0 ms. These are submission latencies, not missed-frame estimates.</li></ul>
<p>Using nearest-option ROI visits spanning approximately 60 ms or more, <b>36/72 trials</b> visited at least two option locations;
<b>34 of those had no stimulus reveal</b>. The median retained visit lasted 119.9 ms.
Even restricting to transitions between option windows whose centers were more than 350 pixels apart, 27 trials had a transition.
Thus rapid movement between locations is real in the recording, but it usually samples dots.</p>
<p><b>Trial 70 is especially clear:</b> gaze visited option 3 at 287–397 ms, option 2 at 407–507 ms, then option 1 at 527–629 ms;
each visit spanned about 100 ms on the camera clock. The monkey touched option 1 at 645 ms and earned reward level 2.
All three locations had remained dots throughout.</p>
<img src="timing_overview.png" alt="All-trial gaze visits and stimulus visibility, reaction times, and exact chance reward distribution">
<p class="note">Colored segments are retained gaze visits, black segments are stimulus exposure before the touch, and red marks are touches.
The reward distribution conditions on each trial’s actual three reward values.</p>
<h2>The nine trials with actual reveals</h2>
<p>All values below are relative to that trial’s dot-on flip. Visibility ends at an explicit option-off event, or at the gray-screen outcome flip
when no separate option-off event was logged. Exposure before touch is capped at the touch timestamp.</p>
{reveal_table}
<p>Trials 12, 18 and 48 provided only 3.0, 0.6 and 5.2 ms between the image-on callback and recorded touch.
Those times do not provide credible evidence that the newly revealed image drove the touch already being executed.
Trials 43 and 57 had 59.5 and 36.4 ms before touch. Trials 1, 45, 55 and 66 had at least 100 ms total exposure;
some use of stimulus information on these few trials remains possible. Across the nine reveal trials the median total pre-touch exposure was 59.5 ms.</p>
<p>In trial 55, gaze loss caused the image to disappear after 41.5 ms, and a later accepted dwell revealed the same option again for 65.1 ms before touch.
This was reacquisition of one option, not comparison of two visible stimuli.</p>
<img src="gaze_examples.png" alt="Eye distance traces for trials 70, 12, 1 and 55 aligned to display flips and touch">
<h2>Why the reward rate looked above chance</h2>
<p>Juice reward is associated with levels 2 and 3, not with one uniquely correct option per trial. A 1/3 baseline therefore does not apply to juice reward.
Of the 216 offered options, 143 (66.2%) carried juice. Randomly selecting one of the actual three options on each trial predicts 47.67 rewarded choices;
51 were observed (70.8%). The exact upper-tail probability is <b>p = 0.198</b>.
The central 95% of this null distribution spans 41–54 rewarded trials, which includes 51.</p>
{performance_table}
<p>Choosing a highest-valued option occurred on 40/72 trials (55.6%), versus 37/72 expected (51.4%), counting ties; p = 0.256.
Mean chosen reward level was 1.861, versus 1.778 expected. Among the 63 no-reveal trials alone, 45 earned juice versus 42.33 expected; p = 0.245.
Thus the main result persists when trials with any potential stimulus information are removed.</p>
<p>The configured frequency map also favors rewarding items: its marginal juice probability is 60.8%; the realized block happened to contain 66.2%.
The primary tests use the realized alternatives, avoiding an assumption that the block exactly matched the sampling distribution.
Reward lookup uses <code>reward_space_TY.csv[shape*36 + color]</code> and reproduces every logged chosen reward (72/72).</p>
<p class="note">The null assumes independent uniform selection among each trial’s options. Failure to reject it does not prove literal random choice:
location preferences, a motor routine, or information use on a small subset can coexist with these results.
An exploratory RT–reward correlation was positive (Spearman ρ = 0.248, unadjusted p = 0.035), but it was also positive in no-reveal trials
(ρ = 0.241, p = 0.057). It does not demonstrate visual discrimination, and was not a prespecified performance test.</p>
<h2>Why the experimenter display can be misleading</h2>
<p>The non-memory post-choice feedback routine <code>_preview_choice_feedback</code> in
<a href="../../task/active_foraging.py">task/active_foraging.py</a> (around line 768) draws all three images on the experimenter preview,
including options that were never revealed to the subject, and highlights the chosen option. It runs after outcome delivery.
Seeing those images in feedback is therefore not evidence they were available to guide the choice.</p>
<p>The saved block also enabled the synchronization bar (147 flips), although the named source config disables it.
Sync-only flips preserve the existing front-buffer scene in <a href="../../bin/sync_display.py">bin/sync_display.py</a>;
they are not option reveals. Their scheduling latencies should not be interpreted as stimulus exposure durations.</p>
<h2>Measurement limits and reproducibility</h2>
<p>The eye recording contains 11,680 observations; its median camera interval is 19.983 ms (about 50 Hz), despite 100-Hz UART polling.
Of the 1,955 observations received between dot onset and touch, 98.9% were valid. There are 244 unrecorded frame sequence numbers across the whole recording
(2.05% of the spanned frames). Missing samples, invalid gaze, and gaps over 60 ms split offline visits.</p>
<p>These are ROI visits and relocations consistent with saccades, not a validated physiological saccade detector. No explicit saccade onset/offset events were logged.
The visit threshold uses first-to-last camera sample duration with 1-ms tolerance for the 19.983-ms period; there are 38, 36, 30 and 19 trials visiting multiple options
at nominal 40-, 60-, 80- and 100-ms thresholds. Assignments use the nearest center within 175 pixels to avoid double-counting overlapping windows;
40 trials had at least two overlapping fixation windows. Online gating evaluates each ROI independently.</p>
<p>CSV <code>task_time_ns</code> records eye-sample reception time. The online gate uses an age-adjusted estimate that is not saved in this CSV.
The camera clock is not synchronized to the task clock. Relative eye-to-display alignment therefore includes acquisition/transport uncertainty,
and these data cannot justify millisecond-precise physiological saccade latencies.
The saved calibration has no multi-point residual error estimate, so window crossings remain calibration-dependent.</p>
<p>Display timings are PsychoPy flip callbacks, not photodiode measurements; touch times are task-recorded input timestamps.
The zero-exposure finding relies on the completeness of event logging and its agreement with the inspected local task implementation.
The run does not identify a saved source commit. The inspected checkout is recorded in <code>metrics.json</code>, along with SHA-256 hashes of inputs and relevant source files.
The local source modifications present at analysis start were left intact; no task code or data on the volume was changed.</p>
<p>The behavior-log “zero transition misses” values are not an independent guarantee of display timing: all 72 trials also report zero scheduled transitions,
because this path is driven by gaze and touch. The analysis uses actual option-on/off and outcome flip timestamps instead.</p>
<h2>Implication for the task</h2>
<p>If the intended task requires informed visual choice, the current response rule does not enforce it.
For gaze-contingent inspection, the direct correction is to make an option eligible only after a logged reveal and a defined minimum visible interval,
and log attempted touches rejected for insufficient exposure. Whether all alternatives must be inspected is a separate task-design decision.
Simply extending the choice window or reducing the dwell threshold does not prevent early blind touches.</p>
<p>If “classic” instead means a simultaneously visible array, disabling <code>wait_for_saccade_on_dot</code> selects the ordinary simultaneous-image path.
That changes task semantics; no configuration or task implementation was changed in this analysis.</p>
<p><a href="trial_timing.csv">All 72 trial timings</a> · <a href="stimulus_reveals.csv">All 10 reveal intervals</a> ·
<a href="gaze_visits_60ms.csv">Retained gaze visits</a> · <a href="gaze_roi_runs_all.csv">All ROI runs</a> ·
<a href="metrics.json">Metrics and input hashes</a> · <a href="analyze.py">Reproducible analysis script</a></p>
<p class="note">Run: <code>python analyze.py</code> with numpy, pandas, scipy and matplotlib installed. The source data path is set at the top of the script.
All outputs are written beside this script. Tests here are consistency checks of timestamps, all 72 reward lookups, and exact finite-sample null calculations.</p>
</html>'''
(OUT/'report.html').write_text(body)

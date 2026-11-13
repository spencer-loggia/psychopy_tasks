import csv
import datetime as dt
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from bin.block_sync import (
    EXPERIMENT_NUMBER_ENV, BLOCK_NUMBER_ENV, EXPERIMENT_DATE_ENV, VIDEO_PATH_ENV,
    configure_task_sync, identity_from_environment, require_eye_acquisition, validate_identity,
)
from bin.eye_tracker_warnings import confirm_without_eye_tracking, _confirm_main
from bin.logger import SessionLogBundle
from bin.screen import ScreenGeometry, ExperimenterPreview


class BlockSyncTests(unittest.TestCase):
    def test_experimenter_status_stays_static_during_codeword(self):
        preview = object.__new__(ExperimenterPreview)
        preview._sync_bar_shared = [0.0] * 4
        preview.configure_sync_bar(20, (800, 600))
        preview.set_sync_codeword(True)
        for phase in (1, 0, 1, 0, 1):
            preview.set_sync_bar_phase(phase)
            self.assertEqual(preview._sync_bar_shared[3], 2)
        preview.set_sync_codeword(False)
        self.assertEqual(preview._sync_bar_shared[3], 1)

    def test_identity_uses_experiment_date_across_midnight_and_rejects_overflow(self):
        with patch.dict(os.environ, {EXPERIMENT_NUMBER_ENV:'255', BLOCK_NUMBER_ENV:'2',
                                   EXPERIMENT_DATE_ENV:'2026-11-13'}, clear=True):
            self.assertEqual(identity_from_environment(dt.datetime(2026,11,14)),
                             (255,2,dt.date(2026,11,13)))
        for experiment,block,date in [(256,1,dt.date(2026,1,1)),(1,256,dt.date(2026,1,1)),
                                       (1,1,dt.date(2128,1,1))]:
            with self.assertRaises(ValueError): validate_identity(experiment,block,date)

    def test_unavailable_acquisition_requires_explicit_confirmation(self):
        tracker=Mock(); tracker.wait_until_ready.side_effect=TimeoutError('no frames')
        for accepted in (False, True):
            with patch('bin.eye_tracker_warnings.confirm_without_eye_tracking',return_value=accepted) as confirm:
                if accepted: require_eye_acquisition(tracker)
                else:
                    with self.assertRaisesRegex(RuntimeError,'cancelled'): require_eye_acquisition(tracker)
                confirm.assert_called_once()
        tracker.wait_until_ready.side_effect=None
        with patch('bin.eye_tracker_warnings.confirm_without_eye_tracking') as confirm:
            require_eye_acquisition(tracker)
            confirm.assert_not_called()

    def test_disabled_sync_still_requires_eye_confirmation(self):
        tracker=Mock(); tracker.wait_until_ready.side_effect=RuntimeError('offline')
        with patch('bin.eye_tracker_warnings.confirm_without_eye_tracking',return_value=False), \
             patch('bin.sync_display.configure_sync_bar') as configure:
            with self.assertRaisesRegex(RuntimeError,'cancelled'):
                configure_task_sync(object(), {'enabled':False}, eye_tracker=tracker)
            configure.assert_not_called()

    def test_readiness_runs_after_display_setup_before_header(self):
        events = []
        tracker = Mock()
        tracker.wait_until_ready.side_effect = lambda **kwargs: events.append('eye')
        display = Mock()
        def configure(*args, **kwargs):
            events.append('display_setup')
            kwargs['before_start']()
            events.append('header')
            return display
        with patch.dict(os.environ, {VIDEO_PATH_ENV: '/camera.h264'}, clear=True), \
             patch('bin.sync_display.configure_sync_bar', side_effect=configure), \
             patch('video.recorder.wait_for_fresh_picture', side_effect=lambda path: events.append('video')):
            win = SimpleNamespace()
            configure_task_sync(win, {'enabled': True}, eye_tracker=tracker)
            win._neuro_tasks_finish_sync()
        self.assertEqual(events, ['display_setup', 'eye', 'video', 'eye', 'header'])

    def test_footer_finishes_once_before_eye_and_log_shutdown(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,{},clear=True):
            logs=SessionLogBundle(output_root=tmp, task_name='active_foraging',config_name='sync')
            tracker=Mock(); logs.eye_tracking=tracker
            display=Mock(); events=[]
            display.finish_block_signal.side_effect=lambda date: events.append('footer')
            tracker.recorder.stop_accepting.side_effect=lambda: events.append('stop_samples')
            tracker.close.side_effect=lambda: events.append('stop_tracker')
            with patch('bin.sync_display.configure_sync_bar', side_effect=lambda *a, **kw: (kw['before_start'](), display)[1]) as configure:
                win=SimpleNamespace()
                configure_task_sync(win, {'enabled':True}, session_logs=logs)
                tracker.wait_until_ready.assert_called_once()
                self.assertEqual(configure.call_args.kwargs['experiment_number'],0)
                protocol=logs.session_dir/'sync_protocol.json'
                self.assertFalse(json.loads(protocol.read_text())['end_complete'])
                logs.close()
                win._neuro_tasks_finish_sync()
                self.assertEqual(events,['footer','stop_samples','stop_tracker'])
                self.assertTrue(json.loads(protocol.read_text())['end_complete'])
                with logs.event_logger.path.open() as f:
                    self.assertEqual(list(csv.DictReader(f,delimiter='\t'))[-1]['event'],'block_end')

    def test_failed_footer_remains_failed_and_cleanup_still_runs(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,{},clear=True):
            logs=SessionLogBundle(output_root=tmp, task_name='active_foraging',config_name='sync')
            tracker=Mock(); logs.eye_tracking=tracker
            display=Mock(); display.finish_block_signal.side_effect=RuntimeError('GPIO failed')
            with patch('bin.sync_display.configure_sync_bar',return_value=display):
                win=SimpleNamespace()
                configure_task_sync(win, {'enabled':True}, session_logs=logs)
                with self.assertRaisesRegex(RuntimeError,'GPIO'): logs.close()
                tracker.close.assert_called_once()
                with self.assertRaisesRegex(RuntimeError,'GPIO'): win._neuro_tasks_finish_sync()
                self.assertFalse(json.loads((logs.session_dir/'sync_protocol.json').read_text())['end_complete'])

    def test_confirmation_failure_or_close_does_not_authorize(self):
        for code in (1,2,-9):
            with patch('bin.eye_tracker_warnings.subprocess.run',return_value=SimpleNamespace(returncode=code)):
                self.assertFalse(confirm_without_eye_tracking('offline'))
        root=Mock()
        screens=(ScreenGeometry(0,0,0,1920,1080),ScreenGeometry(1,1920,0,800,480))
        with patch('tkinter.Tk',return_value=root), patch('bin.screen.resolve_task_screens',return_value=screens), \
             patch('bin.tk_dialogs.choose_dialog',return_value=False) as choose:
            self.assertEqual(_confirm_main('offline',None),1)
            root.geometry.assert_called_once_with('760x460+1940+20')
            self.assertFalse(choose.call_args.kwargs['cancel_value'])

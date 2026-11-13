"""Reproduce remote-start/local-timeout, blocked first block, then recovery.

The peer injects lost replies; this proves the client consequence, not the
hardware cause of a particular reported failure.
"""
import datetime as dt
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from bin.eye_tracker_wire.controller import ResponseTimeout
from bin.eye_tracker_wire.protocol import CommandCode, StatusFlag, decode_command
from bin.eye_tracking import (
    EXPERIMENT_BLOCK_NUMBER_ENV, EXPERIMENT_UART_PENDING_ENV,
    EXPERIMENT_UART_READY_ENV, UARTEyeTracker,
)
from interface.experiment_eye_tracker import ExperimentEyeTracker
from tests.test_eye_uart import status_bytes


class StatefulPeer:
    """Idempotent remote state, with a first-experiment acknowledgement outage."""
    def __init__(self, failed_stage, dropped_replies=None):
        self.failed_stage = failed_stage
        self.dropped_replies = dropped_replies
        self.seen = set()
        self.commands = []
        self.experiments_started = 0
        self.tracking_started = 0
        self.active = False
        self.recording = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def close(self):
        pass

    def exchange(self, packet, timeout_s):
        command = decode_command(packet)
        self.commands.append(command)
        if packet not in self.seen:
            self.seen.add(packet)
            if command.code == CommandCode.START_EXPERIMENT:
                self.experiments_started += 1
                self.active = True
            elif command.code == CommandCode.STOP_EXPERIMENT:
                self.active = self.recording = False
            elif command.code == CommandCode.START_TRACKING:
                assert self.active
                self.tracking_started += 1
                self.recording = True
            elif command.code == CommandCode.STOP_TRACKING:
                self.recording = False
        if self.experiments_started == 1 and command.code == self.failed_stage:
            if self.dropped_replies is None or self.dropped_replies > 0:
                if self.dropped_replies is not None:
                    self.dropped_replies -= 1
                raise ResponseTimeout('injected missing acknowledgement')
        flags = StatusFlag.CONFIGURED
        if self.active:
            flags |= StatusFlag.EXPERIMENT_ACTIVE
        if self.recording:
            flags |= StatusFlag.TRACKING | StatusFlag.RECORDING
        return status_bytes(sequence=command.sequence, edits=((4, int(flags)), (5, 0)))


class FirstExperimentFailureTests(unittest.TestCase):
    config = {'eye_tracker_experiment': {
        'enabled': True, 'start_timeout_s': .02,
        'uart': {'device': '/fake', 'response_timeout_s': .01, 'state_timeout_s': .02},
    }}

    def start(self, directory, number):
        directory.mkdir()
        experiment = ExperimentEyeTracker(
            self.config, subject='Test', started_at=dt.datetime(2026,11,13),
            directory=directory, experiment_number=number,
            experiment_id=f'a0000000-0000-4000-8000-{number:012d}',
        )
        experiment.start_future.result(timeout=2)
        return experiment

    def run_block(self, experiment, peer):
        with patch.dict(os.environ, {
            EXPERIMENT_UART_PENDING_ENV: str(experiment.pending_path),
            EXPERIMENT_UART_READY_ENV: str(experiment.ready_path),
            EXPERIMENT_BLOCK_NUMBER_ENV: '1',
        }, clear=True), patch('bin.eye_tracking.SerialPacketChannel', return_value=peer) as opened:
            tracker = UARTEyeTracker()
            tracker._on_measurement = lambda state: tracker._stop.set()
            tracker._run()
            return tracker, opened.call_count

    def test_first_timeout_blocks_tracking_until_new_confirmed_experiment(self):
        for stage in (CommandCode.START_EXPERIMENT, CommandCode.SET_EXPERIMENT_ID):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as tmp:
                peer = StatefulPeer(stage)
                with patch('interface.experiment_eye_tracker.SerialPacketChannel', return_value=peer), \
                     patch('bin.eye_tracker_warnings.show_eye_tracker_warning'):
                    first = self.start(Path(tmp)/'first', 1)
                    self.assertTrue(peer.active, 'Remote experiment really did start')
                    self.assertFalse(first.ready_path.exists())
                    self.assertFalse(first.pending_path.exists())
                    tracker, opens = self.run_block(first, peer)
                    self.assertEqual(opens, 0, 'First block does not even open UART')
                    self.assertEqual(peer.tracking_started, 0)
                    self.assertIn('experiment start was not confirmed', tracker.get_state().error)
                    first.finish()
                    first.stop_future.result(timeout=2)
                    self.assertFalse(peer.active)
                    second = self.start(Path(tmp)/'second', 2)
                    self.assertTrue(second.ready_path.exists())
                    _, opens = self.run_block(second, peer)
                    self.assertEqual(opens, 1)
                    self.assertEqual(peer.tracking_started, 1)
                    self.assertEqual(peer.experiments_started, 2)
                    second.finish()
                    second.stop_future.result(timeout=2)

    def test_recent_final_ack_recovery_handles_two_lost_replies_without_restart(self):
        for stage in (CommandCode.START_EXPERIMENT, CommandCode.SET_EXPERIMENT_ID):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as tmp:
                peer = StatefulPeer(stage, dropped_replies=2)
                with patch('interface.experiment_eye_tracker.SerialPacketChannel', return_value=peer), \
                     patch('bin.eye_tracker_warnings.show_eye_tracker_warning') as warning:
                    experiment = self.start(Path(tmp)/'first', 1)
                    self.assertTrue(experiment.ready_path.exists())
                    warning.assert_not_called()
                    self.assertEqual(peer.experiments_started, 1)
                    attempts = [c for c in peer.commands if c.code == stage]
                    self.assertEqual(len(attempts), 3)
                    self.assertTrue(all(c == attempts[0] for c in attempts))
                    experiment.finish()
                    experiment.stop_future.result(timeout=2)

#!/usr/bin/env python3
"""Test the timeout buzzer using a Pi-Plates DAQC2 DOUT output."""
import argparse
import sys
import time
from pathlib import Path


_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from bin.daqc2_outputs import DAQC2DigitalOutputs
from bin.config import load_config
from bin.task_lifecycle import USER_EXIT_CODE


def parse_args():
    parser = argparse.ArgumentParser(description="Pulse the buzzer via DAQC2 DOUT.")
    parser.add_argument("--config", help="Path to JSON config file")
    parser.add_argument("--daq_address", type=int, default=None, help="DAQC2plate address, 0-7")
    parser.add_argument("--buzz_pin", type=int, default=None, help="DAQC2 DOUT bit for the buzzer, 0-7")
    parser.add_argument("--duration", type=float, default=None, help="Pulse duration in seconds")
    parser.add_argument("--daq_module", default=None, help="Python module for the DAQC2plate driver")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config) if args.config else {}

    def _get(name, default):
        value = getattr(args, name)
        return value if value is not None else cfg.get(name, default)

    duration_s = max(0.0, float(_get("duration", 2.0)))
    outputs = DAQC2DigitalOutputs(
        address=int(_get("daq_address", 0)),
        module_name=str(_get("daq_module", "piplates.DAQC2plate")),
    )

    try:
        outputs.open()
        output_on, output_off = outputs.bind_bit(int(_get("buzz_pin", 1)))
    except Exception as exc:
        print(f"ERROR: Could not initialize DAQC2 DOUT output: {exc}", file=sys.stderr)
        print("Install the Pi-Plates driver and check the DAQC2plate address.", file=sys.stderr)
        sys.exit(1)

    exit_code = 0
    try:
        output_on()
        print(
            f"Testing buzzer on DAQC2 address {int(_get('daq_address', 0))} DOUT{int(_get('buzz_pin', 1))} "
            f"for {duration_s:.3f} seconds."
        )
        print("DOUT is open-drain: logical on pulls the terminal near 0 V; off releases it high.")
        time.sleep(duration_s)
    except KeyboardInterrupt:
        print("\nInterrupted; turning buzzer off.")
        exit_code = USER_EXIT_CODE
    except Exception as exc:
        print(f"\nERROR: Buzzer pulse failed: {exc}", file=sys.stderr)
        exit_code = 1
    finally:
        try:
            output_off()
            print("Buzzer DOUT cleared.")
        except Exception as exc:
            print(f"ERROR: Failed to clear buzzer DOUT: {exc}", file=sys.stderr)
            exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()

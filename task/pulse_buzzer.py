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


def parse_args():
    parser = argparse.ArgumentParser(description="Pulse the buzzer via DAQC2 DOUT.")
    parser.add_argument("--daq_address", type=int, default=0, help="DAQC2plate address, 0-7")
    parser.add_argument("--buzz_pin", type=int, default=1, help="DAQC2 DOUT bit for the buzzer, 0-7")
    parser.add_argument("--duration", type=float, default=2.0, help="Pulse duration in seconds")
    parser.add_argument("--daq_module", default="piplates.DAQC2plate", help="Python module for the DAQC2plate driver")
    return parser.parse_args()


def main():
    args = parse_args()
    duration_s = max(0.0, float(args.duration))
    outputs = DAQC2DigitalOutputs(
        address=int(args.daq_address),
        module_name=str(args.daq_module),
    )

    try:
        outputs.open()
    except Exception as exc:
        print(f"ERROR: Could not initialize DAQC2 DOUT output: {exc}", file=sys.stderr)
        print("Install the Pi-Plates driver and check the DAQC2plate address.", file=sys.stderr)
        sys.exit(1)

    print(
        f"Testing buzzer on DAQC2 address {int(args.daq_address)} DOUT{int(args.buzz_pin)} "
        f"for {duration_s:.3f} seconds."
    )
    print("DOUT is open-drain: logical on pulls the terminal near 0 V; off releases it high.")

    exit_code = 0
    try:
        outputs.write(int(args.buzz_pin), True)
        time.sleep(duration_s)
    except KeyboardInterrupt:
        print("\nInterrupted; turning buzzer off.")
    except Exception as exc:
        print(f"\nERROR: Buzzer pulse failed: {exc}", file=sys.stderr)
        exit_code = 1
    finally:
        try:
            outputs.write(int(args.buzz_pin), False)
            print("Buzzer DOUT cleared.")
        except Exception as exc:
            print(f"ERROR: Failed to clear buzzer DOUT: {exc}", file=sys.stderr)
            exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()

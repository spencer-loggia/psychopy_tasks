#!/usr/bin/env python3
"""
Simple test script to send a 5-second 5V pulse through Raspberry Pi GPIO pin 18.

Usage:
    python task/test_raspi_pulse.py

Requirements:
    - pigpio daemon must be running: sudo pigpiod
    - Run with appropriate permissions for GPIO access
"""
import time
import sys

def main():
    try:
        import pigpio
    except ImportError:
        print("ERROR: pigpio not installed. Install with: pip install pigpio")
        sys.exit(1)

    # GPIO pin to use (BCM numbering)
    PIN = 18
    PULSE_DURATION_S = 5.0

    print(f"Connecting to pigpio daemon...")
    pi = pigpio.pi()

    if not pi.connected:
        print("ERROR: Could not connect to pigpio daemon.")
        print("Make sure pigpiod is running: sudo pigpiod")
        sys.exit(1)

    print(f"Connected to pigpio daemon.")
    print(f"Sending 5V pulse on GPIO pin {PIN} for {PULSE_DURATION_S} seconds...")

    try:
        # Set pin to output mode
        pi.set_mode(PIN, pigpio.OUTPUT)

        # Send high (5V on Raspberry Pi)
        pi.write(PIN, 1)
        start = time.time()
        print(f"Pin {PIN} set HIGH at {time.strftime('%H:%M:%S')}")

        # Wait for pulse duration
        time.sleep(PULSE_DURATION_S)

        # Send low (0V)
        pi.write(PIN, 0)
        end = time.time()
        actual_duration = end - start
        print(f"Pin {PIN} set LOW at {time.strftime('%H:%M:%S')}")
        print(f"Actual pulse duration: {actual_duration:.3f} seconds")

    except KeyboardInterrupt:
        print("\nInterrupted by user, cleaning up...")
        pi.write(PIN, 0)
    except Exception as e:
        print(f"ERROR: {e}")
        pi.write(PIN, 0)
    finally:
        # Cleanup
        pi.stop()
        print("Disconnected from pigpio daemon.")

if __name__ == "__main__":
    main()

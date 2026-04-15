#!/usr/bin/env python3
"""
Simple test script to send a 5-second pulse through Raspberry Pi GPIO pin 18.

This demonstrates hardware-timed GPIO pulse using lgpio's tx_pulse function.
For integration with PsychoPy visual flips, use win.callOnFlip() to ensure
the GPIO write happens at the exact moment the frame is presented (minimizing
latency between visual stimulus and GPIO pulse).

Usage:
    python task/test_raspi_pulse.py

Requirements:
    - lgpio library installed: pip install lgpio
    - Run with appropriate permissions for GPIO access
"""
import time
import sys

def main():
    try:
        import lgpio
    except ImportError:
        print("ERROR: lgpio not installed. Install with: pip install lgpio")
        sys.exit(1)

    # GPIO pin to use (BCM numbering)
    PIN = 18
    PULSE_DURATION_S = 5.0

    print(f"Opening GPIO chip...")
    try:
        chip = lgpio.gpiochip_open(0)  # 0 is the default chip for RPi5
    except Exception as e:
        print(f"ERROR: Could not open GPIO chip: {e}")
        print("Make sure you have permission to access GPIO (may need sudo)")
        sys.exit(1)

    print(f"GPIO chip opened successfully.")
    print(f"Sending pulse on GPIO pin {PIN} for {PULSE_DURATION_S} seconds...")
    print(f"Using hardware-timed pulse for microsecond precision.\n")

    try:
        # Claim pin as output
        lgpio.gpio_claim_output(chip, PIN)

        # Send high immediately (3.3V on Raspberry Pi 5)
        lgpio.gpio_write(chip, PIN, 1)
        start = time.time()
        print(f"Pin {PIN} set HIGH at {time.strftime('%H:%M:%S.%f')[:-3]}")

        # Use hardware-timed pulse to turn off after duration
        # tx_pulse(handle, gpio, pulse_on, pulse_off, pulse_offset=0, pulse_cycles=0)
        # pulse_on=0 means turn off immediately after current state
        # pulse_off=duration means stay off for duration
        # pulse_cycles=1 means execute once
        duration_us = int(PULSE_DURATION_S * 1_000_000)
        result = lgpio.tx_pulse(chip, PIN, 0, duration_us, 0, 1)
        
        if result < 0:
            raise RuntimeError(f"tx_pulse failed with error code {result}")
            
        print(f"Hardware pulse scheduled for {duration_us} microseconds")
        
        # Wait for pulse to complete
        time.sleep(PULSE_DURATION_S + 0.1)
        
        end = time.time()
        actual_duration = end - start
        print(f"\nPin {PIN} turned LOW (hardware-timed)")
        print(f"Measured duration from script: {actual_duration:.3f} seconds")
        print(f"Note: Hardware timing is precise to microseconds,")
        print(f"      independent of Python's timing inaccuracies.")

    except KeyboardInterrupt:
        print("\nInterrupted by user, cleaning up...")
        try:
            lgpio.gpio_write(chip, PIN, 0)
        except Exception:
            pass
    except Exception as e:
        print(f"\nERROR: Hardware pulse failed: {e}")
        print("\nHardware-timed GPIO pulse is required for timing precision.")
        print("This may indicate:")
        print("  - lgpio version does not support tx_pulse")
        print("  - Hardware PWM/timing features not available")
        print("  - Insufficient permissions")
        print(f"  - Error details: {e}")
        print("\nPlease check your lgpio installation and hardware support.")
        try:
            lgpio.gpio_write(chip, PIN, 0)
        except Exception:
            pass
    finally:
        # Cleanup
        try:
            lgpio.gpiochip_close(chip)
            print("\nGPIO chip closed.")
        except Exception:
            pass

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Check TensorBoard Event Files
Reads and displays metrics from TensorBoard event files to verify logging
"""

from tensorboard.backend.event_processing import event_accumulator
import os
import sys
import glob

def check_event_file(event_file):
    """Check a single event file and display metrics"""
    print("=" * 70)
    print("Reading TensorBoard Event File")
    print("=" * 70)
    print(f"File: {event_file}")
    print(f"File size: {os.path.getsize(event_file) / 1024:.1f} KB")
    print()

    # Load the event file
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()

    # Show what's available
    print("Available scalar metrics:")
    scalars = ea.Tags().get('scalars', [])
    for tag in scalars:
        print(f"  - {tag}")
    print()

    # Check loss metrics
    loss_tags = [tag for tag in scalars if 'loss' in tag.lower()]
    if loss_tags:
        print("=" * 70)
        print("LOSS METRICS")
        print("=" * 70)
        for tag in loss_tags:
            events = ea.Scalars(tag)
            if events:
                print(f"\n{tag}:")
                print(f"  Total data points: {len(events)}")
                print(f"  Step range: {events[0].step} to {events[-1].step}")
                print(f"  Latest values:")
                for e in events[-10:]:  # Last 10
                    print(f"    Step {e.step:6d}: {e.value:.6f}")
    else:
        print("⚠️  No loss metrics found!")

    # Check all scalar metrics
    print("\n" + "=" * 70)
    print("ALL METRICS SUMMARY")
    print("=" * 70)
    for tag in scalars[:20]:  # First 20 metrics
        events = ea.Scalars(tag)
        if events:
            print(f"{tag:40s} | Points: {len(events):4d} | Latest step: {events[-1].step:6d}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    if scalars:
        max_step = max([ea.Scalars(tag)[-1].step for tag in scalars if ea.Scalars(tag)])
        print(f"✓ Event file is valid")
        print(f"✓ Contains {len(scalars)} metrics")
        print(f"✓ Latest logged step: {max_step}")
        if max_step < 1000:
            print(f"⚠️  Only {max_step} steps logged (expected more)")
        else:
            print(f"✓ Data looks good!")
    else:
        print("❌ No scalar data found in event file")


def find_latest_event_file():
    """Find the most recent event file"""
    pattern = "./outputs/**/events.out.tfevents.*"
    event_files = glob.glob(pattern, recursive=True)
    
    if not event_files:
        print("❌ No event files found!")
        print(f"   Searched: {pattern}")
        return None
    
    # Sort by modification time, most recent first
    event_files.sort(key=os.path.getmtime, reverse=True)
    return event_files[0]


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Use provided event file path
        event_file = sys.argv[1]
    else:
        # Auto-find latest event file
        print("Searching for latest event file...")
        event_file = find_latest_event_file()
        if event_file is None:
            sys.exit(1)
        print(f"Found: {event_file}\n")
    
    if not os.path.exists(event_file):
        print(f"❌ File not found: {event_file}")
        sys.exit(1)
    
    check_event_file(event_file)


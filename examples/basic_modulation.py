#!/usr/bin/env python3
"""
Basic WaveCaster modulation example

This example demonstrates basic signal modulation using different schemes.
"""

import sys
from pathlib import Path

# Add parent directory to path to import wavecaster
sys.path.insert(0, str(Path(__file__).parent.parent))

from wavecaster import (
    ModulationConfig,
    ModulationScheme,
    DigitalModulator,
    ContentAnalyzer,
    text_to_bits,
)
import numpy as np


def example_bfsk_modulation():
    """Example: BFSK modulation"""
    print("=" * 60)
    print("Example 1: BFSK Modulation")
    print("=" * 60)

    # Create configuration
    config = ModulationConfig(
        sample_rate=44100,
        carrier_freq=1000,
        bit_rate=100
    )

    # Text to modulate
    text = "Hello, WaveCaster!"
    print(f"\nText: {text}")

    # Convert to bits
    bits = text_to_bits(text)
    print(f"Bits: {bits[:50]}... (showing first 50)")

    # Modulate
    signal = DigitalModulator.bfsk(bits, config)
    print(f"\nSignal length: {len(signal)} samples")
    print(f"Signal duration: {len(signal) / config.sample_rate:.2f} seconds")
    print(f"Signal range: [{signal.min():.3f}, {signal.max():.3f}]")


def example_qpsk_modulation():
    """Example: QPSK modulation"""
    print("\n" + "=" * 60)
    print("Example 2: QPSK Modulation")
    print("=" * 60)

    config = ModulationConfig()
    text = "QPSK is more efficient than BFSK"
    print(f"\nText: {text}")

    bits = text_to_bits(text)

    # Ensure even number of bits for QPSK
    if len(bits) % 2 != 0:
        bits.append(0)

    signal = DigitalModulator.qpsk(bits, config)
    print(f"Signal length: {len(signal)} samples")
    print(f"Bits per symbol: 2")
    print(f"Symbols: {len(bits) // 2}")


def example_adaptive_modulation():
    """Example: Content-aware adaptive modulation"""
    print("\n" + "=" * 60)
    print("Example 3: Adaptive Modulation")
    print("=" * 60)

    analyzer = ContentAnalyzer()

    # Test with different texts
    texts = [
        "Simple text",
        "Complex text with varied structure and symbols: !@#$%^&*()",
        "aaaaaaaaaaaaaaaaaaaaaa",  # Low entropy
    ]

    for text in texts:
        print(f"\nAnalyzing: \"{text}\"")
        analysis = analyzer.analyze(text)

        print(f"  Entropy: {analysis['entropy']:.3f}")
        print(f"  Complexity: {analysis['complexity']:.3f}")
        print(f"  Redundancy: {analysis['redundancy']:.3f}")

        # Recommend scheme based on analysis
        if analysis['entropy'] < 0.3:
            scheme = ModulationScheme.BFSK
        elif analysis['entropy'] < 0.6:
            scheme = ModulationScheme.BPSK
        else:
            scheme = ModulationScheme.QPSK

        print(f"  Recommended scheme: {scheme.name}")


def example_signal_quality():
    """Example: Signal quality metrics"""
    print("\n" + "=" * 60)
    print("Example 4: Signal Quality Metrics")
    print("=" * 60)

    from wavecaster import SignalUtils

    config = ModulationConfig()
    text = "Quality test message"
    bits = text_to_bits(text)

    # Generate signal
    signal = DigitalModulator.bpsk(bits, config)

    # Calculate SNR
    snr = SignalUtils.calculate_snr(signal)
    print(f"\nSignal-to-Noise Ratio: {snr:.2f} dB")

    # Add noise and recalculate
    noisy_signal = SignalUtils.add_awgn(signal, snr_db=10)
    noisy_snr = SignalUtils.calculate_snr(noisy_signal)
    print(f"Noisy Signal SNR: {noisy_snr:.2f} dB")


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("WaveCaster - Basic Modulation Examples")
    print("=" * 60)

    try:
        example_bfsk_modulation()
        example_qpsk_modulation()
        example_adaptive_modulation()
        example_signal_quality()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

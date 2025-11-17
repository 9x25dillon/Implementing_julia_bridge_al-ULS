#!/usr/bin/env python3
"""
CCL Analysis Example

This example demonstrates how to use the CCL (Categorical Coherence Linter)
to analyze Python code for logical inconsistencies and ghost detection.
"""

import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ccl import analyze, probe, infer_impurity, H


def example_analyze_simple_function():
    """Example: Analyze a simple pure function"""
    print("=" * 60)
    print("Example 1: Analyzing a Pure Function")
    print("=" * 60)

    # Create a simple test module
    test_code = '''
def double(x):
    """Pure function: doubles its input"""
    return x * 2

def add(a, b):
    """Pure function: adds two numbers"""
    return a + b
'''

    # Write test file
    test_file = Path("test_pure.py")
    test_file.write_text(test_code)

    try:
        # Analyze it
        result = analyze(test_file, samples=50, seed=42)

        print(f"\nModules analyzed: {result['summary']['modules']}")
        print(f"Functions analyzed: {result['summary']['functions_analyzed']}")

        # Show top hotspots
        print("\nTop hotspots:")
        for hotspot in result['summary']['top_hotspots']:
            print(f"  {hotspot['function']}: score={hotspot['score']:.3f}")
            print(f"    - Entropy: {hotspot['entropy_bits']:.3f} bits")
            print(f"    - Idempotent rate: {hotspot['idempotent_rate']:.3f}")
            print(f"    - Impure: {hotspot['impure']}")

    finally:
        # Clean up
        test_file.unlink(missing_ok=True)


def example_analyze_impure_function():
    """Example: Analyze an impure function"""
    print("\n" + "=" * 60)
    print("Example 2: Analyzing an Impure Function")
    print("=" * 60)

    test_code = '''
import random

def random_value():
    """Impure: uses random module"""
    return random.random()

def write_file(data):
    """Impure: performs I/O"""
    with open("output.txt", "w") as f:
        f.write(str(data))
    return data
'''

    test_file = Path("test_impure.py")
    test_file.write_text(test_code)

    try:
        result = analyze(test_file, samples=50, seed=42)

        print(f"\nFunctions analyzed: {result['summary']['functions_analyzed']}")

        # Show impurity details
        print("\nImpurity analysis:")
        for func_report in result['functions']:
            purity = func_report['purity']
            if purity.get('impure'):
                print(f"  {func_report['qualname']}:")
                print(f"    - Impure: Yes")
                print(f"    - Reasons: {purity.get('reasons', [])}")

    finally:
        test_file.unlink(missing_ok=True)


def example_entropy_calculation():
    """Example: Calculate entropy of outputs"""
    print("\n" + "=" * 60)
    print("Example 3: Entropy Calculation")
    print("=" * 60)

    # Test with different value sets
    test_cases = [
        ([1, 1, 1, 1, 1], "All same values"),
        ([1, 2, 3, 4, 5], "All different values"),
        ([1, 1, 2, 2, 3, 3], "Some repetition"),
    ]

    for values, description in test_cases:
        entropy = H(values)
        print(f"\n{description}")
        print(f"  Values: {values}")
        print(f"  Entropy: {entropy:.3f} bits")


def example_function_probing():
    """Example: Probe a specific function"""
    print("\n" + "=" * 60)
    print("Example 4: Function Probing")
    print("=" * 60)

    # Define test function
    def square(x):
        """Square function"""
        if isinstance(x, (int, float)):
            return x * x
        return None

    # Probe it
    result = probe(square, samples=100, seed=42)

    print("\nProbing results for square(x):")
    print(f"  Idempotent rate: {result['idempotent_rate']:.3f}")
    print(f"  Entropy: {result['entropy_bits']:.3f} bits")
    print(f"  Sensitivity: {result['sensitivity']:.6f}")
    print(f"  Samples: {result['sample_count']}")
    print(f"  Anomalies: {len(result['anomalies'])}")


def example_ghost_score():
    """Example: Ghost score interpretation"""
    print("\n" + "=" * 60)
    print("Example 5: Ghost Score Interpretation")
    print("=" * 60)

    test_code = '''
def inconsistent(x, state=[]):
    """Function with mutable default argument - potential ghost!"""
    state.append(x)
    return sum(state)

def pure_sum(numbers):
    """Pure function"""
    return sum(numbers)
'''

    test_file = Path("test_ghost.py")
    test_file.write_text(test_code)

    try:
        result = analyze(test_file, samples=100, seed=42)

        print("\nGhost hotspots (highest to lowest):")
        for i, hotspot in enumerate(result['ghost_hotspots'][:5], 1):
            print(f"\n{i}. {hotspot['function']}")
            print(f"   Ghost Score: {hotspot['score']:.3f}")
            print(f"   Entropy: {hotspot['entropy_bits']:.3f} bits")
            print(f"   Idempotent: {hotspot['idempotent_rate']:.1%}")
            print(f"   Impure: {hotspot['impure']}")

            # Interpretation
            if hotspot['score'] > 0.7:
                severity = "🔴 HIGH"
            elif hotspot['score'] > 0.4:
                severity = "🟡 MEDIUM"
            else:
                severity = "🟢 LOW"
            print(f"   Severity: {severity}")

    finally:
        test_file.unlink(missing_ok=True)


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("CCL Analysis Examples")
    print("=" * 60)

    try:
        example_analyze_simple_function()
        example_analyze_impure_function()
        example_entropy_calculation()
        example_function_probing()
        example_ghost_score()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

"""
Comprehensive test suite for WaveCaster signal modulation engine
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import json

# Import WaveCaster components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from wavecaster import (
    ModulationScheme,
    ModulationConfig,
    FECScheme,
    ContentAnalyzer,
    AdaptiveModulationPlanner,
    DigitalModulator,
    SecurityUtils,
    SignalUtils,
)


class TestModulationConfig:
    """Test ModulationConfig dataclass"""

    def test_default_config(self):
        """Test default configuration values"""
        config = ModulationConfig()
        assert config.sample_rate == 44100
        assert config.carrier_freq == 1000
        assert config.bit_rate == 100
        assert config.amplitude == 1.0

    def test_custom_config(self):
        """Test custom configuration"""
        config = ModulationConfig(
            sample_rate=48000,
            carrier_freq=2000,
            bit_rate=200
        )
        assert config.sample_rate == 48000
        assert config.carrier_freq == 2000
        assert config.bit_rate == 200


class TestContentAnalyzer:
    """Test content analysis functionality"""

    def test_analyze_simple_text(self):
        """Test analysis of simple text"""
        analyzer = ContentAnalyzer()
        text = "Hello world"
        analysis = analyzer.analyze(text)

        assert 'entropy' in analysis
        assert 'complexity' in analysis
        assert 'redundancy' in analysis
        assert 'length' in analysis
        assert analysis['length'] == len(text)
        assert 0 <= analysis['entropy'] <= 1
        assert 0 <= analysis['complexity'] <= 1

    def test_analyze_empty_text(self):
        """Test analysis of empty text"""
        analyzer = ContentAnalyzer()
        analysis = analyzer.analyze("")

        assert analysis['length'] == 0
        assert analysis['entropy'] == 0
        assert analysis['complexity'] == 0

    def test_analyze_high_entropy_text(self):
        """Test analysis of high-entropy text"""
        analyzer = ContentAnalyzer()
        # Random-like text with high entropy
        text = "xqz8w9v7u6t5s4r3p2o1n0m"
        analysis = analyzer.analyze(text)

        assert analysis['entropy'] > 0.5  # Should be relatively high

    def test_analyze_low_entropy_text(self):
        """Test analysis of low-entropy text"""
        analyzer = ContentAnalyzer()
        # Repetitive text with low entropy
        text = "aaaaaaaaaaaaaaaaaa"
        analysis = analyzer.analyze(text)

        assert analysis['entropy'] < 0.3  # Should be relatively low


class TestAdaptiveModulationPlanner:
    """Test adaptive modulation planning"""

    def test_select_scheme_for_simple_text(self):
        """Test scheme selection for simple text"""
        planner = AdaptiveModulationPlanner()
        analysis = {
            'entropy': 0.3,
            'complexity': 0.2,
            'redundancy': 0.5,
            'length': 100
        }

        scheme = planner.select_scheme(analysis)
        assert isinstance(scheme, ModulationScheme)
        # Simple text should use simpler schemes like BFSK or BPSK
        assert scheme in [ModulationScheme.BFSK, ModulationScheme.BPSK]

    def test_select_scheme_for_complex_text(self):
        """Test scheme selection for complex text"""
        planner = AdaptiveModulationPlanner()
        analysis = {
            'entropy': 0.8,
            'complexity': 0.9,
            'redundancy': 0.1,
            'length': 500
        }

        scheme = planner.select_scheme(analysis)
        assert isinstance(scheme, ModulationScheme)
        # Complex text can use more advanced schemes
        assert scheme in [ModulationScheme.QPSK, ModulationScheme.QAM16, ModulationScheme.OFDM]


@pytest.mark.unit
class TestDigitalModulator:
    """Test digital modulation functions"""

    def test_bfsk_modulation(self):
        """Test BFSK modulation"""
        config = ModulationConfig()
        bits = [0, 1, 0, 1, 1, 0]
        signal = DigitalModulator.bfsk(bits, config)

        assert len(signal) > 0
        assert signal.dtype == np.float32
        assert np.all(np.isfinite(signal))
        assert -1.5 <= signal.min() <= 1.5
        assert -1.5 <= signal.max() <= 1.5

    def test_bpsk_modulation(self):
        """Test BPSK modulation"""
        config = ModulationConfig()
        bits = [0, 1, 0, 1]
        signal = DigitalModulator.bpsk(bits, config)

        assert len(signal) > 0
        assert signal.dtype == np.float32
        assert np.all(np.isfinite(signal))

    def test_qpsk_modulation(self):
        """Test QPSK modulation"""
        config = ModulationConfig()
        bits = [0, 0, 0, 1, 1, 0, 1, 1]  # Multiple of 2 for QPSK
        signal = DigitalModulator.qpsk(bits, config)

        assert len(signal) > 0
        assert signal.dtype == np.float32
        assert np.all(np.isfinite(signal))

    def test_qam16_modulation(self):
        """Test QAM16 modulation"""
        config = ModulationConfig()
        bits = [0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1]  # Multiple of 4 for QAM16
        signal = DigitalModulator.qam16(bits, config)

        assert len(signal) > 0
        assert signal.dtype == np.float32
        assert np.all(np.isfinite(signal))

    def test_ofdm_modulation(self):
        """Test OFDM modulation"""
        config = ModulationConfig()
        bits = [0, 1] * 32  # Need enough bits for OFDM
        signal = DigitalModulator.ofdm(bits, config)

        assert len(signal) > 0
        assert signal.dtype == np.float32
        assert np.all(np.isfinite(signal))


@pytest.mark.unit
class TestSecurityUtils:
    """Test security utilities"""

    def test_generate_watermark(self):
        """Test watermark generation"""
        watermark = SecurityUtils.generate_watermark()
        assert isinstance(watermark, str)
        assert len(watermark) > 0

    def test_encrypt_decrypt(self):
        """Test encryption and decryption"""
        key = SecurityUtils.generate_key()
        plaintext = b"Test message"

        ciphertext = SecurityUtils.encrypt(plaintext, key)
        assert ciphertext != plaintext

        decrypted = SecurityUtils.decrypt(ciphertext, key)
        assert decrypted == plaintext

    def test_hmac_generation(self):
        """Test HMAC generation"""
        key = SecurityUtils.generate_key()
        data = b"Test data"

        hmac = SecurityUtils.generate_hmac(data, key)
        assert isinstance(hmac, str)
        assert len(hmac) > 0

    def test_hmac_verification(self):
        """Test HMAC verification"""
        key = SecurityUtils.generate_key()
        data = b"Test data"

        hmac = SecurityUtils.generate_hmac(data, key)
        assert SecurityUtils.verify_hmac(data, hmac, key) is True

        # Test with wrong HMAC
        wrong_hmac = "0" * 64
        assert SecurityUtils.verify_hmac(data, wrong_hmac, key) is False


@pytest.mark.unit
class TestSignalUtils:
    """Test signal processing utilities"""

    def test_calculate_snr(self):
        """Test SNR calculation"""
        # Create a clean signal
        signal = np.sin(2 * np.pi * 1000 * np.linspace(0, 1, 44100))
        snr = SignalUtils.calculate_snr(signal)

        assert snr > 0
        assert np.isfinite(snr)

    def test_calculate_evm(self):
        """Test EVM calculation"""
        signal = np.random.randn(1000) + 1j * np.random.randn(1000)
        reference = np.ones(1000, dtype=complex)

        evm = SignalUtils.calculate_evm(signal, reference)
        assert 0 <= evm <= 100
        assert np.isfinite(evm)

    def test_add_awgn(self):
        """Test AWGN addition"""
        signal = np.ones(1000)
        snr_db = 10

        noisy_signal = SignalUtils.add_awgn(signal, snr_db)
        assert len(noisy_signal) == len(signal)
        assert not np.array_equal(signal, noisy_signal)


@pytest.mark.integration
class TestWaveCasterIntegration:
    """Integration tests for full WaveCaster workflow"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_text_to_signal_workflow(self):
        """Test complete text-to-signal workflow"""
        config = ModulationConfig()
        analyzer = ContentAnalyzer()
        planner = AdaptiveModulationPlanner()

        # Analyze text
        text = "Test message for modulation"
        analysis = analyzer.analyze(text)

        # Select scheme
        scheme = planner.select_scheme(analysis)

        # Convert to bits
        bits = [int(b) for byte in text.encode() for b in format(byte, '08b')]

        # Modulate
        if scheme == ModulationScheme.BFSK:
            signal = DigitalModulator.bfsk(bits, config)
        elif scheme == ModulationScheme.BPSK:
            signal = DigitalModulator.bpsk(bits, config)
        elif scheme == ModulationScheme.QPSK:
            # Pad bits if needed
            if len(bits) % 2 != 0:
                bits.append(0)
            signal = DigitalModulator.qpsk(bits, config)

        assert len(signal) > 0
        assert np.all(np.isfinite(signal))


@pytest.mark.performance
class TestPerformance:
    """Performance benchmark tests"""

    def test_bfsk_performance(self, benchmark):
        """Benchmark BFSK modulation"""
        config = ModulationConfig()
        bits = [0, 1] * 50  # 100 bits

        result = benchmark(DigitalModulator.bfsk, bits, config)
        assert len(result) > 0

    def test_qpsk_performance(self, benchmark):
        """Benchmark QPSK modulation"""
        config = ModulationConfig()
        bits = [0, 1] * 50  # 100 bits

        result = benchmark(DigitalModulator.qpsk, bits, config)
        assert len(result) > 0


@pytest.mark.security
class TestSecurityFeatures:
    """Security-focused tests"""

    def test_encryption_strength(self):
        """Test that encryption produces non-predictable output"""
        key = SecurityUtils.generate_key()
        plaintext = b"Secret message"

        # Encrypt same plaintext twice
        ciphertext1 = SecurityUtils.encrypt(plaintext, key)
        ciphertext2 = SecurityUtils.encrypt(plaintext, key)

        # Due to IV, ciphertexts should be different
        assert ciphertext1 != ciphertext2

    def test_watermark_uniqueness(self):
        """Test that watermarks are unique"""
        watermarks = [SecurityUtils.generate_watermark() for _ in range(100)]
        assert len(set(watermarks)) == 100  # All should be unique


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

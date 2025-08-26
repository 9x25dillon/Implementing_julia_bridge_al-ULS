#!/usr/bin/env python3
# integrated_wavecaster.py
# SPDX-License-Identifier: MIT
"""
Integrated WaveCaster System
---------------------------
Dual LLM orchestration + adaptive modulation + signal processing
Features:
- Text generation via local/remote LLM coordination
- Multiple modulation schemes (BFSK, BPSK, QPSK, 16QAM, OFDM)
- Adaptive modulation selection based on content analysis
- Signal quality metrics and visualization
- Security features (encryption, watermarking, HMAC)
- Simple reinforcement learning for modulation optimization

Dependencies:
  pip install numpy scipy requests matplotlib sounddevice pycryptodome
"""

from __future__ import annotations
import argparse, base64, binascii, hashlib, json, logging, math, os, struct, sys, time, warnings
import uuid, zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Callable
from enum import Enum, auto
from datetime import datetime
import re
from collections import Counter

# Hard requirements
try:
    import numpy as np
    from scipy import signal as sp_signal
    from scipy.fft import rfft, rfftfreq, fft, ifft
except ImportError as e:
    raise SystemExit("numpy and scipy required: pip install numpy scipy") from e

# Optional dependencies
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    requests = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import sounddevice as sd
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False

try:
    from Crypto.Cipher import AES
    from Crypto.Random import get_random_bytes
    from Crypto.Protocol.KDF import PBKDF2
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("wavecaster")

# =========================================================
# Configuration Classes
# =========================================================

class ModulationScheme(Enum):
    BFSK = auto()
    BPSK = auto()
    QPSK = auto()
    QAM16 = auto()
    OFDM = auto()
    DSSS_BPSK = auto()

class FECScheme(Enum):
    NONE = auto()
    HAMMING74 = auto()
    REPETITION = auto()

@dataclass
class HTTPConfig:
    base_url: str
    api_key: Optional[str] = None
    model: Optional[str] = None
    timeout: int = 60
    mode: str = "openai-chat"
    verify_ssl: bool = True
    max_retries: int = 2
    retry_delay: float = 0.8

@dataclass
class ModulationConfig:
    sample_rate: int = 48000
    symbol_rate: int = 1200
    amplitude: float = 0.7
    f0: float = 1200.0     # BFSK mark frequency
    f1: float = 2200.0     # BFSK space frequency
    fc: float = 1800.0     # PSK/QAM carrier frequency
    clip: bool = True
    # OFDM parameters
    ofdm_subcarriers: int = 64
    ofdm_cp_length: int = 16
    # DSSS parameters
    dsss_chip_rate: int = 4800

@dataclass
class FrameConfig:
    use_crc32: bool = True
    use_crc16: bool = False
    preamble: bytes = b"\x55" * 8
    version: int = 1

@dataclass
class SecurityConfig:
    password: Optional[str] = None
    watermark: Optional[str] = None
    hmac_key: Optional[str] = None

# =========================================================
# Content Analysis
# =========================================================

class ContentAnalyzer:
    """Analyzes text content to inform modulation decisions."""
    
    def __init__(self):
        self.stopwords = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", 
            "of", "with", "by", "is", "are", "was", "were", "be", "been", "being"
        }
    
    def analyze(self, text: str) -> Dict[str, float]:
        """Extract metrics from text content."""
        if not text:
            return {"entropy": 0.0, "complexity": 0.0, "redundancy": 1.0}
        
        # Shannon entropy
        entropy = self._calculate_entropy(text)
        
        # Lexical diversity (unique words / total words)
        words = [w.lower().strip(".,!?;:()[]\"'") for w in text.split()]
        word_counts = Counter(w for w in words if w and w not in self.stopwords)
        diversity = len(word_counts) / max(1, len(words)) if words else 0.0
        
        # Compression ratio as complexity measure
        compressed = zlib.compress(text.encode('utf-8'))
        compression_ratio = len(compressed) / max(1, len(text.encode('utf-8')))
        
        # Repetition analysis
        redundancy = self._calculate_redundancy(text)
        
        return {
            "entropy": entropy,
            "diversity": diversity,
            "compression_ratio": compression_ratio,
            "redundancy": redundancy,
            "complexity": (entropy + diversity) / 2.0,
            "length": len(text)
        }
    
    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of character distribution."""
        if not text:
            return 0.0
        
        counts = Counter(text)
        total = len(text)
        entropy = 0.0
        
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        
        return entropy
    
    def _calculate_redundancy(self, text: str) -> float:
        """Calculate text redundancy (higher = more repetitive)."""
        if len(text) < 10:
            return 0.0
        
        # Count repeated n-grams
        bigrams = [text[i:i+2] for i in range(len(text)-1)]
        trigrams = [text[i:i+3] for i in range(len(text)-2)]
        
        bigram_repeats = sum(1 for count in Counter(bigrams).values() if count > 1)
        trigram_repeats = sum(1 for count in Counter(trigrams).values() if count > 1)
        
        redundancy = (bigram_repeats + trigram_repeats) / max(1, len(bigrams) + len(trigrams))
        return min(1.0, redundancy * 2.0)  # Scale to 0-1 range

# =========================================================
# Adaptive Modulation Planner
# =========================================================

class AdaptiveModulationPlanner:
    """Selects optimal modulation scheme based on content analysis and link conditions."""
    
    def __init__(self, db_path: str = "modulation_history.json"):
        self.db_path = Path(db_path)
        self.history = self._load_history()
        self.analyzer = ContentAnalyzer()
    
    def select_modulation(self, text: str, snr_db: float = 20.0) -> Tuple[ModulationScheme, Dict[str, Any]]:
        """Select modulation scheme based on content and channel conditions."""
        analysis = self.analyzer.analyze(text)
        
        # Decision logic based on content characteristics
        scheme = self._decide_scheme(analysis, snr_db)
        
        explanation = {
            "selected_scheme": scheme.name,
            "content_analysis": analysis,
            "snr_db": snr_db,
            "decision_factors": self._get_decision_factors(analysis, snr_db, scheme)
        }
        
        return scheme, explanation
    
    def _decide_scheme(self, analysis: Dict[str, float], snr_db: float) -> ModulationScheme:
        """Core decision logic for modulation selection."""
        
        # Low SNR: use robust schemes
        if snr_db < 10:
            return ModulationScheme.BFSK
        
        # High complexity/entropy content: use higher-order modulation
        if analysis["complexity"] > 0.7 and snr_db > 20:
            if analysis["entropy"] > 4.0:
                return ModulationScheme.QAM16
            else:
                return ModulationScheme.QPSK
        
        # Medium complexity: QPSK or BPSK
        if analysis["complexity"] > 0.4:
            return ModulationScheme.QPSK if snr_db > 15 else ModulationScheme.BPSK
        
        # Low complexity or high redundancy: simple schemes work fine
        if analysis["redundancy"] > 0.6:
            return ModulationScheme.BFSK
        
        # Default fallback
        return ModulationScheme.QPSK if snr_db > 12 else ModulationScheme.BPSK
    
    def _get_decision_factors(self, analysis: Dict[str, float], snr_db: float, scheme: ModulationScheme) -> List[str]:
        """Explain the decision factors."""
        factors = []
        
        if snr_db < 10:
            factors.append(f"Low SNR ({snr_db:.1f}dB) requires robust modulation")
        elif snr_db > 20:
            factors.append(f"High SNR ({snr_db:.1f}dB) allows efficient modulation")
        
        if analysis["complexity"] > 0.7:
            factors.append(f"High content complexity ({analysis['complexity']:.2f}) benefits from higher-order modulation")
        elif analysis["complexity"] < 0.3:
            factors.append(f"Low content complexity ({analysis['complexity']:.2f}) suits simple modulation")
        
        if analysis["redundancy"] > 0.6:
            factors.append(f"High redundancy ({analysis['redundancy']:.2f}) allows robust encoding")
        
        if analysis["entropy"] > 4.0:
            factors.append(f"High entropy ({analysis['entropy']:.2f}) suggests information-dense content")
        
        return factors
    
    def record_performance(self, text: str, scheme: ModulationScheme, success: bool, 
                          snr_db: float, metrics: Dict[str, Any]) -> None:
        """Record modulation performance for future learning."""
        record = {
            "timestamp": datetime.now().isoformat(),
            "text_hash": hashlib.sha256(text.encode()).hexdigest()[:16],
            "scheme": scheme.name,
            "success": success,
            "snr_db": snr_db,
            "content_analysis": self.analyzer.analyze(text),
            "metrics": metrics
        }
        
        self.history.append(record)
        
        # Keep only last 1000 records
        if len(self.history) > 1000:
            self.history = self.history[-1000:]
        
        self._save_history()
    
    def _load_history(self) -> List[Dict[str, Any]]:
        """Load performance history from disk."""
        if not self.db_path.exists():
            return []
        
        try:
            with open(self.db_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            log.warning(f"Failed to load history from {self.db_path}: {e}")
            return []
    
    def _save_history(self) -> None:
        """Save performance history to disk."""
        try:
            with open(self.db_path, 'w') as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            log.warning(f"Failed to save history to {self.db_path}: {e}")

# =========================================================
# Signal Processing Utilities
# =========================================================

def calculate_signal_metrics(signal: np.ndarray, sample_rate: int) -> Dict[str, float]:
    """Calculate basic signal quality metrics."""
    if len(signal) == 0:
        return {"rms": 0.0, "peak": 0.0, "papr_db": 0.0, "bandwidth_hz": 0.0}
    
    rms = float(np.sqrt(np.mean(signal**2)))
    peak = float(np.max(np.abs(signal)))
    papr_db = 20 * math.log10(peak / (rms + 1e-12)) if rms > 1e-12 else 0.0
    
    # Estimate 99% power bandwidth
    spectrum = np.abs(rfft(signal))**2
    freqs = rfftfreq(len(signal), 1.0/sample_rate)
    
    # Find frequencies containing 99% of power
    cumulative_power = np.cumsum(spectrum) / np.sum(spectrum)
    idx_1pct = np.argmax(cumulative_power >= 0.005)
    idx_99pct = np.argmax(cumulative_power >= 0.995)
    bandwidth = freqs[idx_99pct] - freqs[idx_1pct]
    
    return {
        "rms": rms,
        "peak": peak,
        "papr_db": papr_db,
        "bandwidth_hz": float(bandwidth)
    }

def to_bits(data: bytes) -> List[int]:
    """Convert bytes to bit list (MSB first)."""
    return [(byte >> i) & 1 for byte in data for i in range(7, -1, -1)]

def from_bits(bits: Sequence[int]) -> bytes:
    """Convert bit list to bytes."""
    if len(bits) % 8 != 0:
        bits = list(bits) + [0] * (8 - len(bits) % 8)
    
    result = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for bit in bits[i:i+8]:
            byte = (byte << 1) | (1 if bit else 0)
        result.append(byte)
    return bytes(result)

def chunk_bits(bits: Sequence[int], n: int) -> List[List[int]]:
    """Split bits into chunks of size n."""
    return [list(bits[i:i+n]) for i in range(0, len(bits), n)]

# =========================================================
# Error Correction
# =========================================================

def hamming74_encode(data_bits: List[int]) -> List[int]:
    """Encode using Hamming(7,4) code."""
    if len(data_bits) % 4 != 0:
        data_bits = data_bits + [0] * (4 - len(data_bits) % 4)
    
    encoded = []
    for i in range(0, len(data_bits), 4):
        d0, d1, d2, d3 = data_bits[i:i+4]
        # Calculate parity bits
        p1 = d0 ^ d1 ^ d3
        p2 = d0 ^ d2 ^ d3
        p3 = d1 ^ d2 ^ d3
        # Output format: p1, p2, d0, p3, d1, d2, d3
        encoded.extend([p1, p2, d0, p3, d1, d2, d3])
    
    return encoded

def repetition_encode(data_bits: List[int], rep_factor: int = 3) -> List[int]:
    """Simple repetition code."""
    return [bit for bit in data_bits for _ in range(rep_factor)]

def apply_fec(bits: List[int], scheme: FECScheme) -> List[int]:
    """Apply forward error correction."""
    if scheme == FECScheme.NONE:
        return list(bits)
    elif scheme == FECScheme.HAMMING74:
        return hamming74_encode(bits)
    elif scheme == FECScheme.REPETITION:
        return repetition_encode(bits, 3)
    else:
        raise ValueError(f"Unsupported FEC scheme: {scheme}")

# =========================================================
# Security Functions
# =========================================================

def crc32_bytes(data: bytes) -> bytes:
    """Calculate CRC32 checksum."""
    return binascii.crc32(data).to_bytes(4, "big")

def add_watermark(data: bytes, watermark: str) -> bytes:
    """Add watermark to data."""
    wm_hash = hashlib.sha256(watermark.encode("utf-8")).digest()[:8]
    return wm_hash + data

def encrypt_data(data: bytes, password: str) -> bytes:
    """Encrypt data using AES-GCM."""
    if not HAS_CRYPTO:
        raise RuntimeError("pycryptodome required for encryption")
    
    salt = get_random_bytes(16)
    key = PBKDF2(password, salt, dkLen=32, count=100000)
    nonce = get_random_bytes(12)
    
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    ciphertext, tag = cipher.encrypt_and_digest(data)
    
    # Format: magic(4) + salt(16) + nonce(12) + tag(16) + ciphertext
    return b"AGCM" + salt + nonce + tag + ciphertext

def add_hmac(data: bytes, hmac_key: str) -> bytes:
    """Add HMAC-SHA256 to data."""
    import hmac
    key = hashlib.sha256(hmac_key.encode("utf-8")).digest()
    mac = hmac.new(key, data, hashlib.sha256).digest()
    return data + b"HMAC" + mac

def frame_data(payload: bytes, config: FrameConfig) -> bytes:
    """Frame payload with header and checksum."""
    timestamp = int(time.time()) & 0xFFFFFFFF
    header = struct.pack(">BBI", 0xA5, config.version, timestamp)
    
    core_data = header + payload
    
    # Add checksums
    if config.use_crc32:
        core_data += crc32_bytes(core_data)
    if config.use_crc16:
        # Simple CRC16-CCITT
        crc = 0xFFFF
        for byte in core_data:
            crc ^= byte << 8
            for _ in range(8):
                if crc & 0x8000:
                    crc = (crc << 1) ^ 0x1021
                else:
                    crc <<= 1
                crc &= 0xFFFF
        core_data += crc.to_bytes(2, "big")
    
    return config.preamble + core_data

def encode_text_payload(text: str, frame_config: FrameConfig, 
                       security_config: SecurityConfig, fec_scheme: FECScheme) -> List[int]:
    """Full encoding pipeline: text -> secured -> framed -> FEC -> bits."""
    data = text.encode("utf-8")
    
    # Apply security measures
    if security_config.watermark:
        data = add_watermark(data, security_config.watermark)
    
    if security_config.password:
        data = encrypt_data(data, security_config.password)
    
    # Frame the data
    framed_data = frame_data(data, frame_config)
    
    # Add HMAC if requested
    if security_config.hmac_key:
        framed_data = add_hmac(framed_data, security_config.hmac_key)
    
    # Convert to bits and apply FEC
    bits = to_bits(framed_data)
    return apply_fec(bits, fec_scheme)

# =========================================================
# Modulation Functions
# =========================================================

class DigitalModulator:
    """Digital modulation implementation."""
    
    @staticmethod
    def bfsk(bits: List[int], config: ModulationConfig) -> np.ndarray:
        """Binary Frequency Shift Keying."""
        samples_per_bit = int(config.sample_rate / config.symbol_rate)
        t = np.arange(samples_per_bit) / config.sample_rate
        
        signal_parts = []
        for bit in bits:
            freq = config.f1 if bit else config.f0
            signal_parts.append(config.amplitude * np.sin(2 * np.pi * freq * t))
        
        if not signal_parts:
            return np.array([], dtype=np.float32)
        
        result = np.concatenate(signal_parts)
        return np.clip(result, -1, 1).astype(np.float32) if config.clip else result.astype(np.float32)
    
    @staticmethod
    def bpsk(bits: List[int], config: ModulationConfig) -> Tuple[np.ndarray, np.ndarray]:
        """Binary Phase Shift Keying."""
        samples_per_bit = int(config.sample_rate / config.symbol_rate)
        t = np.arange(samples_per_bit) / config.sample_rate
        
        audio_parts = []
        iq_parts = []
        
        for bit in bits:
            phase = 0.0 if bit else np.pi
            
            # Audio signal (upconverted)
            audio_parts.append(config.amplitude * np.sin(2 * np.pi * config.fc * t + phase))
            
            # IQ baseband
            iq_symbol = config.amplitude * (np.cos(phase) + 1j * np.sin(phase))
            iq_parts.append(iq_symbol * np.ones(samples_per_bit, dtype=np.complex64))
        
        audio = np.concatenate(audio_parts) if audio_parts else np.array([], dtype=np.float32)
        iq = np.concatenate(iq_parts) if iq_parts else np.array([], dtype=np.complex64)
        
        if config.clip:
            audio = np.clip(audio, -1, 1)
        
        return audio.astype(np.float32), iq
    
    @staticmethod
    def qpsk(bits: List[int], config: ModulationConfig) -> Tuple[np.ndarray, np.ndarray]:
        """Quadrature Phase Shift Keying."""
        bit_pairs = chunk_bits(bits, 2)
        samples_per_symbol = int(config.sample_rate / config.symbol_rate)
        
        symbols = []
        for pair in bit_pairs:
            b0, b1 = (pair + [0, 0])[:2]  # Pad if needed
            
            # Gray mapping
            if (b0, b1) == (0, 0):
                symbol = 1 + 1j
            elif (b0, b1) == (0, 1):
                symbol = -1 + 1j
            elif (b0, b1) == (1, 1):
                symbol = -1 - 1j
            else:  # (1, 0)
                symbol = 1 - 1j
            
            symbols.append(symbol / math.sqrt(2))  # Normalize for unit energy
        
        return DigitalModulator._psk_to_signals(symbols, config, samples_per_symbol)
    
    @staticmethod
    def qam16(bits: List[int], config: ModulationConfig) -> Tuple[np.ndarray, np.ndarray]:
        """16-QAM modulation."""
        bit_quads = chunk_bits(bits, 4)
        samples_per_symbol = int(config.sample_rate / config.symbol_rate)
        
        def gray_map_2bit(b0: int, b1: int) -> int:
            """Map 2 bits to {-3, -1, 1, 3} using Gray coding."""
            if (b0, b1) == (0, 0): return -3
            elif (b0, b1) == (0, 1): return -1
            elif (b0, b1) == (1, 1): return 1
            else: return 3  # (1, 0)
        
        symbols = []
        for quad in bit_quads:
            b0, b1, b2, b3 = (quad + [0, 0, 0, 0])[:4]
            
            i_val = gray_map_2bit(b0, b1)
            q_val = gray_map_2bit(b2, b3)
            
            symbol = (i_val + 1j * q_val) / math.sqrt(10)  # Normalize
            symbols.append(symbol)
        
        return DigitalModulator._psk_to_signals(symbols, config, samples_per_symbol)
    
    @staticmethod
    def ofdm(bits: List[int], config: ModulationConfig) -> Tuple[np.ndarray, np.ndarray]:
        """Simple OFDM with QPSK subcarrier mapping."""
        n_subcarriers = config.ofdm_subcarriers
        cp_len = config.ofdm_cp_length
        
        # Group bits for OFDM symbols
        bits_per_ofdm_symbol = n_subcarriers * 2  # 2 bits per QPSK symbol
        bit_groups = chunk_bits(bits, bits_per_ofdm_symbol)
        
        audio_blocks = []
        iq_blocks = []
        
        for bit_group in bit_groups:
            # Map bits to QPSK symbols
            qpsk_symbols = []
            bit_pairs = chunk_bits(bit_group, 2)
            
            for pair in bit_pairs:
                b0, b1 = (pair + [0, 0])[:2]
                if (b0, b1) == (0, 0): symbol = 1 + 1j
                elif (b0, b1) == (0, 1): symbol = -1 + 1j
                elif (b0, b1) == (1, 1): symbol = -1 - 1j
                else: symbol = 1 - 1j
                qpsk_symbols.append(symbol / math.sqrt(2))
            
            # Pad to subcarrier count
            while len(qpsk_symbols) < n_subcarriers:
                qpsk_symbols.append(0 + 0j)
            
            # IFFT to get time domain
            freq_domain = np.array(qpsk_symbols[:n_subcarriers], dtype=np.complex64)
            time_domain = ifft(freq_domain)
            
            # Add cyclic prefix
            cp = time_domain[-cp_len:]
            ofdm_symbol = np.concatenate([cp, time_domain])
            
            # Scale amplitude and store IQ
            ofdm_symbol *= config.amplitude
            iq_blocks.append(ofdm_symbol.astype(np.complex64))
            
            # Convert to audio (real part upconverted)
            t = np.arange(len(ofdm_symbol)) / config.sample_rate
            audio_block = ofdm_symbol.real * np.cos(2 * np.pi * config.fc * t) - \
                         ofdm_symbol.imag * np.sin(2 * np.pi * config.fc * t)
            audio_blocks.append(audio_block.astype(np.float32))
        
        audio = np.concatenate(audio_blocks) if audio_blocks else np.array([], dtype=np.float32)
        iq = np.concatenate(iq_blocks) if iq_blocks else np.array([], dtype=np.complex64)
        
        if config.clip:
            audio = np.clip(audio, -1, 1)
        
        return audio, iq
    
    @staticmethod
    def _psk_to_signals(symbols: List[complex], config: ModulationConfig, 
                       samples_per_symbol: int) -> Tuple[np.ndarray, np.ndarray]:
        """Convert PSK/QAM symbols to audio and IQ signals."""
        if not symbols:
            return np.array([], dtype=np.float32), np.array([], dtype=np.complex64)
        
        # Upsample symbols (rectangular pulse shaping for simplicity)
        i_data = np.repeat([s.real for s in symbols], samples_per_symbol) * config.amplitude
        q_data = np.repeat([s.imag for s in symbols], samples_per_symbol) * config.amplitude
        
        # Generate time vector
        t = np.arange(len(i_data)) / config.sample_rate
        
        # Create audio signal (upconverted)
        audio = i_data * np.cos(2 * np.pi * config.fc * t) - \
                q_data * np.sin(2 * np.pi * config.fc * t)
        
        # Create IQ baseband
        iq = i_data + 1j * q_data
        
        if config.clip:
            audio = np.clip(audio, -1, 1)
        
        return audio.astype(np.float32), iq.astype(np.complex64)

def modulate_bits(bits: List[int], scheme: ModulationScheme, 
                 config: ModulationConfig) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Route bits to appropriate modulation function."""
    if scheme == ModulationScheme.BFSK:
        return DigitalModulator.bfsk(bits, config), None
    elif scheme == ModulationScheme.BPSK:
        return DigitalModulator.bpsk(bits, config)
    elif scheme == ModulationScheme.QPSK:
        return DigitalModulator.qpsk(bits, config)
    elif scheme == ModulationScheme.QAM16:
        return DigitalModulator.qam16(bits, config)
    elif scheme == ModulationScheme.OFDM:
        return DigitalModulator.ofdm(bits, config)
    else:
        raise ValueError(f"Unsupported modulation scheme: {scheme}")

# =========================================================
# LLM Integration
# =========================================================

class LLMClient:
    """Simplified LLM client supporting multiple backends."""
    
    def __init__(self, config: HTTPConfig):
        if not HAS_REQUESTS:
            raise RuntimeError("requests library required for LLM functionality")
        self.config = config
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using configured LLM."""
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 512)
        
        if self.config.mode == "openai-chat":
            return self._openai_chat(prompt, temperature, max_tokens)
        elif self.config.mode == "llama-cpp":
            return self._llama_cpp(prompt, temperature, max_tokens)
        else:
            raise ValueError(f"Unsupported LLM mode: {self.config.mode}")
    
    def _openai_chat(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """OpenAI chat completions API."""
        url = f"{self.config.base_url.rstrip('/')}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        
        payload = {
            "model": self.config.model or "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        response = requests.post(url, headers=headers, json=payload, 
                               timeout=self.config.timeout, verify=self.config.verify_ssl)
        response.raise_for_status()
        
        data = response.json()
        return data["choices"][0]["message"]["content"]
    
    def _llama_cpp(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """Llama.cpp server API."""
        url = f"{self.config.base_url.rstrip('/')}/completion"
        payload = {
            "prompt": prompt,
            "temperature": temperature,
            "n_predict": max_tokens
        }
        
        response = requests.post(url, json=payload, timeout=self.config.timeout)
        response.raise_for_status()
        
        data = response.json()
        return data.get("content", "")

class DualLLMOrchestrator:
    """Coordinates local and remote LLMs for text generation."""
    
    def __init__(self, local_config: HTTPConfig, remote_config: Optional[HTTPConfig] = None):
        self.local_llm = LLMClient(local_config)
        self.remote_llm = LLMClient(remote_config) if remote_config else None
        self.max_context_chars = 8000
    
    def generate_text(self, user_prompt: str, resource_files: List[str] = None, 
                     resource_texts: List[str] = None) -> Dict[str, str]:
        """Generate text using dual LLM approach."""
        
        # Load and process resource materials
        resources = self._load_resources(resource_files or [], resource_texts or [])
        
        # Use remote LLM for resource summarization if available
        if self.remote_llm and resources:
            summary_prompt = f"""Summarize and structure the following content concisely:

{resources}

Focus on key information and maintain factual accuracy."""
            
            resource_summary = self.remote_llm.generate(summary_prompt, temperature=0.3)
        else:
            resource_summary = self._local_summarize(resources)
        
        # Create final prompt for local LLM
        final_prompt = self._build_final_prompt(user_prompt, resource_summary)
        
        # Generate final response with local LLM
        final_response = self.local_llm.generate(final_prompt, temperature=0.7)
        
        return {
            "final_response": final_response,
            "resource_summary": resource_summary,
            "prompt_used": final_prompt
        }
    
    def _load_resources(self, file_paths: List[str], texts: List[str]) -> str:
        """Load resource files and combine with provided texts."""
        content_parts = []
        
        # Load files
        for path_str in file_paths:
            path = Path(path_str)
            if path.exists() and path.is_file():
                try:
                    content = path.read_text(encoding="utf-8", errors="ignore")
                    content_parts.append(f"=== {path.name} ===\n{content}")
                except Exception as e:
                    content_parts.append(f"=== {path.name} ===\n[ERROR: Could not read file - {e}]")
            else:
                content_parts.append(f"=== {path_str} ===\n[ERROR: File not found]")
        
        # Add inline texts
        for i, text in enumerate(texts):
            content_parts.append(f"=== Resource {i+1} ===\n{text}")
        
        combined = "\n\n".join(content_parts)
        return combined[:self.max_context_chars]
    
    def _local_summarize(self, text: str) -> str:
        """Simple local summarization fallback."""
        if not text:
            return "No resources provided."
        
        # Simple extractive summarization
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if len(sentences) <= 3:
            return text[:500]
        
        # Score sentences by length and word frequency
        words = text.lower().split()
        word_freq = Counter(words)
        
        scored_sentences = []
        for sentence in sentences:
            score = len(sentence) * 0.1
            sent_words = sentence.lower().split()
            score += sum(word_freq.get(word, 0) for word in sent_words)
            scored_sentences.append((sentence, score))
        
        # Take top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [s[0] for s in scored_sentences[:min(5, len(scored_sentences))]]
        
        summary = " ".join(top_sentences)
        return summary[:800]
    
    def _build_final_prompt(self, user_prompt: str, resource_summary: str) -> str:
        """Build the final prompt for local LLM."""
        if resource_summary and resource_summary != "No resources provided.":
            return f"""Context from resources:
{resource_summary}

User request: {user_prompt}

Please provide a clear, direct response based on the context above."""
        else:
            return user_prompt

# =========================================================
# File I/O Functions
# =========================================================

def write_wav_file(path: Path, signal: np.ndarray, sample_rate: int) -> None:
    """Write audio signal to WAV file."""
    import wave
    
    # Ensure signal is in valid range
    signal = np.clip(signal, -1.0, 1.0)
    
    # Convert to 16-bit PCM
    pcm_data = (signal * 32767).astype(np.int16)
    
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data.tobytes())

def write_iq_file(path: Path, iq_data: np.ndarray) -> None:
    """Write complex IQ data as interleaved float32."""
    if iq_data.ndim != 1 or not np.iscomplexobj(iq_data):
        raise ValueError("IQ data must be 1D complex array")
    
    # Interleave I and Q
    interleaved = np.empty(iq_data.size * 2, dtype=np.float32)
    interleaved[0::2] = iq_data.real.astype(np.float32)
    interleaved[1::2] = iq_data.imag.astype(np.float32)
    
    path.write_bytes(interleaved.tobytes())

def plot_signal_analysis(audio: np.ndarray, iq: Optional[np.ndarray], 
                        sample_rate: int, title: str, output_path: Path) -> None:
    """Create signal analysis plots."""
    if not HAS_MPL:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Time domain plot (first 50ms)
    t_max_samples = min(len(audio), int(0.05 * sample_rate))
    t = np.arange(t_max_samples) / sample_rate
    axes[0, 0].plot(t, audio[:t_max_samples])
    axes[0, 0].set_title("Time Domain (first 50ms)")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].grid(True)
    
    # Frequency domain
    spectrum = np.abs(rfft(audio))
    freqs = rfftfreq(len(audio), 1.0/sample_rate)
    axes[0, 1].semilogy(freqs, spectrum / np.max(spectrum))
    axes[0, 1].set_xlim(0, min(8000, sample_rate//2))
    axes[0, 1].set_title("Frequency Spectrum")
    axes[0, 1].set_xlabel("Frequency (Hz)")
    axes[0, 1].set_ylabel("Normalized Magnitude")
    axes[0, 1].grid(True)
    
    # IQ constellation (if available)
    if iq is not None and len(iq) > 0:
        # Downsample for plotting
        step = max(1, len(iq) // 1000)
        iq_plot = iq[::step]
        axes[1, 0].scatter(iq_plot.real, iq_plot.imag, s=2, alpha=0.6)
        axes[1, 0].set_title("IQ Constellation")
        axes[1, 0].set_xlabel("I")
        axes[1, 0].set_ylabel("Q")
        axes[1, 0].grid(True)
        axes[1, 0].set_aspect('equal')
    else:
        axes[1, 0].text(0.5, 0.5, "No IQ data", ha='center', va='center')
        axes[1, 0].set_title("IQ Constellation")
    
    # Signal metrics
    metrics = calculate_signal_metrics(audio, sample_rate)
    metrics_text = f"""RMS: {metrics['rms']:.3f}
Peak: {metrics['peak']:.3f}
PAPR: {metrics['papr_db']:.1f} dB
BW: {metrics['bandwidth_hz']:.0f} Hz"""
    
    axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=10, verticalalignment='center')
    axes[1, 1].set_title("Signal Metrics")
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

# =========================================================
# Integrated WaveCaster
# =========================================================

@dataclass
class OutputFiles:
    wav_path: Optional[Path] = None
    iq_path: Optional[Path] = None
    plot_path: Optional[Path] = None
    metadata_path: Optional[Path] = None

class IntegratedWaveCaster:
    """Main integrated system combining LLM orchestration with adaptive modulation."""
    
    def __init__(self, local_config: HTTPConfig, remote_config: Optional[HTTPConfig] = None):
        self.orchestrator = DualLLMOrchestrator(local_config, remote_config)
        self.planner = AdaptiveModulationPlanner()
        self.analyzer = ContentAnalyzer()
    
    def cast_text_to_waveform(self, text: str, output_dir: Path, 
                             frame_config: FrameConfig,
                             security_config: SecurityConfig,
                             fec_scheme: FECScheme = FECScheme.HAMMING74,
                             force_scheme: Optional[ModulationScheme] = None,
                             snr_hint: float = 20.0,
                             want_wav: bool = True,
                             want_iq: bool = True,
                             want_plot: bool = True) -> OutputFiles:
        """Convert text to modulated waveform with adaptive scheme selection."""
        
        # Select modulation scheme
        if force_scheme:
            scheme = force_scheme
            explanation = {"forced_scheme": scheme.name}
        else:
            scheme, explanation = self.planner.select_modulation(text, snr_hint)
        
        log.info(f"Selected modulation: {scheme.name}")
        
        # Create modulation config
        mod_config = ModulationConfig()
        
        # Encode text to bits
        bits = encode_text_payload(text, frame_config, security_config, fec_scheme)
        
        # Modulate
        audio, iq = modulate_bits(bits, scheme, mod_config)
        
        # Prepare output files
        timestamp = int(time.time())
        base_name = f"cast_{scheme.name.lower()}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        files = OutputFiles()
        
        # Write WAV file
        if want_wav and audio is not None and len(audio) > 0:
            files.wav_path = output_dir / f"{base_name}.wav"
            write_wav_file(files.wav_path, audio, mod_config.sample_rate)
        
        # Write IQ file
        if want_iq:
            if iq is None and audio is not None:
                # Generate IQ from audio using Hilbert transform
                try:
                    analytic = sp_signal.hilbert(audio)
                    iq = analytic.astype(np.complex64)
                except Exception:
                    iq = audio.astype(np.complex64)  # Real-only fallback
            
            if iq is not None and len(iq) > 0:
                files.iq_path = output_dir / f"{base_name}.iq"
                write_iq_file(files.iq_path, iq)
        
        # Create visualization
        if want_plot and audio is not None and len(audio) > 0:
            files.plot_path = output_dir / f"{base_name}.png"
            plot_signal_analysis(audio, iq, mod_config.sample_rate, 
                                f"{scheme.name} Modulation", files.plot_path)
        
        # Write metadata
        metadata = {
            "timestamp": timestamp,
            "scheme": scheme.name,
            "config": {
                "sample_rate": mod_config.sample_rate,
                "symbol_rate": mod_config.symbol_rate,
                "amplitude": mod_config.amplitude
            },
            "security": {
                "encrypted": bool(security_config.password),
                "watermarked": bool(security_config.watermark),
                "hmac": bool(security_config.hmac_key)
            },
            "fec": fec_scheme.name,
            "selection_explanation": explanation
        }
        
        if audio is not None:
            metadata["signal_metrics"] = calculate_signal_metrics(audio, mod_config.sample_rate)
            metadata["duration_seconds"] = len(audio) / mod_config.sample_rate
        
        files.metadata_path = output_dir / f"{base_name}.json"
        with open(files.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return files
    
    def llm_cast_to_waveform(self, prompt: str, output_dir: Path,
                            resource_files: List[str] = None,
                            resource_texts: List[str] = None,
                            **kwargs) -> Tuple[OutputFiles, Dict[str, str]]:
        """Generate text via LLM orchestration, then modulate to waveform."""
        
        # Generate text
        llm_result = self.orchestrator.generate_text(prompt, resource_files, resource_texts)
        generated_text = llm_result["final_response"]
        
        # Convert to waveform
        files = self.cast_text_to_waveform(generated_text, output_dir, **kwargs)
        
        return files, llm_result

# =========================================================
# Command Line Interface
# =========================================================

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="integrated_wavecaster",
        description="Integrated LLM orchestration and adaptive modulation system"
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Common arguments
    def add_modulation_args(subparser):
        subparser.add_argument("--output-dir", type=str, default="output", 
                              help="Output directory")
        subparser.add_argument("--scheme", choices=[s.name.lower() for s in ModulationScheme],
                              help="Force specific modulation scheme")
        subparser.add_argument("--snr", type=float, default=20.0,
                              help="Expected SNR in dB for scheme selection")
        subparser.add_argument("--no-wav", action="store_true", help="Skip WAV output")
        subparser.add_argument("--no-iq", action="store_true", help="Skip IQ output") 
        subparser.add_argument("--no-plot", action="store_true", help="Skip plots")
        subparser.add_argument("--play", action="store_true", help="Play audio after generation")
        
        # Security options
        subparser.add_argument("--password", type=str, help="Encryption password")
        subparser.add_argument("--watermark", type=str, help="Watermark string")
        subparser.add_argument("--hmac-key", type=str, help="HMAC key")
        subparser.add_argument("--fec", choices=[f.name.lower() for f in FECScheme],
                              default="hamming74", help="Forward error correction")
    
    def add_llm_args(subparser):
        subparser.add_argument("--local-url", type=str, default="http://localhost:8080",
                              help="Local LLM URL")
        subparser.add_argument("--local-mode", choices=["openai-chat", "llama-cpp"],
                              default="llama-cpp", help="Local LLM API mode")
        subparser.add_argument("--local-model", type=str, help="Local LLM model name")
        subparser.add_argument("--local-key", type=str, help="Local LLM API key")
        
        subparser.add_argument("--remote-url", type=str, help="Remote LLM URL")
        subparser.add_argument("--remote-mode", choices=["openai-chat", "llama-cpp"],
                              default="openai-chat", help="Remote LLM API mode")
        subparser.add_argument("--remote-model", type=str, help="Remote LLM model name")
        subparser.add_argument("--remote-key", type=str, help="Remote LLM API key")
    
    # Direct modulation command
    mod_parser = subparsers.add_parser("modulate", help="Modulate text directly to waveform")
    mod_parser.add_argument("--text", type=str, required=True, help="Text to modulate")
    add_modulation_args(mod_parser)
    
    # LLM generation + modulation
    cast_parser = subparsers.add_parser("cast", help="Generate text via LLM then modulate")
    cast_parser.add_argument("--prompt", type=str, required=True, help="Generation prompt")
    cast_parser.add_argument("--resource-file", action="append", default=[],
                            help="Resource files to include")
    cast_parser.add_argument("--resource-text", action="append", default=[],
                            help="Inline resource text")
    add_llm_args(cast_parser)
    add_modulation_args(cast_parser)
    
    # Analysis command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze text content")
    analyze_parser.add_argument("--text", type=str, help="Text to analyze")
    analyze_parser.add_argument("--file", type=str, help="File to analyze")
    
    # Visualization command  
    viz_parser = subparsers.add_parser("visualize", help="Visualize existing WAV file")
    viz_parser.add_argument("--wav-file", type=str, required=True, help="WAV file to visualize")
    viz_parser.add_argument("--output", type=str, help="Output plot file")
    
    return parser

def load_audio_file(path: str) -> Tuple[np.ndarray, int]:
    """Load audio from WAV file."""
    import wave
    
    with wave.open(path, "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        audio_data = wav_file.readframes(n_frames)
        
        # Convert to float
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32767.0
        
        return audio_array, sample_rate

def safe_json_dump(obj: Any) -> str:
    """JSON serialization with numpy support."""
    def default_encoder(x):
        if isinstance(x, (np.floating,)):
            return float(x)
        elif isinstance(x, (np.integer,)):
            return int(x)
        elif isinstance(x, (np.ndarray,)):
            return x.tolist()
        elif isinstance(x, complex):
            return {"real": float(x.real), "imag": float(x.imag)}
        elif isinstance(x, datetime):
            return x.isoformat()
        return str(x)
    
    return json.dumps(obj, indent=2, default=default_encoder, ensure_ascii=False)

def play_audio_file(audio: np.ndarray, sample_rate: int) -> None:
    """Play audio using sounddevice."""
    if not HAS_AUDIO:
        log.warning("sounddevice not available - cannot play audio")
        return
    
    try:
        sd.play(audio, sample_rate)
        sd.wait()
    except Exception as e:
        log.error(f"Audio playback failed: {e}")

# =========================================================
# Command Implementations
# =========================================================

def cmd_modulate(args: argparse.Namespace) -> int:
    """Direct text modulation command."""
    
    # Create configurations
    frame_config = FrameConfig()
    security_config = SecurityConfig(
        password=args.password,
        watermark=args.watermark,
        hmac_key=args.hmac_key
    )
    fec_scheme = FECScheme[args.fec.upper()]
    force_scheme = ModulationScheme[args.scheme.upper()] if args.scheme else None
    
    # Create wavecaster (no LLM needed for direct modulation)
    wavecaster = IntegratedWaveCaster(
        local_config=HTTPConfig(base_url="http://localhost:8080"),  # Dummy
        remote_config=None
    )
    
    # Generate waveform
    output_files = wavecaster.cast_text_to_waveform(
        text=args.text,
        output_dir=Path(args.output_dir),
        frame_config=frame_config,
        security_config=security_config,
        fec_scheme=fec_scheme,
        force_scheme=force_scheme,
        snr_hint=args.snr,
        want_wav=not args.no_wav,
        want_iq=not args.no_iq,
        want_plot=not args.no_plot
    )
    
    # Play audio if requested
    if args.play and output_files.wav_path:
        try:
            audio, sr = load_audio_file(str(output_files.wav_path))
            play_audio_file(audio, sr)
        except Exception as e:
            log.error(f"Audio playback failed: {e}")
    
    # Print results
    result = {
        "files": {
            "wav": str(output_files.wav_path) if output_files.wav_path else None,
            "iq": str(output_files.iq_path) if output_files.iq_path else None,
            "plot": str(output_files.plot_path) if output_files.plot_path else None,
            "metadata": str(output_files.metadata_path) if output_files.metadata_path else None
        }
    }
    
    print(safe_json_dump(result))
    return 0

def cmd_cast(args: argparse.Namespace) -> int:
    """LLM generation + modulation command."""
    
    if not HAS_REQUESTS:
        log.error("requests library required for LLM functionality")
        return 1
    
    # Create LLM configurations
    local_config = HTTPConfig(
        base_url=args.local_url,
        mode=args.local_mode,
        model=args.local_model,
        api_key=args.local_key
    )
    
    remote_config = None
    if args.remote_url:
        remote_config = HTTPConfig(
            base_url=args.remote_url,
            mode=args.remote_mode,
            model=args.remote_model,
            api_key=args.remote_key
        )
    
    # Create other configs
    frame_config = FrameConfig()
    security_config = SecurityConfig(
        password=args.password,
        watermark=args.watermark,
        hmac_key=args.hmac_key
    )
    fec_scheme = FECScheme[args.fec.upper()]
    force_scheme = ModulationScheme[args.scheme.upper()] if args.scheme else None
    
    # Create wavecaster
    wavecaster = IntegratedWaveCaster(local_config, remote_config)
    
    # Generate and modulate
    try:
        output_files, llm_result = wavecaster.llm_cast_to_waveform(
            prompt=args.prompt,
            output_dir=Path(args.output_dir),
            resource_files=args.resource_file,
            resource_texts=args.resource_text,
            frame_config=frame_config,
            security_config=security_config,
            fec_scheme=fec_scheme,
            force_scheme=force_scheme,
            snr_hint=args.snr,
            want_wav=not args.no_wav,
            want_iq=not args.no_iq,
            want_plot=not args.no_plot
        )
        
        # Play audio if requested
        if args.play and output_files.wav_path:
            try:
                audio, sr = load_audio_file(str(output_files.wav_path))
                play_audio_file(audio, sr)
            except Exception as e:
                log.error(f"Audio playback failed: {e}")
        
        # Print results
        result = {
            "generated_text": llm_result["final_response"][:200] + "..." if len(llm_result["final_response"]) > 200 else llm_result["final_response"],
            "resource_summary": llm_result["resource_summary"][:200] + "..." if len(llm_result["resource_summary"]) > 200 else llm_result["resource_summary"],
            "files": {
                "wav": str(output_files.wav_path) if output_files.wav_path else None,
                "iq": str(output_files.iq_path) if output_files.iq_path else None,
                "plot": str(output_files.plot_path) if output_files.plot_path else None,
                "metadata": str(output_files.metadata_path) if output_files.metadata_path else None
            }
        }
        
        print(safe_json_dump(result))
        return 0
        
    except Exception as e:
        log.error(f"Cast operation failed: {e}")
        return 1

def cmd_analyze(args: argparse.Namespace) -> int:
    """Content analysis command."""
    
    if args.text:
        text = args.text
    elif args.file:
        try:
            text = Path(args.file).read_text(encoding="utf-8")
        except Exception as e:
            log.error(f"Failed to read file {args.file}: {e}")
            return 1
    else:
        log.error("Must provide either --text or --file")
        return 1
    
    analyzer = ContentAnalyzer()
    analysis = analyzer.analyze(text)
    
    planner = AdaptiveModulationPlanner()
    recommended_scheme, explanation = planner.select_modulation(text)
    
    result = {
        "content_analysis": analysis,
        "recommended_modulation": {
            "scheme": recommended_scheme.name,
            "explanation": explanation
        }
    }
    
    print(safe_json_dump(result))
    return 0

def cmd_visualize(args: argparse.Namespace) -> int:
    """Visualization command."""
    
    if not HAS_MPL:
        log.error("matplotlib required for visualization")
        return 1
    
    try:
        audio, sample_rate = load_audio_file(args.wav_file)
        
        output_path = Path(args.output) if args.output else Path(args.wav_file).with_suffix('.png')
        
        plot_signal_analysis(audio, None, sample_rate, f"Analysis: {Path(args.wav_file).name}", output_path)
        
        result = {
            "input_file": args.wav_file,
            "output_plot": str(output_path),
            "signal_metrics": calculate_signal_metrics(audio, sample_rate)
        }
        
        print(safe_json_dump(result))
        return 0
        
    except Exception as e:
        log.error(f"Visualization failed: {e}")
        return 1

def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)
    
    if args.command == "modulate":
        return cmd_modulate(args)
    elif args.command == "cast":
        return cmd_cast(args)
    elif args.command == "analyze":
        return cmd_analyze(args)
    elif args.command == "visualize":
        return cmd_visualize(args)
    else:
        parser.print_help()
        return 1

if __name__ == "__main__":
    sys.exit(main())

# =========================================================
# Example Usage and Testing
# =========================================================

def run_example():
    """Example usage of the integrated system."""
    
    # Example 1: Direct text modulation
    print("=== Example 1: Direct Modulation ===")
    
    wavecaster = IntegratedWaveCaster(
        local_config=HTTPConfig(base_url="http://localhost:8080"),
        remote_config=None
    )
    
    test_text = "Hello world! This is a test transmission using adaptive modulation."
    
    files = wavecaster.cast_text_to_waveform(
        text=test_text,
        output_dir=Path("example_output"),
        frame_config=FrameConfig(),
        security_config=SecurityConfig(),
        want_wav=True,
        want_iq=True,
        want_plot=True
    )
    
    print(f"Generated files: {files}")
    
    # Example 2: Content analysis
    print("\n=== Example 2: Content Analysis ===")
    
    analyzer = ContentAnalyzer()
    analysis = analyzer.analyze(test_text)
    print(f"Content analysis: {safe_json_dump(analysis)}")
    
    planner = AdaptiveModulationPlanner()
    scheme, explanation = planner.select_modulation(test_text, snr_db=15.0)
    print(f"Recommended modulation: {scheme.name}")
    print(f"Explanation: {safe_json_dump(explanation)}")

# Additional utility functions for testing and validation

def validate_modulation_output(audio: np.ndarray, iq: Optional[np.ndarray], 
                              scheme: ModulationScheme, config: ModulationConfig) -> Dict[str, Any]:
    """Validate modulation output quality."""
    
    results = {"scheme": scheme.name, "valid": True, "issues": []}
    
    # Check audio signal
    if len(audio) == 0:
        results["valid"] = False
        results["issues"].append("Empty audio signal")
        return results
    
    metrics = calculate_signal_metrics(audio, config.sample_rate)
    results["metrics"] = metrics
    
    # Check for obvious issues
    if metrics["peak"] > 1.0:
        results["issues"].append(f"Audio clipped: peak = {metrics['peak']:.3f}")
    
    if metrics["rms"] < 0.01:
        results["issues"].append(f"Very low signal level: RMS = {metrics['rms']:.3f}")
    
    if metrics["papr_db"] > 15.0:
        results["issues"].append(f"High PAPR: {metrics['papr_db']:.1f} dB")
    
    # Scheme-specific validation
    if scheme in [ModulationScheme.BFSK]:
        # For FSK, check if we have energy near expected frequencies
        spectrum = np.abs(rfft(audio))**2
        freqs = rfftfreq(len(audio), 1.0/config.sample_rate)
        
        # Find peaks near f0 and f1
        f0_idx = np.argmin(np.abs(freqs - config.f0))
        f1_idx = np.argmin(np.abs(freqs - config.f1))
        
        f0_power = spectrum[max(0, f0_idx-2):f0_idx+3].sum()
        f1_power = spectrum[max(0, f1_idx-2):f1_idx+3].sum()
        total_power = spectrum.sum()
        
        fsk_power_ratio = (f0_power + f1_power) / total_power
        if fsk_power_ratio < 0.3:
            results["issues"].append(f"Low FSK tone power ratio: {fsk_power_ratio:.2f}")
    
    # IQ validation for complex schemes
    if iq is not None and scheme in [ModulationScheme.QPSK, ModulationScheme.QAM16]:
        # Check constellation properties
        if len(iq) > 0:
            # Downsample to symbol rate for constellation analysis
            symbol_period = int(config.sample_rate / config.symbol_rate)
            constellation_samples = iq[symbol_period//2::symbol_period]
            
            if len(constellation_samples) > 4:
                # Check if constellation points cluster appropriately
                magnitude_std = np.std(np.abs(constellation_samples))
                if magnitude_std > 0.5:
                    results["issues"].append(f"High constellation magnitude variation: {magnitude_std:.3f}")
    
    results["valid"] = len(results["issues"]) == 0
    return results

def benchmark_modulation_schemes(text: str, snr_values: List[float] = None) -> Dict[str, Any]:
    """Benchmark all modulation schemes for given text and SNR conditions."""
    
    if snr_values is None:
        snr_values = [5, 10, 15, 20, 25]
    
    analyzer = ContentAnalyzer()
    content_analysis = analyzer.analyze(text)
    
    results = {
        "text_analysis": content_analysis,
        "scheme_performance": {}
    }
    
    config = ModulationConfig()
    frame_config = FrameConfig()
    security_config = SecurityConfig()
    
    for scheme in ModulationScheme:
        scheme_results = {}
        
        try:
            # Generate bits
            bits = encode_text_payload(text, frame_config, security_config, FECScheme.HAMMING74)
            
            # Modulate
            audio, iq = modulate_bits(bits, scheme, config)
            
            if audio is not None and len(audio) > 0:
                # Calculate metrics
                metrics = calculate_signal_metrics(audio, config.sample_rate)
                validation = validate_modulation_output(audio, iq, scheme, config)
                
                # Estimate spectral efficiency
                data_bits = len(to_bits(text.encode("utf-8")))
                symbol_duration = len(audio) / config.sample_rate
                spectral_efficiency = data_bits / (symbol_duration * metrics["bandwidth_hz"]) if metrics["bandwidth_hz"] > 0 else 0
                
                scheme_results = {
                    "success": True,
                    "signal_metrics": metrics,
                    "validation": validation,
                    "spectral_efficiency_bps_hz": spectral_efficiency,
                    "duration_seconds": symbol_duration
                }
            else:
                scheme_results = {"success": False, "error": "No audio generated"}
                
        except Exception as e:
            scheme_results = {"success": False, "error": str(e)}
        
        results["scheme_performance"][scheme.name] = scheme_results
    
    return results

# Test functions
def test_basic_functionality():
    """Basic functionality test."""
    print("Testing basic modulation functionality...")
    
    test_text = "Test message 123"
    config = ModulationConfig()
    
    for scheme in [ModulationScheme.BFSK, ModulationScheme.BPSK, ModulationScheme.QPSK]:
        try:
            bits = to_bits(test_text.encode("utf-8"))
            audio, iq = modulate_bits(bits, scheme, config)
            
            if audio is not None and len(audio) > 0:
                metrics = calculate_signal_metrics(audio, config.sample_rate)
                print(f"  ✓ {scheme.name}: {len(audio)} samples, RMS={metrics['rms']:.3f}")
            else:
                print(f"  ✗ {scheme.name}: Failed to generate audio")
                
        except Exception as e:
            print(f"  ✗ {scheme.name}: Exception - {e}")
    
    print("Basic test completed.")

if __name__ == "__main__":
    # Uncomment to run tests
    # test_basic_functionality()
    # run_example()
    
    sys.exit(main())

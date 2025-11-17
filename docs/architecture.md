# System Architecture

## Overview

The CCL + WaveCaster system is a comprehensive platform combining categorical coherence analysis with signal modulation capabilities.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CCL + Julia Bridge System                 │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐      ┌──────────────┐      ┌───────────┐ │
│  │  CCL Tools   │─────▶│ Julia Bridge │─────▶│  AL-ULS   │ │
│  │  (Python)    │      │ (LIMPSBridge)│      │  Server   │ │
│  └──────────────┘      └──────────────┘      └───────────┘ │
│         │                      │                     │       │
│         │                      │                     │       │
│         ▼                      ▼                     ▼       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │            PyTorch TA-ULS Trainer                     │  │
│  │         (Stability-aware feature learning)            │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                               │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    WaveCaster Engine                         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐      ┌──────────────┐      ┌───────────┐ │
│  │ Content      │─────▶│  Adaptive    │─────▶│ Digital   │ │
│  │ Analyzer     │      │  Modulation  │      │ Modulator │ │
│  └──────────────┘      │  Planner     │      └───────────┘ │
│         │              └──────────────┘             │       │
│         │                      │                     │       │
│         ▼                      ▼                     ▼       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Dual LLM Orchestrator                    │  │
│  │         (Local Ollama + Remote API)                   │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. CCL Analysis Engine (ccl.py)

**Purpose:** Entropy-driven ghost detection in Python code

**Key Features:**
- Function behavior probing with random inputs
- Side-effect detection through AST analysis
- Idempotency, commutativity, and associativity testing
- Entropy calculation for function outputs
- Hotspot identification based on ghost scores

**Workflow:**
1. Load Python modules from target path
2. Discover all public functions
3. Probe each function with random inputs
4. Analyze source code for impurity
5. Calculate entropy and sensitivity metrics
6. Generate ghost score and hotspot report

### 2. Julia Bridge (LIMPSBridge.jl)

**Purpose:** High-performance coherence analysis and optimization routing

**Key Features:**
- HTTP server for coherence and optimization endpoints
- Matrix conversion with safety guards
- KFP (Kinetic Force Projection) smoothing
- Integration with AL-ULS backend

**Endpoints:**
- `GET /health` - Health check
- `POST /coherence` - Compute ghost score from CCL results
- `POST /optimize` - Matrix optimization via AL-ULS

**Workflow:**
1. Receive CCL analysis results
2. Parse and validate function probes
3. Calculate aggregate coherence metrics
4. Apply KFP smoothing for stability
5. Compute ghost score using sigmoid function
6. Return hotspots and metrics

### 3. Mock AL-ULS Server (mock_al_uls_server.py)

**Purpose:** FastAPI server simulating AL-ULS optimization backend

**Key Features:**
- Matrix optimization algorithms (sparsity, SVD low-rank)
- Polynomial feature creation
- Function dispatcher for Julia operations
- Health check endpoint

**Endpoints:**
- `GET /health` - Service health
- `POST /optimize` - Optimize adjacency matrix
- `POST /` - Generic function dispatcher

**Workflow:**
1. Receive optimization request
2. Parse adjacency matrix
3. Apply optimization algorithm
4. Return optimized routing and metrics

### 4. WaveCaster Engine (wavecaster.py)

**Purpose:** Adaptive signal modulation with LLM integration

**Key Components:**

#### ContentAnalyzer
- Calculates Shannon entropy of text
- Measures complexity (unique chars / total chars)
- Detects redundancy through compression ratio
- Provides metrics for adaptive modulation

#### AdaptiveModulationPlanner
- Selects optimal modulation scheme based on content
- Considers entropy, complexity, and redundancy
- Balances efficiency vs. robustness

#### DigitalModulator
- Implements multiple modulation schemes:
  - BFSK (Binary Frequency Shift Keying)
  - BPSK (Binary Phase Shift Keying)
  - QPSK (Quadrature Phase Shift Keying)
  - QAM16 (16-Quadrature Amplitude Modulation)
  - OFDM (Orthogonal Frequency Division Multiplexing)
- Generates audio waveforms from bits
- Supports error correction (Hamming, repetition)

#### SecurityUtils
- AES-GCM encryption
- Watermark generation and embedding
- HMAC authentication
- Key derivation

#### DualLLMOrchestrator
- Coordinates local and remote LLMs
- Fallback mechanisms
- Response validation
- Performance tracking

## Data Flow

### CCL Analysis Flow

```
Python Code → AST Parse → Function Discovery →
Random Probing → Entropy Calculation → Impurity Analysis →
Ghost Score Computation → Report Generation
```

### Julia Bridge Flow

```
CCL Results → JSON Parse → Metric Extraction →
KFP Smoothing → Ghost Score Calculation →
Hotspot Identification → JSON Response
```

### WaveCaster Flow

```
Text Input → Content Analysis → Scheme Selection →
Text to Bits → Error Correction → Modulation →
Security Layer → Signal Output (WAV/IQ)
```

### Integrated Flow

```
Code Analysis (CCL) → Coherence Metrics (Julia) →
Optimization (AL-ULS) → Training (PyTorch) →
Text Generation (LLM) → Signal Modulation (WaveCaster)
```

## Deployment Architecture

### Docker Services

```yaml
Services:
  - mock-al-uls:     Port 8000  (FastAPI)
  - julia-bridge:    Port 8099  (Julia HTTP server)
  - ccl-tools:       On-demand  (CLI tools)
  - wavecaster:      On-demand  (CLI/Service)
```

### Service Dependencies

```
ccl-tools → julia-bridge → mock-al-uls
     ↓           ↓              ↓
  Analysis → Coherence → Optimization
```

### Network Communication

- CCL tools communicate with Julia bridge via HTTP (port 8099)
- Julia bridge communicates with AL-ULS via HTTP (port 8000)
- All services use JSON for data exchange
- Health checks ensure service availability

## Scalability Considerations

### Horizontal Scaling
- CCL analysis can be parallelized across multiple workers
- Julia bridge is stateless and can be replicated
- Mock AL-ULS can handle concurrent requests

### Performance Optimization
- Julia provides high-performance numerical computing
- Matrix operations leverage BLAS/LAPACK
- WaveCaster uses NumPy/SciPy for efficient signal processing

### Caching Strategies
- CCL results can be cached per code version
- Julia bridge can cache coherence calculations
- WaveCaster can cache modulation schemes

## Security Architecture

### Authentication
- API keys for LLM services
- HMAC for message authentication
- Environment variable-based secrets

### Encryption
- AES-GCM for data encryption
- TLS for network communication (production)
- Secure key derivation (PBKDF2)

### Input Validation
- JSON schema validation
- Type checking in Julia
- Sanitization of user inputs

## Monitoring & Observability

### Logging
- Structured logging with levels (DEBUG, INFO, WARN, ERROR)
- Log aggregation via Docker logs
- Performance metrics tracking

### Health Checks
- HTTP health endpoints for all services
- Docker healthcheck configuration
- Automated restart on failure

### Metrics
- Request/response times
- Error rates
- Resource utilization
- Ghost score distributions

## Extension Points

### Adding New Modulation Schemes
1. Implement modulation function in `DigitalModulator`
2. Add scheme to `ModulationScheme` enum
3. Update `AdaptiveModulationPlanner` selection logic
4. Add tests

### Adding New CCL Metrics
1. Add metric calculation in `probe()` function
2. Update `analyze()` to aggregate new metric
3. Update Julia bridge to parse new metric
4. Update ghost score formula

### Integrating New LLM Providers
1. Add provider configuration
2. Implement provider-specific API client
3. Update `DualLLMOrchestrator`
4. Add authentication handling

## Future Enhancements

### Planned Features
- Real-time signal demodulation
- Web UI for visualization
- Database for persistent storage
- Advanced ML integration
- Distributed processing

### Research Directions
- Improved ghost detection algorithms
- Novel modulation schemes
- LLM-driven optimization
- Automated coherence repair

## References

- [Shannon Entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory))
- [Digital Modulation](https://en.wikipedia.org/wiki/Modulation)
- [Julia Language](https://julialang.org/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

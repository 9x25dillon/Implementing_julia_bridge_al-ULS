.PHONY: help install test clean build start stop restart ccl bridge-test examples docs

# Default target
help:
	@echo "CCL + WaveCaster System - Build Automation"
	@echo ""
	@echo "Available targets:"
	@echo "  install          - Install Python and Julia dependencies"
	@echo "  test             - Run test suite"
	@echo "  clean            - Clean build artifacts and output files"
	@echo "  build            - Build Docker containers"
	@echo "  start            - Start all services with Docker Compose"
	@echo "  stop             - Stop all services"
	@echo "  restart          - Restart all services"
	@echo "  ccl TARGET=path  - Run CCL analysis on target path"
	@echo "  bridge-test      - Test Julia bridge connection"
	@echo "  examples         - Run example scripts"
	@echo "  docs             - Build documentation"
	@echo "  lint             - Run linters (black, flake8, mypy)"
	@echo "  format           - Format code with black and isort"

# Installation
install:
	@echo "Installing Python dependencies..."
	pip install -r requirements.txt
	@echo "Installing Julia dependencies..."
	cd vectorizer && julia --project -e 'using Pkg; Pkg.instantiate()'
	@echo "Installation complete!"

install-dev:
	@echo "Installing development dependencies..."
	pip install -e ".[dev]"
	pre-commit install
	@echo "Development environment ready!"

# Testing
test:
	@echo "Running test suite..."
	pytest -v --cov=. --cov-report=term-missing

test-unit:
	@echo "Running unit tests..."
	pytest -v -m unit

test-integration:
	@echo "Running integration tests..."
	pytest -v -m integration

test-security:
	@echo "Running security tests..."
	pytest -v -m security

test-performance:
	@echo "Running performance benchmarks..."
	pytest -v -m performance --benchmark-only

# Code quality
lint:
	@echo "Running linters..."
	black --check .
	flake8 .
	mypy .
	@echo "Linting complete!"

format:
	@echo "Formatting code..."
	black .
	isort .
	@echo "Formatting complete!"

# Docker operations
build:
	@echo "Building Docker containers..."
	docker-compose build
	@echo "Build complete!"

start:
	@echo "Starting services..."
	docker-compose up -d mock-al-uls julia-bridge
	@echo "Services started!"
	@echo "Mock AL-ULS: http://localhost:8000"
	@echo "Julia Bridge: http://localhost:8099"

stop:
	@echo "Stopping services..."
	docker-compose down
	@echo "Services stopped!"

restart: stop start

logs:
	docker-compose logs -f

# CCL operations
ccl:
	@if [ -z "$(TARGET)" ]; then \
		echo "Error: TARGET not specified. Usage: make ccl TARGET=/path/to/code"; \
		exit 1; \
	fi
	@echo "Running CCL analysis on $(TARGET)..."
	python ccl.py $(TARGET) --report ccl_report.json
	@echo "Report saved to ccl_report.json"

ccl-wavecaster:
	@echo "Running CCL analysis on wavecaster.py..."
	python ccl.py wavecaster.py --report wavecaster_ccl_report.json
	@echo "Report saved to wavecaster_ccl_report.json"

# Bridge testing
bridge-test:
	@echo "Testing Julia bridge health..."
	curl -f http://localhost:8099/health || echo "Julia bridge not running"

bridge-optimize:
	@echo "Testing Julia bridge optimization..."
	curl -X POST http://localhost:8099/optimize \
		-H "Content-Type: application/json" \
		-d '{"adjacency": [[0.0, 0.8], [0.2, 0.0]], "mode": "kfp", "beta": 0.8}'

mock-test:
	@echo "Testing mock AL-ULS server..."
	curl -f http://localhost:8000/health || echo "Mock AL-ULS not running"

# Examples
examples:
	@echo "Running example scripts..."
	@if [ -d "examples" ]; then \
		for script in examples/*.py; do \
			echo "Running $$script..."; \
			python $$script; \
		done; \
	else \
		echo "No examples directory found"; \
	fi

# Documentation
docs:
	@echo "Building documentation..."
	@if [ -d "docs" ]; then \
		cd docs && make html; \
	else \
		echo "No docs directory found"; \
	fi

# Cleanup
clean:
	@echo "Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.log" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	rm -f .coverage coverage.xml
	rm -f *.wav *.iq *.png
	rm -f modulation_history.json
	rm -f ccl_report.json wavecaster_ccl_report.json
	@echo "Cleanup complete!"

deep-clean: clean
	@echo "Deep cleaning (including Docker)..."
	docker-compose down -v --rmi all 2>/dev/null || true
	rm -rf dist build
	@echo "Deep cleanup complete!"

# Development
dev-server:
	@echo "Starting development server..."
	uvicorn mock_al_uls_server:app --host 0.0.0.0 --port 8000 --reload

julia-repl:
	@echo "Starting Julia REPL with project..."
	cd vectorizer && julia --project

# CI/CD
ci: install lint test
	@echo "CI pipeline complete!"

# All-in-one commands
all: clean install build test
	@echo "Full build and test complete!"

quick-start: build start bridge-test mock-test
	@echo "Quick start complete!"
	@echo ""
	@echo "Services are running:"
	@echo "  - Mock AL-ULS: http://localhost:8000/docs"
	@echo "  - Julia Bridge: http://localhost:8099/health"
	@echo ""
	@echo "Try running:"
	@echo "  make ccl TARGET=wavecaster.py"
	@echo "  make examples"

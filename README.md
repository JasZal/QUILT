# Enjoying The High: One-Time Functional Encryption for Polynomials

This artifact benchmarks an one-time multi-input functional encryption scheme for evaluating polynomials of arbitrary degree.

## Overview

Running the benchmark starts the `benchmark.sh` script, which first executes `main.go` to generate the benchmark data. 
Once the benchmarks have finished, `plotHigher.py` is executed automatically to generate the figures from the collected data.

The benchmark configuration can be adjusted directly in `main.go`. In particular, it allows configuring:
- `rounds`: the number of benchmark iterations used for averaging,
- `deg`: the polynomial degree to evaluate,
- as well as other experiment parameters.

## Requirements

The artifact requires:
- Linux
- Go 1.24 or later
- Python 3 with the `pandas` and `matplotlib` packages installed

## Repository Structure

- **schemes/**  
  Implements the polynomial functional encryption scheme together with the cryptographic building blocks described in the paper.

- **benchmarking/**  
  Contains the generic benchmarking framework. Benchmarks are performed for all core algorithms: Setup, Encryption, Key Generation, and Decryption.

- **utils/**  
  Provides helper functions used throughout the implementation.

## Running the Benchmarks

To execute the complete benchmark pipeline, simply run:

```bash
./benchmark.sh
```

This script first executes `main.go` to generate the benchmark data and then runs `plotHigher.py` to automatically generate the corresponding plots.

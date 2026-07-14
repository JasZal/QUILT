# Artifact for Benchmarking QUILT - A New Construction for Quadratic One-Time Noisy Multi-Client Functional Encryption Schemes
This artifact benchmarks QUILT.

## Requirements
This repo requires a linux system, with golang version 1.24 or higher installed. 

## Folder Structure
- **schemes**  
  Contains the code for **QUILT** and some building blocks.

- **benchmarking**  
  Contains code for generic benchmarking and comparison to alternative schemes.

- **training**  
  Contains code for logistic regression training.

  
## Benchmarking

main.go performs benchmarking for the scheme by Zalonis et al. (ZSHA) and QUILT.
The schemes are evaluated for all algorithms: setup, encryption, keygen and decryption.

## Training
This method trains a logistic regrestion and stores results in the specified file (*resultsAverage.txt* )

# Reproducibility Guide

> **Status**: TODO - To be completed upon paper acceptance  
> **Last Updated**: 2025-10-05

---

## Overview

This guide will enable complete reproduction of results from *Designing Multi-Compartment Last-Mile Vehicle Fleets: An Open-Source Matheuristic* (submitted to *Computers and Industrial Engineering*).

**Paper Version**: Will be tagged as `paper-1.0.0` in the repository upon acceptance

## Quick Start

```bash
# Install FleetMix
git clone https://github.com/ekohan/fleetmix.git
cd fleetmix
./init.sh

# Run benchmarks (commands TBD upon paper finalization)
fleetmix benchmark mcvrp
fleetmix benchmark case
```

## Experiments

### Section 5: Effectiveness of Matheuristic

**TODO**: Document commands to reproduce Table 1 and benchmark comparisons with Henke et al. (2015, 2019)

**Instances**: Located in `src/fleetmix/benchmarking/datasets/mcvrp/`

### Section 6: Case Study

**TODO**: Document commands to reproduce baseline scenario and sensitivity analysis

**Data**: Located in `src/fleetmix/benchmarking/datasets/case/`

## Computational Environment

**Hardware** (paper experiments):
- **Processor**: Apple Silicon M1 (2020)
- **Cores**: 8 CPU cores
- **Memory**: 16 GB RAM

**Software**:
- **Python**: 3.12
- **Gurobi**: (version TBD)
- **PyVRP**: (version TBD)

## Citation

If using FleetMix in research:

```bibtex
@article{Kohan2025FleetMix,
  author  = {Eric Kohan and Fabricio Torres and Victor Silva-Febre and J.C. Pina-Pardo},
  title   = {Designing Multi-Compartment Last-Mile Vehicle Fleets: An Open-Source Matheuristic},
  journal = {Computers and Industrial Engineering},
  year    = {2025},
  note    = {Submitted}
}
```

## Support

- **Issues**: [github.com/ekohan/fleetmix/issues](https://github.com/ekohan/fleetmix/issues)
- **Contact**: See paper for author contact information

---

**Last Updated**: 2025-10-05

**Navigation**: [← Docs Home](README.md) | [Architecture](ARCHITECTURE.md)

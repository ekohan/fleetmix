# FleetMix Documentation

> **Comprehensive documentation for the FleetMix matheuristic package**  
> Companion to: *Designing Multi-Compartment Last-Mile Vehicle Fleets: An Open-Source Matheuristic*

---

## 📚 Documentation Guide

This documentation serves **two audiences**:

1. **🔬 Researchers**: Understand the matheuristic, reproduce results, extend the methodology
2. **👷 Practitioners**: Use FleetMix for real-world fleet design, customize for specific needs

Choose your starting point based on your goal:

### For Researchers

| Document | Purpose |
|----------|---------|
| [**ARCHITECTURE.md**](ARCHITECTURE.md) | System overview, how modules connect to the matheuristic pipeline |
| [**mapping.md**](mapping.md) | Direct cross-reference between paper sections and code |
| [**specs/**](specs/) | Detailed module specifications with mathematical formulations |
| [**REPRODUCIBILITY.md**](REPRODUCIBILITY.md) | How to reproduce all experiments from the paper |

### For Practitioners

| Document | Purpose |
|----------|---------|
| [**quickstart.md**](quickstart.md) | Get started quickly with a real example |
| [**USER_GUIDE.md**](USER_GUIDE.md) | Complete end-to-end workflow guide |
| [**specs/configuration.md**](specs/configuration.md) | All configuration parameters explained |
| [**specs/protocols.md**](specs/protocols.md) | How to plug in custom components |

### For Both Audiences

| Document | Purpose |
|----------|---------|
| [**experimental_design.md**](experimental_design.md) | Experimental methodology and parameter sensitivity |
| [**debugging.md**](debugging.md) | Troubleshooting and diagnostic tools |
| [**parallelism.md**](parallelism.md) | Performance optimization strategies |

---

## 📐 Module Specifications

The `specs/` directory contains detailed specifications for each major component:

### Core Pipeline Modules

1. [**vehicle_configurations.md**](specs/vehicle_configurations.md) - Generating vehicle configurations (Paper §4.1)
2. [**clustering.md**](specs/clustering.md) - Feasible cluster generation (Paper §4.2)
3. [**route_time_estimation.md**](specs/route_time_estimation.md) - Route duration computation (Paper §4.2)
4. [**optimization.md**](specs/optimization.md) - Fleet size and mix MILP (Paper §4.3)
5. [**post_optimization.md**](specs/post_optimization.md) - Improvement phase (Paper §4.4)
6. [**pipeline.md**](specs/pipeline.md) - End-to-end orchestration

### Supporting Modules

7. [**benchmarking.md**](specs/benchmarking.md) - Benchmark suite and validation (Paper §5-6)
8. [**data_model.md**](specs/data_model.md) - Core data structures and types
9. [**configuration.md**](specs/configuration.md) - Configuration system
10. [**protocols.md**](specs/protocols.md) - Plugin architecture

---

## 🗺️ Navigation Map

```
FleetMix Documentation Structure
│
├── README.md (you are here)
│
├── Getting Started
│   ├── quickstart.md
│   └── USER_GUIDE.md
│
├── Architecture & Design
│   ├── ARCHITECTURE.md
│   ├── mapping.md (Paper ↔ Code)
│   └── specs/ (detailed module specs)
│
├── Research & Extension
│   ├── REPRODUCIBILITY.md
│   └── experimental_design.md
│
└── Operations & Debugging
    ├── debugging.md
    └── parallelism.md
```

---

## 🎯 Quick Access by Task

### "I want to understand how the matheuristic works"
→ Start with [ARCHITECTURE.md](ARCHITECTURE.md), then explore [specs/](specs/)

### "I want to reproduce the paper experiments"
→ Read [REPRODUCIBILITY.md](REPRODUCIBILITY.md)

### "I want to solve my own fleet design problem"
→ Follow [quickstart.md](quickstart.md) then [USER_GUIDE.md](USER_GUIDE.md)

### "I want to implement a variant"
→ Study [specs/protocols.md](specs/protocols.md) for extension points

### "I want to add a custom clustering algorithm"
→ See [specs/protocols.md](specs/protocols.md) and [specs/clustering.md](specs/clustering.md)

### "I want to understand what a specific paper equation does in code"
→ Check [mapping.md](mapping.md) to find the implementation

---

## 📖 Documentation Philosophy

FleetMix is **spec-driven**: each module has a detailed specification that:

1. **Connects to the paper**: Maps directly to mathematical formulations and algorithms
2. **Explains design decisions**: Why this approach, what trade-offs were made
3. **Provides interfaces**: Clear input/output contracts
4. **Enables extension**: Shows how to customize or replace components
5. **Serves both audiences**: Academic rigor + practical usability

All specifications follow a [consistent template](templates/MODULE_SPEC_TEMPLATE.md).

---

## 🔗 External Resources

- **Paper**: *Designing Multi-Compartment Last-Mile Vehicle Fleets: An Open-Source Matheuristic*
- **Repository**: [github.com/ekohan/fleetmix](https://github.com/ekohan/fleetmix)
- **PyPI**: [pypi.org/project/fleetmix](https://pypi.org/project/fleetmix)
- **Issues**: [github.com/ekohan/fleetmix/issues](https://github.com/ekohan/fleetmix/issues)

---

## 📝 Contributing to Documentation

Found an error or gap? Documentation improvements are welcome!

1. Follow the [MODULE_SPEC_TEMPLATE.md](templates/MODULE_SPEC_TEMPLATE.md) for new specs
2. Maintain bidirectional cross-references (paper → code, code → paper)
3. Include examples for both researchers and practitioners
4. Keep notation consistent with the paper

---

**Version**: 1.0.0 (aligned with paper submission)  
**Last Updated**: 2025-10-05


# Experimental Design for Multi-Compartment Vehicle Fleet Optimization Study

## Abstract

This document presents the experimental design for a comprehensive comparison study evaluating the cost-effectiveness of Multi-Compartment Vehicles (MCVs) versus Single-Compartment Vehicles (SCVs) in last-mile food distribution. The study comprises two complementary experiments: (1) homogeneous fleet comparison and (2) mixed fleet optimization, both employing a Randomized Complete Block Design (RCBD) with 70 historical demand days as blocks.

## 1. Research Objectives

### Primary Objective
Quantify the cost and operational performance differences between traditional single-compartment vehicle fleets and multi-compartment vehicle fleets for temperature-controlled last-mile delivery across varying MCV cost parameters.

### Research Questions
1. Under what cost conditions do homogeneous MCV fleets outperform homogeneous SCV fleets?
2. When given the choice, how does an optimizer balance SCVs and MCVs in a mixed fleet configuration?
3. What is the relative benefit of fleet heterogeneity versus homogeneity in multi-temperature distribution?

## 2. Experimental Framework

### 2.1 Overall Design Structure

| Component | Description |
|-----------|-------------|
| **Design Type** | Randomized Complete Block Design (RCBD) with paired comparisons |
| **Blocks** | 70 unique historical demand days capturing real-world variation in customer orders, volumes, and geographic patterns |
| **Replication** | Every block receives all treatments (complete design) |
| **Randomization** | Order of treatment application within blocks randomized to avoid systematic bias |
| **Common Random Numbers** | Identical demand realizations and external conditions across treatments within each block |

### 2.2 Two-Experiment Structure

The study consists of two parallel experiments sharing the same blocking structure and parameter grid:

**Experiment 1: Homogeneous Fleet Comparison**
- Control: Homogeneous SCV fleet (baseline)
- Treatment: Homogeneous MCV fleet with varying cost parameters

**Experiment 2: Mixed Fleet Optimization**
- Control: Homogeneous SCV fleet (same baseline)
- Treatment: Heterogeneous fleet where optimizer selects from both SCV and MCV options

## 3. Treatment Structure

### 3.1 Fleet Configurations

| Experiment | Fleet Type | Vehicle Options | Description |
|------------|------------|-----------------|-------------|
| 1 | SCV Baseline | SCV only | Each vehicle handles single temperature class |
| 1 | MCV Homogeneous | MCV only | Each vehicle handles multiple temperature classes |
| 2 | SCV Baseline | SCV only | Identical to Experiment 1 baseline |
| 2 | Mixed Fleet | SCV + MCV | Optimizer chooses vehicle mix based on demand and costs |

### 3.2 Cost Parameter Grid

The MCV cost structure is parameterized by two factors:

**α (Vehicle Surcharge Factor)**
- Definition: Fixed cost multiplier where MCV fixed cost = α × SCV fixed cost
- Grid: α ∈ {1.00, 1.05, 1.10, ..., 1.40}
- Interpretation: Percentage premium for MCV acquisition/operation

**C (Compartment Setup Cost)**
- Definition: Additional fixed cost per compartment beyond the first
- Grid: C ∈ {0%, 5%, 10%, 15%} of baseline SCV fixed cost
- Interpretation: Operational overhead for multi-compartment configuration

**Note**: Cost surcharges apply exclusively to MCVs; SCV costs remain at baseline across all treatments.

### 3.3 Complete Treatment Set

Each experiment evaluates:
- 1 SCV baseline configuration
- |α| × |C| MCV configurations (9 × 4 = 36 in standard grid)
- Total treatments per experiment: 37 configurations
- Total observations: 37 configurations × 70 blocks = 2,590 per experiment

## 4. Control Variables

The following variables are held constant within each block to ensure fair comparison:

### 4.1 Operational Constraints
- Maximum route duration per driver (hours)
- Vehicle gross payload capacity (kg)
- Per-stop service time (minutes)
- Driver shift patterns and availability

### 4.2 Demand Characteristics
- Customer geographic locations (longitude, latitude)
- Demand quantities & mix

### 4.3 External Conditions
- External operating conditions (traffic, weather) through the **common‑random‑numbers (CRN)** principle

## 5. Response Variables

### 5.1 Primary Metrics
- **Total Cost**: Sum of fixed vehicle costs, variable routing costs, and compartment setup costs
- **Cost Difference**: d_ij = Cost_Treatment_ij - Cost_SCV_ij for block i, configuration j

### 5.2 Secondary Metrics
| Metric | Definition | Unit |
|--------|------------|------|
| Fleet Size | Number of vehicles deployed | vehicles |
| MCV Share | Proportion of MCVs in mixed fleet (Exp. 2 only) | % |
| Cost per Drop | Total cost / number of deliveries | $/delivery |
| Cost per kg | Total cost / total demand weight | $/kg |
| Split Delivery Rate | Customers receiving multiple visits | % |
| Vehicle Utilization | Average payload usage | % |
| Route Duration | Average time per route | hours |

## 6. Mathematical Formulation

### 6.1 Cost Model

For any fleet configuration:

TotalCost = Σ(f_v × x_v) + Σ(c_r × y_r) + Σ(C × n_v^comp)

Where:
- f_v = fixed cost of vehicle type v
- x_v = number of vehicles of type v used
- c_r = variable cost of route r
- y_r = binary indicator for route r selection
- n_v^comp = additional compartments in vehicle type v

### 6.2 MCV Cost Application

For MCV vehicles specifically:
- f_MCV = α × f_SCV (vehicle surcharge)
- Additional cost = C × (compartments - 1) per vehicle

## 7. Statistical Analysis Plan

### 7.1 Linear Mixed-Effects Model

For each experiment:

Cost_ij = μ + τ_j + β_i + ε_ij

Where:
- i = block (demand day), i ∈ {1, ..., 70}
- j = treatment configuration, j ∈ {1, ..., J}
- μ = grand mean
- τ_j = fixed treatment effect
- β_i ~ N(0, σ²_β) = random block effect
- ε_ij ~ N(0, σ²) = residual error

### 7.2 Hypothesis Testing

**Global Test**
- H₀: All treatment effects equal (τ₁ = τ₂ = ... = τ_J)
- H_A: At least one treatment differs
- Test: F-test or likelihood ratio test at α = 0.05

**Pairwise Comparisons**
- Primary contrasts: Each MCV configuration vs SCV baseline
- Method: 95% confidence intervals with Benjamini-Hochberg adjustment
- Effect size: Cohen's d for paired differences

### 7.3 Non-parametric Alternatives

If normality assumptions violated:
- Friedman test for global hypothesis
- Wilcoxon signed-rank test for pairwise comparisons
- Nemenyi post-hoc test with appropriate corrections

## 8. Implementation Details

### 8.1 Computational Infrastructure
- **Optimization Solver**: FleetMix matheuristic (open-source)
- **Clustering Algorithm**: Temperature-aware hierarchical clustering
- **Route Optimization**: Variable Neighborhood Search (VNS) with local search operators
- **Computing Environment**: Parallel execution across parameter grid

### 8.2 Data Sources
- **Demand Data**: 70 days of real-world historical demand from commercial food distributor
  - Captures natural variation in daily order patterns, customer mix, and volumes
  - Each day represents a unique demand realization with different characteristics
- **Geographic Data**: Real customer locations with actual driving distances
- **Product Mix**: Actual SKU-level demand across three temperature classes (Ambient/Chilled/Frozen)

### 8.3 Reproducibility Measures
- Fixed random seeds per block for stochastic components
- Version-controlled configuration files
- Complete audit trail of solver decisions

## 9. Expected Outcomes and Interpretation

### 9.1 Parameter Sensitivity Analysis
- Heat map of cost differences across (α, C) grid
- Break-even contours where MCV = SCV performance
- Probability of superiority surfaces

### 9.2 Fleet Composition Analysis (Experiment 2 only)
- MCV adoption rate as function of cost parameters
- Demand characteristics driving vehicle type selection
- Economies of scope in multi-temperature distribution

## 10. Limitations and Assumptions

### 10.1 Scope Limitations
- Single depot operations only
- Homogeneous vehicle capacities within type
- Historical demand realization (no forecasting uncertainty modeled)
- No driver-specific constraints or preferences

### 10.2 Key Assumptions
- Linear cost scaling with compartments
- No learning curve effects in MCV operations
- Perfect information about demand at planning time
- Negligible differences in vehicle reliability/maintenance

## 11. Key Differentiators Between Experiments

### Experiment 1: Homogeneous Fleet Analysis
- **Objective**: Establish pure performance comparison between vehicle types
- **Constraint**: Fleet must be entirely SCV or entirely MCV
- **Interpretation**: Reveals maximum theoretical benefit of MCVs under perfect fleet homogeneity

### Experiment 2: Mixed Fleet Analysis
- **Objective**: Determine optimal fleet composition under realistic operational flexibility
- **Constraint**: Optimizer free to choose any combination of SCVs and MCVs
- **Interpretation**: Reveals practical benefit when fleet heterogeneity is permitted
- **Additional Insight**: MCV adoption rate indicates demand patterns favoring multi-compartment solutions
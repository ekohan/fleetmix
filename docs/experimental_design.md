# Experimental Design: Multi-Compartment Vehicle Fleet Optimization

## Overview

We evaluate Multi-Compartment Vehicles (MCVs) versus Single-Compartment Vehicles (SCVs) for multi-product last-mile delivery through two experiments using Randomized Complete Block Design with 70 historical demand days as blocks.

## Research Questions

1. **Homogeneous Fleet Comparison**: Under what cost conditions do pure MCV fleets outperform pure SCV fleets?
2. **Mixed Fleet Optimization**: When given choice, what fleet composition emerges from cost minimization?
3. **Value of Flexibility**: What is the benefit of heterogeneous versus homogeneous fleet constraints?

## Experimental Design

**Design Type**: Randomized Complete Block Design (RCBD)  
**Blocks**: 70 historical demand days (June–August 2024) from commercial food distributor  
**Common Random Numbers**: Identical demand, geography, and operational conditions within each block

### Two-Experiment Structure

| Experiment | Treatment | Fleet Options | Research Focus |
|------------|-----------|---------------|----------------|
| 1 | SCV Baseline | SCV only | Pure SCV performance |
| 1 | MCV Homogeneous | MCV only | Pure MCV performance at varying costs |
| 2 | SCV Baseline | SCV only | Same as Experiment 1 |
| 2 | Mixed Fleet | SCV + MCV | Endogenous vehicle selection |

## Treatment Parameters

MCV cost structure varies across two dimensions:

**α (Vehicle Premium)**
- Definition: MCV fixed cost = α × SCV fixed cost
- Range: α ∈ [1.0, 2.0] in 11 steps (0%, 10%, 20%, ..., 100% premium)

**C (Compartment Setup Cost)**  
- Definition: Additional fixed cost per compartment beyond the first
- Range: C ∈ [0%, 50%] in 6 steps (0%, 10%, 20%, 30%, 40%, 50% of SCV fixed cost)

**Treatment Count**
- Experiment 1: 1 SCV baseline + 66 MCV configurations (11 × 6) = 67 treatments
- Experiment 2: 1 SCV baseline + 66 mixed configurations = 67 treatments  
- Total observations per experiment: 67 × 70 = 4,690

## Control Variables

Held constant within each block:
- Route duration limit, vehicle capacity, service time per stop
- Customer locations (latitude, longitude)  
- Demand quantities and temperature-class mix
- External conditions (via common random numbers)

## Response Variables

**Primary Metric**: Total cost difference vs SCV baseline  
Δ*ij* = Cost(*MCV/Mixed, i, j*) − Cost(*SCV, i*)

**Secondary Metrics**:
- Fleet composition (vehicles deployed, MCV share in Exp. 2)
- Operational efficiency (route duration, vehicle utilization, distance traveled)
- Service quality (split delivery rate, visits per customer)
- Unit economics (cost per delivery, cost per kg)

## Cost Model

Total daily cost comprises:

**Total Cost** = Fixed + Compartment Setup + Variable (hours of operation per route)

Where:
- Fixed = Σ(f*v* × x*v*) for vehicle type *v* with fixed cost f*v* and count x*v*
- Variable = Σ(c*r*) for selected routes *r* with cost c*r*  
- Compartment = Σ(C × [n*v* − 1]) for MCVs with n*v* compartments

MCV cost structure:
- f*MCV* = α × f*SCV*  
- Additional cost = C per compartment beyond first

## Statistical Analysis

**Design**: Randomized Complete Block Design with 70 demand days as blocks

**Outputs** (averaged across 70 demand days):
- Cost difference across (α, C) grid
- Fleet composition patterns (Experiment 2)

## Implementation

**Solver**: FleetMix matheuristic
**Data**: 70 historical demand days (June–August 2024), real customer locations, three product classes  

## Demand Characteristics

Summary statistics across 70 historical demand days (June–August 2024):

| Metric | Min | Max | Mean | Median |
|--------|-----|-----|------|--------|
| Daily customers | 208 | 691 | 379 | 373 |
| Daily demand (tonnes) | 12.6 | 49.1 | 24.9 | 24.7 |

**Product mix**: ~46% of customers require multiple multiple product classes (Dry/Chilled/Frozen in all combinations).
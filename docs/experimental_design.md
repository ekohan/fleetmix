# Experimental Design for FleetMix Comparison Study
<!-- TODO: make shorter -->

## 1. Objective

Quantify the cost and operational performance difference between **single‑compartment vehicle (SCV)** fleets and **multi‑compartment vehicle (MCV)** fleets for last‑mile food distribution, across a grid of MCV cost parameters *(α, C)*.

## 2. Design Summary – Randomised Complete Block Design (RCBD)

| Element                    | Description                                                                                                     |
| -------------------------- | --------------------------------------------------------------------------------------------------------------- |
| **Blocks**                 | 70 historical demand days (real customer orders & routes)                                                       |
| **Treatments**             | 1 × SCV baseline + *J* MCV configurations (unique combinations of vehicle‑surcharge **α** and setup‑cost **C**) |
| **Observations per block** | *J + 1* total‑cost evaluations — one for each fleet configuration                                               |
| **Replication**            | Every block receives *all* treatments (hence “complete”)                                                        |
| **Randomisation**          | Order of fleet runs within each block is randomised to avoid systematic solver side‑effects                     |

## 3. Control Variables (held *ceteris paribus* within each block)

* Route‑time limit per driver (hours)
* Gross payload capacity (kg)
* Per‑stop service time (minutes)
* Geographic customer locations
* SKU‑level demand quantities & temperature mix (Dry/Chill/Frozen)
* External operating conditions (traffic, weather) through the **common‑random‑numbers (CRN)** principle

## 4. Cost Model

\$\text{TotalCost} = f\_{\text{veh}} + C\_{\text{setup}} + c\_{\text{dist}}\$

* **α (vehicle surcharge)**: \$f\_{\text{MCV}} = α,f\_{\text{SCV}}\$
  Reported as “+ % over SCV”.
* **C (compartment setup)**: Per‑additional‑compartment cost, expressed as a % of SCV fixed cost.

## 5. Statistical Analysis Plan

### 5.1 Linear Mixed‑Effects Model

```text
Cost_{ij} = μ + τ_j + β_i + ε_{ij}
```

* *i* = day (1…70) — random block effect βᵢ ∼ 𝒩(0, σ²\_β)
* *j* = fleet configuration (1…J+1) — fixed treatment effect τⱼ

> Implemented with `statsmodels.MixedLM` (Python) or `lme4::lmer` (R).

### 5.2 Global Hypothesis Test

* *H₀*: all treatment effects equal (τ₁ = … = τⱼ).
* F‑test (or likelihood‑ratio) at α = 0.05.

### 5.3 Post‑hoc Contrasts

* Pairwise contrasts **MCVⱼ – SCV** with 95 % CIs.
* Adjust p-values for multiple comparisons (Benjamini–Hochberg).

### 5.4 Non‑parametric Robustness

* Friedman test (k‑sample extension of Wilcoxon signed‑rank).
* Nemenyi or Dunn–Šidák post‑hoc if distributional assumptions fail.

## 6. Parameter‑Sensitivity Surface

* Grid over α ∈ {0 %, 5 %, … 40 %} and C ∈ {0 %, 5 %, 10 %, 15 %}.
* For each (α, C) cell: run full RCBD, compute Δ% cost and operational KPIs.
* Visualise with:

  * **Break‑even heat‑map** — Δ% cost surface, contour at 0 %.
  * **Probability‑of‑superiority** — % of days where MCV < SCV.

## 7. Key Metrics Reported (all indexed to SCV = 1.00)

* Δ% Total cost, cost‑per‑kg, cost‑per‑drop
* Vehicles per day, average load factor, split‑delivery rate
* Driver‑hours (if modelled)
* 95 % confidence intervals across demand days

## 8. Advantages of This Design

1. **Variance reduction** — blocking removes day‑to‑day heterogeneity.
2. **Fair comparison** — CRN enforces identical demand realisations across fleets.
3. **Statistical power** — 70 paired observations more efficient than 140 unpaired.
4. **Scalability insights** — same framework reused across the (α, C) grid.

## 9. Reproducibility & Artefacts

* Solver: **FleetMix** (open‑source, PyPI & GitHub).
* Data: 70 anonymised distribution days (available upon request or via companion repo).
* RNG seeds fixed per block to guarantee CRN.
* Analysis notebooks & figures auto‑generated; CI propagation via bootstrapping.

## 10. References

* Montgomery, *Design and Analysis of Experiments*, 9th ed.
* Law & Kelton, *Simulation Modeling and Analysis*, 5th ed.
* Benjamini & Hochberg, “Controlling the False Discovery Rate,” 1995.

---

*Last update: July 26, 2025.*

### Experimental Design for Fleet Performance Comparison

| Section | Detail | Description |
| :--- | :--- | :--- |
| **1. Research Objective** | | To evaluate the relative cost-effectiveness of a homogeneous Multi-Compartment Vehicle (MCV) fleet against a homogeneous Single-Compartment Vehicle (SCV) baseline across a range of real-world demand scenarios and MCV cost parameters. |
| **2. Experimental Design** | **Design Type** | Paired Comparison using a Randomized Block Design. |
| | **Core Principle** | Common Random Numbers (CRN) applied to ensure a fair comparison by using identical inputs for both treatment levels within each block. |
| **3. Factors and Levels** | **Treatment Factor** | **Fleet Configuration** (2 Levels) <br> &nbsp;&nbsp;&nbsp;&nbsp;Level 1 (Control): Homogeneous SCV Fleet <br> &nbsp;&nbsp;&nbsp;&nbsp;Level 2 (Treatment): Homogeneous MCV Fleet |
| | **Blocking Factor** | **Daily Demand Realization** (70 Levels) <br> &nbsp;&nbsp;&nbsp;&nbsp;Each block consists of a unique, real-world daily demand profile. |
| **4. Control Variables** | **Demand** | Specific customer set, geographic locations, demand quantities, and product mix are held constant within each block. |
| | **Constraints** | Maximum route duration, vehicle gross payload capacity, and per-stop service times are held constant within each block. |
| | **Conditions** | External factors (e.g., traffic) are assumed to be identical for both fleet types within a block. |
| **5. Sensitivity Analysis** | **Procedure** | The entire blocked experiment is systematically repeated for various combinations of MCV cost parameters. |
| | **Parameters** | &alpha;: Variable cost multiplier for MCV fleet. <br> C: Fixed cost component for MCV fleet. |
| **6. Measurement** | **Response Variable** | Paired Cost Difference per Day (d_i): <br> &nbsp;&nbsp;&nbsp;&nbsp; `d_i = TotalCost_MCV,i - TotalCost_SCV,i` |
| **7. Analysis** | **Statistical Test** | **Wilcoxon Signed-Rank Test** <br> &nbsp;&nbsp;&nbsp;&nbsp;*Null Hypothesis (H₀)*: The median of the paired cost differences is zero. |


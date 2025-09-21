# Investigation Plan: SCV Baseline Outperforming Mixed Fleet

## Problem Statement
In 622 cases (13.5%) from our experiments, the SCV-only baseline achieves lower operational costs than mixed fleet configurations. This is counterintuitive because the mixed fleet optimizer should be able to select an SCV-only fleet if that's the optimal solution.

### Key Statistics
- **Cases affected**: 622 out of 4,620 (13.5%)
- **Mean cost penalty**: 1.6%
- **Maximum cost penalty**: 6.1%
- **Worst case**: sales_2024-07-04_demand with α=1.8, C=40.0

## Known Facts (Eliminated Causes)
- ✗ Clustering is identical for both SCV and mixed runs
- ✗ Not related to solver gaps or timeouts
- ✗ No light load penalties involved
- ✗ Math model formulation will not be changed
- ✗ Vehicle speeds and service times are identical

## Simplified Investigation Approach

### Remaining Hypotheses

#### H1: Vehicle Configuration Generation
**Theory**: Mixed fleet generates different vehicle configurations or too many options
**Check**: Compare number and types of configurations generated
**Target**: Cases with high α values (many potential MCV types)
**Quick Test**: Print configurations for 5 worst cases

#### H2: Compartment Penalty Application
**Theory**: Mixed fleet applies compartment penalties differently or when SCV doesn't
**Check**: Compare penalty calculations between runs
**Target**: Cases with non-zero penalties
**Quick Test**: Detail penalty breakdown for worst cases

#### H3: Fixed Cost Assignment
**Theory**: Vehicle fixed costs are calculated or assigned differently
**Check**: Compare fixed cost calculations for identical vehicle usage
**Target**: Cases with similar vehicle counts but different costs
**Quick Test**: Print cost breakdowns

#### H4: Vehicle Selection Logic
**Theory**: Mixed fleet has a bias or constraint that prevents pure SCV selection
**Check**: See if mixed ever selects 100% SCVs when it should
**Target**: Cases with low MCV share (< 10%)
**Quick Test**: Force SCV-only configuration in mixed fleet

#### H5: Numerical Precision in Cost Aggregation
**Theory**: Small rounding differences accumulate differently in complex vs simple fleets
**Check**: Compare exact cost calculations with high precision
**Target**: Cases with very small cost differences (< 1%)
**Quick Test**: Use decimal arithmetic for comparison

### Implementation Process

For each hypothesis:

1. **Analysis Phase** (2-3 hours)
   - Extract target instances based on indicators
   - Run detailed diagnostics on sample
   - Confirm hypothesis relevance

2. **Implementation Phase** (3-4 hours)
   - Make targeted changes to core pipeline
   - Ensure backward compatibility
   - Add unit tests for changes

3. **Validation Phase** (1-2 hours)
   - Run improved version on test sample
   - Compare against baseline
   - Document improvements

4. **Decision Phase** (30 min)
   - If improvement > 20% reduction in affected cases: KEEP
   - If improvement 10-20%: REFINE and retest
   - If improvement < 10%: REJECT and document

### Priority Order

1. **H1** (Solver Configuration) - Quick win, easy to test
2. **H3** (Initial Solution) - Medium effort, high impact
3. **H2** (Configuration Enumeration) - May explain high-α cases
4. **H5** (Constraints) - Fundamental issue if present
5. **H4** (Precision) - Lower priority, affects few cases
6. **H6** (Cost Calculation) - Last resort, requires careful validation

### Testing Infrastructure

#### Sample Selection Tool
```python
def select_hypothesis_sample(hypothesis: str, n: int = 15) -> List[Tuple[str, float, float]]:
    """Select instances most likely affected by hypothesis."""
    # Implementation in scv_wins_diagnostic.py
```

#### Quick Validation Script
```python
def validate_hypothesis_fix(
    hypothesis: str,
    sample: List[Tuple[str, float, float]], 
    modification: Callable
) -> HypothesisResult:
    """Run targeted test on hypothesis sample."""
    # Implementation in improvement_tester.py
```

## Success Criteria

### Must Have
- Reduce SCV wins from 13.5% to <5%
- No performance degradation in unaffected cases
- Clear documentation of changes

### Nice to Have
- Reduce SCV wins to <2%
- Improve average solution quality
- Reduce solver runtime

## Risk Mitigation

### Technical Risks
- **Changes break existing functionality**: Comprehensive test suite
- **Performance degradation**: Benchmark before/after each change
- **Solver instability**: Test with multiple solver backends

### Process Risks
- **Scope creep**: Strict phase gates and success criteria
- **Analysis paralysis**: Time-boxed investigation phases
- **Insufficient testing**: Automated test framework

## Deliverables

1. **Analysis Report**: Root causes of SCV wins
2. **Test Framework**: Automated testing for improvements
3. **Implementation**: Code changes with documentation
4. **Performance Report**: Before/after metrics
5. **Recommendations**: Long-term improvements

## Next Steps

1. Create `scv_wins_diagnostic.py` to extract and analyze problematic cases
2. Set up A/B testing framework for comparing solutions
3. Begin Phase 1.1 data collection

---

*Created: 2025-09-21*
*Status: Planning*
*Owner: TBD*

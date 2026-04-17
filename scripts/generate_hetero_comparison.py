#!/usr/bin/env python3
"""
Generate the heterogeneous Henke experiment report (HTML).

Reads:
  - results/henke_heterogeneous/hetero_all.csv       (n=10, 300 rows)
  - results/henke_heterogeneous_n15/hetero_n15_all.csv (n=15, 10 rows)
  - results/henke_heterogeneous/tsp_of_all.csv        (TSP-all sanity)

Writes:
  - reports/hetero_comparison.html

Usage:
    uv run python scripts/generate_hetero_comparison.py
"""

from __future__ import annotations

import csv
import statistics
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HETERO_CSV = PROJECT_ROOT / "results/henke_heterogeneous/hetero_all.csv"
HETERO_N15_CSV = PROJECT_ROOT / "results/henke_heterogeneous_n15/hetero_n15_all.csv"
TSP_ALL_CSV = PROJECT_ROOT / "results/henke_heterogeneous/tsp_of_all.csv"
OUT_HTML = PROJECT_ROOT / "reports/hetero_comparison.html"

# ── Inline styles ─────────────────────────────────────────────────────────
TH = (
    'style="border: 1px solid #ccc; padding: 6px 10px; background: #f5f5f5; '
    'font-weight: 600; text-align: center; font-size: 12px;"'
)
TD = 'style="border: 1px solid #ccc; padding: 5px 10px; text-align: center; font-size: 12px;"'
TD_L = 'style="border: 1px solid #ccc; padding: 5px 10px; text-align: left; font-size: 12px;"'
TD_R = 'style="border: 1px solid #ccc; padding: 5px 10px; text-align: right; font-size: 12px;"'
TD_B = (
    'style="border: 1px solid #ccc; padding: 5px 10px; text-align: center; '
    'font-weight: 600; font-size: 12px;"'
)
GREEN = '<span style="color:green">{}</span>'
TABLE = '<table style="border-collapse: collapse; margin: 12px 0;">\n'
CALLOUT_YELLOW = (
    '<div style="background: #fffbe6; border-left: 4px solid #e0a800; '
    'padding: 10px 14px; margin: 12px 0; font-size: 13px;">{}</div>'
)
CALLOUT_GREEN = (
    '<div style="background: #e8f5e9; border-left: 4px solid #388e3c; '
    'padding: 12px 16px; margin: 14px 0; font-size: 13px;">{}</div>'
)


def _load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _split_by_method(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {}
    for r in rows:
        out.setdefault(r["method"], []).append(r)
    return out


def _fleet_bucket(n_a: int, n_b: int) -> str:
    if n_a > 0 and n_b == 0:
        return "A-only"
    if n_a == 0 and n_b > 0:
        return "B-only"
    return "mixed"


def _signed(v: float) -> str:
    return f"+{v:.2f}%" if v >= 0 else f"{v:.2f}%"


# ── Summary tables (reused for n=10 and n=15) ─────────────────────────────


def _fleet_agreement_table(by_method: dict[str, list[dict[str, str]]]) -> str:
    S: list[str] = [TABLE, "<thead><tr>"]
    for h in ["Method", "Instances", "Same fleet size",
              "Same A count", "Same B count", "Full composition match"]:
        S.append(f"  <th {TH}>{h}</th>")
    S.append("</tr></thead><tbody>")
    for method in ("BHH", "TSP"):
        mr = by_method.get(method, [])
        if not mr:
            continue
        n = len(mr)
        same_fleet = sum(1 for r in mr if int(r["v_gap"]) == 0)
        same_a = sum(1 for r in mr if int(r["h_n_A"]) == int(r["e_n_A"]))
        same_b = sum(1 for r in mr if int(r["h_n_B"]) == int(r["e_n_B"]))
        same_full = sum(1 for r in mr
                        if int(r["h_n_A"]) == int(r["e_n_A"])
                        and int(r["h_n_B"]) == int(r["e_n_B"]))
        S.append(f"<tr><td {TD_L}><b>{method}</b></td><td {TD}>{n}</td>"
                 f"<td {TD_B}>{same_fleet}/{n}</td>"
                 f"<td {TD_B}>{same_a}/{n}</td>"
                 f"<td {TD_B}>{same_b}/{n}</td>"
                 f"<td {TD_B}>{same_full}/{n}</td></tr>")
    S.append("</tbody></table>")
    return "\n".join(S)


def _fleet_mix_table(by_method: dict[str, list[dict[str, str]]]) -> str:
    S: list[str] = [TABLE, "<thead><tr>"]
    for h in ["Method", "A-only", "B-only", "Mixed (A+B)", "Total"]:
        S.append(f"  <th {TH}>{h}</th>")
    S.append("</tr></thead><tbody>")
    for method in ("BHH", "TSP"):
        mr = by_method.get(method, [])
        if not mr:
            continue
        from collections import Counter
        buckets = Counter(_fleet_bucket(int(r["e_n_A"]), int(r["e_n_B"])) for r in mr)
        n = len(mr)
        S.append(f"<tr><td {TD_L}><b>{method}</b> (exhaustive-optimal)</td>"
                 f"<td {TD}>{buckets.get('A-only',0)} ({100*buckets.get('A-only',0)/n:.0f}%)</td>"
                 f"<td {TD}>{buckets.get('B-only',0)} ({100*buckets.get('B-only',0)/n:.0f}%)</td>"
                 f"<td {TD_B}>{buckets.get('mixed',0)} ({100*buckets.get('mixed',0)/n:.0f}%)</td>"
                 f"<td {TD}>{n}</td></tr>")
    S.append("</tbody></table>")
    return "\n".join(S)


def _cost_gap_table(by_method: dict[str, list[dict[str, str]]]) -> str:
    S: list[str] = [TABLE, "<thead><tr>"]
    for h in ["Method", "Instances", "Avg |gap|", "Median", "p90", "Max", "Exact (0%)"]:
        S.append(f"  <th {TH}>{h}</th>")
    S.append("</tr></thead><tbody>")
    for method in ("BHH", "TSP"):
        mr = by_method.get(method, [])
        if not mr:
            continue
        gaps = sorted(abs(float(r["cost_gap_pct"])) for r in mr)
        n = len(gaps)
        exact = sum(1 for g in gaps if g == 0)
        S.append(f"<tr><td {TD_L}><b>{method}</b></td><td {TD}>{n}</td>"
                 f"<td {TD}>{statistics.mean(gaps):.2f}%</td>"
                 f"<td {TD}>{statistics.median(gaps):.2f}%</td>"
                 f"<td {TD}>{gaps[min(int(0.9*n), n-1)]:.2f}%</td>"
                 f"<td {TD}>{max(gaps):.2f}%</td>"
                 f"<td {TD_B}>{exact}/{n}</td></tr>")
    S.append("</tbody></table>")
    return "\n".join(S)


def _cost_gap_split_table(by_method: dict[str, list[dict[str, str]]]) -> str:
    S: list[str] = [TABLE, "<thead><tr>"]
    for h in ["Method", "Composition vs exhaustive", "Instances", "Avg |gap|", "Max |gap|"]:
        S.append(f"  <th {TH}>{h}</th>")
    S.append("</tr></thead><tbody>")
    for method in ("BHH", "TSP"):
        mr = by_method.get(method, [])
        if not mr:
            continue
        same = [r for r in mr if r["h_n_A"] == r["e_n_A"] and r["h_n_B"] == r["e_n_B"]]
        diff = [r for r in mr if not (r["h_n_A"] == r["e_n_A"] and r["h_n_B"] == r["e_n_B"])]
        for label, group in [("Same composition", same), ("Different composition", diff)]:
            if not group:
                continue
            gaps = [abs(float(r["cost_gap_pct"])) for r in group]
            S.append(f"<tr><td {TD_L}><b>{method}</b></td><td {TD_L}>{label}</td>"
                     f"<td {TD}>{len(group)}</td>"
                     f"<td {TD}>{statistics.mean(gaps):.2f}%</td>"
                     f"<td {TD}>{max(gaps):.2f}%</td></tr>")
    S.append("</tbody></table>")
    return "\n".join(S)


def _cluster_pool_table(by_method: dict[str, list[dict[str, str]]]) -> str:
    S: list[str] = [TABLE, "<thead><tr>"]
    for h in ["Method", "Instances",
              "Matheuristic clusters<br>(avg / min / max)",
              "Exhaustive clusters<br>(avg / min / max)", "Ratio (Math / Exh)"]:
        S.append(f"  <th {TH}>{h}</th>")
    S.append("</tr></thead><tbody>")
    for method in ("BHH", "TSP"):
        mr = by_method.get(method, [])
        if not mr:
            continue
        h_c = [int(r["h_clusters"]) for r in mr]
        e_c = [int(r["e_clusters"]) for r in mr]
        ratios = [h / e for h, e in zip(h_c, e_c) if e > 0]
        S.append(f"<tr><td {TD_L}><b>{method}</b></td><td {TD}>{len(mr)}</td>"
                 f"<td {TD}>{statistics.mean(h_c):.0f} / {min(h_c)} / {max(h_c)}</td>"
                 f"<td {TD}>{statistics.mean(e_c):.0f} / {min(e_c)} / {max(e_c)}</td>"
                 f"<td {TD_B}>{statistics.mean(ratios)*100:.1f}%</td></tr>")
    S.append("</tbody></table>")
    return "\n".join(S)


def _runtime_table(by_method: dict[str, list[dict[str, str]]]) -> str:
    S: list[str] = [TABLE, "<thead><tr>"]
    for h in ["Method", "Instances", "Matheuristic (s)<br>avg / max",
              "Exhaustive (s)<br>avg / max", "Speedup (Exh / Math)"]:
        S.append(f"  <th {TH}>{h}</th>")
    S.append("</tr></thead><tbody>")
    for method in ("BHH", "TSP"):
        mr = by_method.get(method, [])
        if not mr:
            continue
        h_t = [float(r["h_time"]) for r in mr]
        e_t = [float(r["e_time"]) for r in mr]
        h_avg, e_avg = statistics.mean(h_t), statistics.mean(e_t)
        speedup = e_avg / h_avg if h_avg > 0 else 0
        S.append(f"<tr><td {TD_L}><b>{method}</b></td><td {TD}>{len(mr)}</td>"
                 f"<td {TD_R}>{h_avg:.1f} / {max(h_t):.1f}</td>"
                 f"<td {TD_R}>{e_avg:.1f} / {max(e_t):.1f}</td>"
                 f"<td {TD_B}>{speedup:.1f}×</td></tr>")
    S.append("</tbody></table>")
    return "\n".join(S)


def _per_instance_tables(by_method: dict[str, list[dict[str, str]]],
                         has_supply: bool = True) -> str:
    S: list[str] = []
    for method in ("BHH", "TSP"):
        mr = sorted(by_method.get(method, []),
                    key=lambda r: (r.get("supply", ""), r["instance"]))
        if not mr:
            continue
        S.append(f"<h4>{method}</h4>")
        S.append(TABLE)
        S.append("<thead><tr>")
        headers = ["Instance"]
        if has_supply:
            headers.append("Supply")
        headers += ["Math #Veh", "Math #A", "Math #B", "Math Cost",
                    "Math Clusters", "Math Time (s)",
                    "Exh #Veh", "Exh #A", "Exh #B", "Exh Cost",
                    "Exh Clusters", "Exh Time (s)", "Fleet gap", "Cost gap"]
        for h in headers:
            S.append(f"  <th {TH}>{h}</th>")
        S.append("</tr></thead><tbody>")
        for r in mr:
            vgap = int(r["v_gap"])
            cgap = float(r["cost_gap_pct"])
            S.append("<tr>")
            S.append(f"  <td {TD_L}>{r['instance']}</td>")
            if has_supply:
                S.append(f"<td {TD}>{r['supply']}</td>")
            S.append(
                f"<td {TD}>{r['h_vehicles']}</td>"
                f"<td {TD}>{r['h_n_A']}</td><td {TD}>{r['h_n_B']}</td>"
                f"<td {TD_R}>{float(r['h_cost']):.2f}</td>"
                f"<td {TD}>{r['h_clusters']}</td>"
                f"<td {TD_R}>{float(r['h_time']):.1f}</td>"
                f"<td {TD}>{r['e_vehicles']}</td>"
                f"<td {TD}>{r['e_n_A']}</td><td {TD}>{r['e_n_B']}</td>"
                f"<td {TD_R}>{float(r['e_cost']):.2f}</td>"
                f"<td {TD}>{r['e_clusters']}</td>"
                f"<td {TD_R}>{float(r['e_time']):.1f}</td>"
                f"<td {TD}>{GREEN.format(0) if vgap == 0 else vgap}</td>"
                f"<td {TD}>{_signed(cgap)}</td>")
            S.append("</tr>")
        S.append("</tbody></table>")
    return "\n".join(S)


# ── Main generation ───────────────────────────────────────────────────────


def generate(rows_n10: list[dict[str, str]],
             tsp_all: list[dict[str, str]],
             rows_n15: list[dict[str, str]]) -> str:
    S: list[str] = []
    S.append(
        '<div style="font-family: -apple-system, BlinkMacSystemFont, \'Segoe UI\', '
        "Roboto, sans-serif; font-size: 14px; line-height: 1.6; color: #222; "
        'max-width: 1200px;">'
    )
    S.append(f"<p><i>Generated {time.strftime('%Y-%m-%d %H:%M')}.</i></p>")

    bm_n10 = _split_by_method(rows_n10)
    bm_n15 = _split_by_method(rows_n15)

    # ══════════════════════════════════════════════════════════════════════
    # SETUP
    # ══════════════════════════════════════════════════════════════════════
    S.append("<h2>Setup</h2>")

    S.append(
        "<p><b>Instances.</b> 150 Henke 2015 instances with n=10 customers "
        "(3 supply variants × 50 each), plus 5 Henke 2019 instances with "
        "n=15. Customer locations and demands are unchanged from the "
        "original benchmarks; we replace only the fleet and cost structure, "
        "which Henke does not parametrize.</p>"
    )

    S.append(
        "<p><b>Fleet.</b> Two vehicle types per instance set:</p>"
    )
    S.append(TABLE)
    S.append("<thead><tr>")
    for h in ["", "Vehicle A", "Vehicle B"]:
        S.append(f"  <th {TH}>{h}</th>")
    S.append("</tr></thead><tbody>")
    S.append(f"<tr><td {TD_L}><b>n=10</b> (Henke 2015)</td>"
             f"<td {TD}>cap 1000 (= Henke), fixed 100</td>"
             f"<td {TD}>cap 500 (half), fixed 60</td></tr>")
    S.append(f"<tr><td {TD_L}><b>n=15</b> (Henke 2019)</td>"
             f"<td {TD}>cap 1125 (= Henke), fixed 110</td>"
             f"<td {TD}>cap 562 (half), fixed 70</td></tr>")
    S.append("</tbody></table>")

    S.append(
        "<p>Fixed costs come from our Bogotá case study (capacities "
        "700 / 1,300 / 2,500 → fixed costs 80 / 140 / 180). Cost grows "
        "sub-linearly with capacity — bigger vehicles are cheaper per unit "
        "capacity. We use that same curve to set the costs here. Economies "
        "of scale are preserved: one A is cheaper than two B.</p>"
    )

    S.append(
        "<p><b>All other parameters are identical to the case study:</b> "
        "variable cost 10/hour, speed 30 km/h, service 25 min/customer, "
        "compartment setup cost 10, max route duration <b>10 h</b>, "
        "split-stops disabled. Three product types (Dry, Chilled, Frozen).</p>"
    )

    # TSP-of-all sanity (inline)
    if tsp_all:
        hours = [float(r["tsp_tour_hours"]) for r in tsp_all]
        exceed = sum(1 for r in tsp_all if r["exceeds_10h"] == "1")
        S.append(
            f"<p><b>Max route duration is binding.</b> We computed the "
            f"optimal TSP tour over all 10 customers per instance (single "
            f"vehicle, unlimited capacity) — i.e., the best-case single-"
            f"vehicle route. Over 150 n=10 instances: min {min(hours):.2f} h, "
            f"mean {statistics.mean(hours):.2f} h, max {max(hours):.2f} h. "
            f"<b>{exceed}/150 ({100*exceed/len(tsp_all):.0f}%) exceed 10 h"
            f"</b> on duration alone; the rest are capacity-bound.</p>"
        )

    S.append(
        "<p><b>Pipelines.</b> Per instance:</p>"
        "<ul>"
        "<li><b>Matheuristic</b> — clustering + MILP + post-optimization.</li>"
        "<li><b>Exhaustive</b> — enumerate every feasible (customer-subset, "
        "vehicle-config) pair → same MILP, no post-opt. Provably optimal "
        "at n=10 and n=15.</li>"
        "</ul>"
        "<p>Both run under BHH (closed-form route-time approximation) and "
        "exact TSP (PyVRP). Four pipelines per instance.</p>"
    )

    S.append(
        "<p><b>Metrics per instance.</b> Fleet size, composition (#A, #B), "
        "total cost, clusters fed to MILP, end-to-end wall-clock time.</p>"
    )

    # ══════════════════════════════════════════════════════════════════════
    # n=10 RESULTS
    # ══════════════════════════════════════════════════════════════════════
    S.append("<hr />")
    S.append("<h2>Results — Henke 2015, n=10 (150 instances)</h2>")

    S.append("<h3>Fleet-size and composition agreement</h3>")
    S.append(_fleet_agreement_table(bm_n10))

    S.append("<h3>Fleet-mix distribution — does heterogeneity matter?</h3>")
    S.append(_fleet_mix_table(bm_n10))

    S.append("<h3>Cost gap — matheuristic vs exhaustive</h3>")
    S.append(_cost_gap_table(bm_n10))

    S.append("<h4>Split by composition match</h4>")
    S.append(_cost_gap_split_table(bm_n10))

    # Diagnosis callout
    n_diff = sum(1 for r in bm_n10.get("BHH", [])
                 if not (r["h_n_A"] == r["e_n_A"] and r["h_n_B"] == r["e_n_B"]))
    S.append(CALLOUT_YELLOW.format(
        f"<b>Diagnosis.</b> The {n_diff} composition mismatches follow one "
        "pattern: the matheuristic picks a different A/B mix than the optimum "
        "but finds the correct total vehicle count. Root cause: the "
        "post-optimization merge phase merges only within the same vehicle "
        "type — a restriction from the original homogeneous setting. "
        "Dropping it (or adding a cross-type reassignment pass) should close "
        "most of the cost-gap tail. Fleet <i>size</i> is exact on 150/150 "
        "under both BHH and TSP. Identified extension for future work."
    ))

    S.append("<h3>Cluster pool fed to the MILP</h3>")
    S.append(_cluster_pool_table(bm_n10))

    S.append("<h3>Runtime (end-to-end, includes cluster generation)</h3>")
    S.append(_runtime_table(bm_n10))

    S.append("<h3>Per-instance detail (n=10)</h3>")
    S.append(_per_instance_tables(bm_n10, has_supply=True))

    # ══════════════════════════════════════════════════════════════════════
    # n=15 RESULTS
    # ══════════════════════════════════════════════════════════════════════
    if rows_n15:
        S.append("<hr />")
        S.append("<h2>Results — Henke 2019, n=15 (5 instances)</h2>")

        S.append(
            "<p>Same design as n=10. Vehicle A capacity matched to Henke 2019 "
            "(1125), Vehicle B at half (562). Exhaustive enumeration at n=15 "
            "generates ~12,000–26,000 clusters per instance vs ~70–300 for "
            "the matheuristic.</p>"
        )

        # Parameter tuning callout
        S.append(CALLOUT_GREEN.format(
            "<b>Parameter tuning (for discussion).</b>"
            "<p style='margin: 6px 0;'>FleetMix exposes two parameters that "
            "control how aggressively the cluster generation explores "
            "candidate groupings before the MILP:</p>"
            "<ul style='margin: 4px 0;'>"
            "<li><code>pre_small_cluster_size</code> — clusters with up to "
            "this many customers are considered \"small\" and eligible for "
            "pre-MILP merging. Two small clusters can be combined into one "
            "larger cluster that may only be feasible for the bigger vehicle, "
            "giving the MILP more options.</li>"
            "<li><code>pre_nearest_merge_candidates</code> — how many nearby "
            "clusters to consider as merge partners.</li>"
            "</ul>"
            "<p style='margin: 6px 0;'>With the case-study defaults "
            "(<code>pre_small_cluster_size=5</code>, "
            "<code>pre_nearest_merge_candidates=50</code>), one of five n=15 "
            "instances had a fleet-size miss: the matheuristic used 3 vehicles "
            "(2A+1B) where the optimum is 2 (2A). Setting "
            "<code>pre_small_cluster_size=15</code> (= n, so all clusters are "
            "eligible for merging) and "
            "<code>pre_nearest_merge_candidates=500</code> resolved it — the "
            "matheuristic found the correct 2-vehicle fleet, and the cost gap "
            "dropped from 23% to 7%.</p>"
            "<p style='margin: 6px 0;'>This is a configuration choice, not a "
            "code change. FleetMix is designed so practitioners can tune the "
            "exploration-vs-speed trade-off for their instance size.</p>"
            "<p style='margin: 6px 0;'><b>Question for the team:</b> "
            "should we adopt <code>pre_small_cluster_size = n</code> as the "
            "default for these experiments? It is a natural choice for small "
            "instances and does not affect the case study (which uses "
            "<code>pre_small_cluster_size=5</code> on much larger instances). "
            "The n=15 BHH results below use the tweaked parameters; n=15 TSP "
            "still uses the defaults (pending rerun, ~8 h).</p>"
        ))

        S.append("<h3>Fleet-size and composition agreement</h3>")
        S.append(_fleet_agreement_table(bm_n15))

        S.append("<h3>Cost gap</h3>")
        S.append(_cost_gap_table(bm_n15))

        S.append("<h3>Cluster pool and runtime</h3>")
        S.append(_cluster_pool_table(bm_n15))
        S.append(_runtime_table(bm_n15))

        S.append("<h3>Per-instance detail (n=15)</h3>")
        S.append(_per_instance_tables(bm_n15, has_supply=False))

    S.append("</div>")
    return "\n".join(S)


def main() -> None:
    rows_n10 = _load_csv(HETERO_CSV)
    rows_n15 = _load_csv(HETERO_N15_CSV)
    tsp_all = _load_csv(TSP_ALL_CSV)
    if not rows_n10:
        print(f"ERROR: No data at {HETERO_CSV}")
        return
    html = generate(rows_n10, tsp_all, rows_n15)
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(html)
    n10_bhh = sum(1 for r in rows_n10 if r["method"] == "BHH")
    n10_tsp = sum(1 for r in rows_n10 if r["method"] == "TSP")
    n15_bhh = sum(1 for r in rows_n15 if r["method"] == "BHH")
    n15_tsp = sum(1 for r in rows_n15 if r["method"] == "TSP")
    print(f"n=10: {len(rows_n10)} rows ({n10_bhh} BHH, {n10_tsp} TSP)")
    print(f"n=15: {len(rows_n15)} rows ({n15_bhh} BHH, {n15_tsp} TSP)")
    print(f"TSP-of-all: {len(tsp_all)} instances")
    print(f"HTML: {OUT_HTML}")


if __name__ == "__main__":
    main()

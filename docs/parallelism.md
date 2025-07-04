# Process-Level Parallelism in FleetMix

FleetMix accelerates the **clustering stage** with Joblib's `loky` backend. `loky` uses separate OS processes; each worker starts with a clean Python interpreter, so nothing created at run-time in the parent is automatically visible inside workers.

## Plugin visibility

Runtime registrations happen only in the parent process:
```python
from fleetmix.registry import register_clusterer

@register_clusterer("my_algo")
class MyAlgo: ...
```
Workers never execute that code, so the registry inside each worker is empty and FleetMix raises:
```
ValueError: Unknown clustering method 'my_algo'
```

## Two reliable fixes
1. **Package the plugin** in a real module and `import` it before invoking FleetMix so every worker imports it too.
2. **Force serial execution** for quick demos:
   ```bash
   export FLEETMIX_N_JOBS=1   # set *before* `import fleetmix`
   ```

## Quick troubleshooting checklist
• Unknown-plugin error? Ensure every worker can `import` the module that registers it.
• Globals look reset? Workers start clean—pass data explicitly.
• Hard to debug? Temporarily set `FLEETMIX_N_JOBS=1` for a single-process run. 
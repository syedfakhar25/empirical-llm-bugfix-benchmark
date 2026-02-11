# ============================================================
# baseline_measurement.py
# Measures idle system + framework overhead
# Runs 30 repetitions and stores average baseline energy (kWh)
# ============================================================

import json
import time
from codecarbon import EmissionsTracker

REPETITIONS = 30
DURATION_SECONDS = 15  # must be >= 10 sec for stable measurement

baseline_values = []

print("Starting baseline energy measurement...\n")

for i in range(REPETITIONS):

    tracker = EmissionsTracker(
        measure_power_secs=1,
        log_level="error"
    )

    tracker.start()

    # Let system idle for fixed time
    time.sleep(DURATION_SECONDS)

    emissions_kg = tracker.stop()

    # CodeCarbon internally tracks energy (kWh)
    energy_kwh = tracker.final_emissions_data.energy_consumed

    baseline_values.append(energy_kwh)

    print(f"Run {i+1:02d}: {energy_kwh:.8f} kWh")

# Compute average baseline energy
avg_baseline = sum(baseline_values) / len(baseline_values)

print("\n--------------------------------------")
print(f"Average Baseline Energy: {avg_baseline:.8f} kWh")
print("--------------------------------------\n")

# Save to JSON file
with open("baseline.json", "w") as f:
    json.dump(
        {"baseline_energy_kwh": avg_baseline},
        f,
        indent=4
    )

print("Baseline saved to baseline.json")
print("Done.")

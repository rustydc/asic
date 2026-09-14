"""Command-line entry point for the appliance simulator."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace

from sim.appliance import ApplianceConfig, Simulation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="sim/config/baseline.json")
    parser.add_argument("--trace", help="write a Chrome/Perfetto JSON trace")
    args = parser.parse_args()

    config = ApplianceConfig.from_json(args.config)
    if args.trace:
        config = replace(config, trace=True)
    simulation = Simulation(config)
    result = simulation.run()
    print(json.dumps(result.as_dict(), indent=2))
    if args.trace:
        simulation.write_chrome_trace(args.trace)


if __name__ == "__main__":
    main()


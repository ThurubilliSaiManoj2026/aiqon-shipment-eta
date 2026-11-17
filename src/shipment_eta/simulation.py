"""Generate complete synthetic journeys from the documented stage equations."""

import numpy as np
import pandas as pd

from .config import validate_config
from .schema import FULL_COLUMNS, validate_journeys
from .timing import hours_between, next_dispatch, utc_timestamp

GENERATOR_VERSION = "1.0.0"


def _disruption(rng, section):
    # Always consume both draws, even when the flag is zero. This is part of
    # generator version 1's reproducibility contract.
    flag = int(rng.binomial(1, section["disruption_probability"]))
    magnitude = rng.uniform(section["disruption_min_h"], section["disruption_max_h"])
    return flag, flag * magnitude


def generate_journeys(config: dict) -> pd.DataFrame:
    """Generate and validate a complete table without touching NumPy global RNG.

    SeedSequence child 0 is used for journeys; child 1 is reserved for future
    observation masking. Route shuffling precedes per-shipment draws. Within
    each shipment: departure normal/flag/magnitude; weather/voyage noise;
    congestion/processing noise/flag/magnitude; inland noise/flag/magnitude.
    """
    config = validate_config(config)
    journey_seed, _observation_seed = np.random.SeedSequence(config["seed"]).spawn(2)
    rng = np.random.Generator(np.random.PCG64(journey_seed))
    routes = config["routes"]
    assignments = np.repeat(np.arange(len(routes)), config["n_shipments"] // len(routes))
    rng.shuffle(assignments)
    departure, voyage, port, inland = (config[k] for k in ("departure", "voyage", "port", "inland"))
    first_departure = utc_timestamp(config["scheduled_start_utc"])
    rows = []
    for index, route_index in enumerate(assignments):
        route = routes[int(route_index)]
        s0 = first_departure + pd.Timedelta(hours=index * config["departure_spacing_h"])
        s1 = s0 + pd.Timedelta(hours=route["voyage_h"])
        s2 = s1 + pd.Timedelta(hours=route["port_processing_h"])
        s3 = s2 + pd.Timedelta(hours=route["inland_h"] + route["delivery_buffer_h"])

        deviation = max(departure["early_bound_h"], rng.normal(departure["normal_mean_h"], departure["normal_std_h"]))
        departure_flag, extra_departure = _disruption(rng, departure)
        a0 = s0 + pd.Timedelta(hours=deviation + extra_departure)
        weather = rng.beta(voyage["weather_beta_a"], voyage["weather_beta_b"])
        voyage_h = max(voyage["minimum_duration_fraction"] * route["voyage_h"],
                       route["voyage_h"] + voyage["weather_effect_h"] * weather
                       + rng.normal(0, voyage["noise_std_fraction"] * route["voyage_h"]))
        a1 = a0 + pd.Timedelta(hours=voyage_h)

        congestion = rng.beta(port["congestion_beta_a"], port["congestion_beta_b"])
        processing_noise = rng.normal(0, port["noise_std_h"])
        port_flag, extra_port = _disruption(rng, port)
        processing_h = max(port["minimum_duration_h"],
                           route["port_processing_h"] * (port["base_multiplier"] + port["congestion_multiplier"] * congestion)
                           + processing_noise + extra_port)
        a2 = a1 + pd.Timedelta(hours=processing_h)
        dispatch = next_dispatch(a2, s2, route["dispatch_interval_h"])
        inland_noise = rng.normal(0, inland["noise_std_fraction"] * route["inland_h"])
        inland_flag, extra_inland = _disruption(rng, inland)
        inland_h = max(inland["minimum_duration_h"], route["inland_h"] + inland_noise + extra_inland)
        a3 = dispatch + pd.Timedelta(hours=inland_h)
        signed_delay = hours_between(a3, s3)
        rows.append({
            "shipment_id": f"SHP-{index + 1:06d}",
            "route_id": route["route_id"], "origin": route["origin"],
            "destination_port": route["destination_port"], "destination": route["destination"],
            "scheduled_departure_utc": s0, "scheduled_port_arrival_utc": s1,
            "scheduled_ready_utc": s2, "scheduled_delivery_utc": s3,
            "actual_departure_utc": a0, "actual_port_arrival_utc": a1, "as_of_utc": a1,
            "planned_voyage_h": float(route["voyage_h"]),
            "planned_port_processing_h": float(route["port_processing_h"]),
            "planned_inland_h": float(route["inland_h"]),
            "delivery_buffer_h": float(route["delivery_buffer_h"]),
            "dispatch_interval_h": float(route["dispatch_interval_h"]),
            "voyage_weather_index": float(weather), "port_congestion_index": float(congestion),
            "actual_voyage_h": hours_between(a1, a0),
            "actual_ready_utc": a2, "actual_dispatch_utc": dispatch, "actual_delivery_utc": a3,
            "actual_port_processing_h": hours_between(a2, a1),
            "dispatch_wait_h": hours_between(dispatch, a2),
            "actual_inland_h": hours_between(a3, dispatch),
            "remaining_h": hours_between(a3, a1),
            "signed_arrival_deviation_h": signed_delay,
            "delay_magnitude_h": max(0.0, signed_delay),
            "departure_disruption": departure_flag, "port_disruption": port_flag,
            "inland_disruption": inland_flag,
        })
    result = pd.DataFrame(rows, columns=FULL_COLUMNS)
    validate_journeys(result, config)
    return result

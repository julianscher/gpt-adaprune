import argparse
import math
from typing import Dict


class Curriculum:
    def __init__(self, args: argparse.Namespace):
        # args.dims and args.points each contain start, end, inc, interval attributes
        # inc denotes the change in n_dims,
        # this change is done every interval,
        # and start/end are the limits of the parameter
        self.n_dims_truncated = args.curriculum["dims"]["start"]
        self.n_points = args.curriculum["points"]["start"]
        self.n_dims_schedule = args.curriculum["dims"]
        self.n_points_schedule = args.curriculum["points"]
        self.step_count = 0

    def update(self) -> None:
        self.step_count += 1
        self.n_dims_truncated = self.update_var(
            self.n_dims_truncated, self.n_dims_schedule
        )
        self.n_points = self.update_var(self.n_points, self.n_points_schedule)

    def update_var(self, var: float, schedule: Dict) -> float:
        if self.step_count % schedule["interval"] == 0:
            var += schedule["inc"]

        return min(var, schedule["end"])


# returns the final value of var after applying curriculum.
def get_final_var(init_var: float, total_steps: int, inc: int, n_steps: int, lim: float) -> float:
    final_var = init_var + math.floor((total_steps) / n_steps) * inc

    return min(final_var, lim)

import json
import os
import time
from enum import Enum
from typing import Optional, Dict, Any

import torch


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:

        if isinstance(obj, Enum):
            return obj.name

        if torch.is_tensor(obj):
            return obj.cpu().detach().numpy().tolist()

        try:
            return super().default(obj)
        except TypeError:
            return str(obj)

class SimpleLogger:
    def __init__(self, run_dir: Optional[str] = None, also_print: bool = True) -> None:
        self.run_dir = run_dir
        self.also_print = also_print
        self.log_file = None
        self.log_dict = {}
        if run_dir:
            os.makedirs(run_dir, exist_ok=True)
            self.log_file = open(os.path.join(run_dir, "log.txt"), "a", buffering=1)

    def log(self, msg: str) -> None:
        t = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{t}] {msg}"
        if self.also_print:
            print(line)
        if self.log_file:
            self.log_file.write(line + "\n")

    def log_validation(self, metrics: Dict[str, Any]) -> None:
        if "validation" not in self.log_dict:
            self.log_dict["validation"] = []

        validation_data = {
            "timestamp": time.time(),
            **metrics
        }
        self.log_dict["validation"].append(validation_data)
        self._save()

    def log_test(self, metrics: Dict[str, Any]) -> None:
        test_data = {
            "timestamp": time.time(),
            **metrics
        }
        self.log_dict["test"] = test_data
        self._save()

    def log_step(self, step: int, metrics: Dict[str, Any]) -> None:
        if "train" not in self.log_dict:
            self.log_dict["train"] = []

        step_entry = None
        for entry in self.log_dict["train"]:
            if entry["step"] == step:
                step_entry = entry
                break

        if step_entry:
            step_entry.update(metrics)
            step_entry["timestamp"] = time.time()
        else:
            step_data = {
                "step": step,
                "timestamp": time.time(),
                **metrics
            }
            self.log_dict["train"].append(step_data)

        self._save()

    def _save(self) -> None:
        """Save logged metrics to disk."""
        metrics_path = f"{self.run_dir}/log.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.log_dict, f, indent=2, cls=CustomJSONEncoder)

    def close(self) -> None:
        if self.log_file:
            self.log_file.close()

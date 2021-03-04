import os
import csv
import rmm
import tempfile
import random
from string import ascii_lowercase


class RMMResourceAnalyzer:
    """
    Class to control enabling, disabling, & parsing RMM resource
    logs.
    """

    def __init__(self, gpu_device_nums=[0]):
        # only a single GPU supported currently
        self._gpu_device_num = gpu_device_nums[0]
        self.max_gpu_util = -1
        self.max_gpu_mem_usage = 0
        log_name_suffix = "".join(random.choice(ascii_lowercase) for _ in range(6))
        log_file_name = "rapids_pytest_benchmarks_" + log_name_suffix
        self._log_file_path = os.path.join(tempfile.gettempdir(), log_file_name)

    def enable_logging(self):
        """
        Enable RMM logging. RMM creates a CSV output file derived from
        provided file name that looks like: log_file_path + ".dev0".
        """
        rmm.enable_logging(log_file_name=self._log_file_path)

    def disable_logging(self):
        """
        Disable RMM logging
        """
        log_output_file = self._log_file_path + ".dev" + str(self._gpu_device_num)
        rmm.mr._flush_logs()
        rmm.disable_logging()
        self._parse_results(log_output_file)
        os.remove(log_output_file)

    def _parse_results(self, log_file):
        """
        Parse CSV results. CSV file has columns:
        Thread,Time,Action,Pointer,Size,Stream
        """
        current_mem_usage = 0
        with open(log_file, mode="r") as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                row_action = row["Action"]
                row_size = int(row["Size"])

                if row_action == "allocate":
                    current_mem_usage += row_size
                    if current_mem_usage > self.max_gpu_mem_usage:
                        self.max_gpu_mem_usage = current_mem_usage

                if row_action == "free":
                    current_mem_usage -= row_size

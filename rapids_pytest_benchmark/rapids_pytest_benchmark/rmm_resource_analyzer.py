# Copyright (c) 2021-2023, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import csv
import rmm
import tempfile


class RMMResourceAnalyzer:
    """
    Class to control enabling, disabling, & parsing RMM resource
    logs.
    """

    def __init__(self):
        self.max_gpu_util = -1
        self.max_gpu_mem_usage = 0
        self.leaked_memory = 0
        log_file_name = "rapids_pytest_benchmarks_log"
        self._log_file_prefix = os.path.join(tempfile.gettempdir(), log_file_name)

    def enable_logging(self):
        """
        Enable RMM logging. RMM creates a CSV output file derived from
        provided file name that looks like: log_file_prefix + ".devX", where
        X is the GPU number.
        """
        rmm.enable_logging(log_file_name=self._log_file_prefix)

    def disable_logging(self):
        """
        Disable RMM logging
        """
        log_output_files = rmm.get_log_filenames()
        rmm.mr._flush_logs()
        rmm.disable_logging()
        # FIXME: potential improvement here would be to only parse the log files for
        # the gpu ID that's passed in via --benchmark-gpu-device
        self._parse_results(log_output_files)
        for _, log_file in log_output_files.items():
            os.remove(log_file)

    def _parse_results(self, log_files):
        """
        Parse CSV results. CSV file has columns:
        Thread,Time,Action,Pointer,Size,Stream
        """
        current_mem_usage = 0
        for _, log_file in log_files.items():
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
        self.leaked_memory = current_mem_usage

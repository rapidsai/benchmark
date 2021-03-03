import os
import csv
import rmm
import tempfile


class RMMResourceAnalyzer():
    """
    Class to control enabling, disabling, & parsing RMM resource
    logs.
    """
    def __init__(self):
        self.max_gpu_util = -1
        self.max_gpu_mem_usage = 0
        _, self._log_file = tempfile.mkstemp()

    def __del__(self):
      """
      Remove log file when instance is garbage collected
      """
      os.remove(self._log_file)

    def enable_logging(self):
      """
      Enable RMM logging
      """
      rmm.enable_logging(log_file_name=self._log_file)

    def disable_logging(self):
      """
      Disable RMM logging
      """
      rmm.mr._flush_logs()
      rmm.disable_logging()


    def parse_results(self):
      """
      Parse CSV results. CSV file has columns:
      Thread,Time,Action,Pointer,Size,Stream
      """
      current_mem_usage = 0
      # RMM appends the device ID is appended to filename
      log_file_name = self._log_file + ".dev0"
      with open(log_file_name, mode='r') as csv_file:
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

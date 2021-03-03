import cudf
from ..rmm_resource_analyzer import RMMResourceAnalyzer


def test_rmm_analyzer():
  inst = RMMResourceAnalyzer()
  inst.enable_logging()
  s = cudf.Series([1])
  del s
  inst.disable_logging()
  inst.parse_results()
  assert inst.max_gpu_mem_usage == 8

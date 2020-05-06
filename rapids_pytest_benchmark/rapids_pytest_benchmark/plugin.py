from functools import partial

import pytest
import pytest_benchmark

from .gpu_metric_poller import startGpuMetricPolling, stopGpuMetricPolling
from .reporting import GPUTableResults


__version__ = "0.1"


def pytest_addoption(parser):
    group = parser.getgroup("benchmark")
    group.addoption(
        "--benchmark-disable-gpu-metrics",
        action="store_true", default=False,
        help="Disable GPU measurements on the 'benchmark' fixture."
        )
    # FIXME: add check for valid dir, similar to "parse_save()" in pytest-benchmark
    group.addoption(
        "--benchmark-asv-output-dir",
        metavar="ASV_DB_DIR", default=None,
        help='ASV "database" directory to update with benchmark results.'
    )


# GPUBenchmarkFixture = None

# class GPUBenchmarkFixture(object):
#     def __init__(self, benchmarkObj):
#         self.__benchmarkObj = benchmarkObj
#     def __call__(self, function_to_benchmark, *args, **kwargs):
#         return self.__benchmarkObj(function_to_benchmark, *args, **kwargs)

#                gpuPollObj = startGpuMetricPolling()
#                stopGpuMetricPolling(gpuPollObj)
#                gpuMems.append(gpuPollObj.maxGpuUtil)
#                gpuUtils.append(gpuPollObj.maxGpuMemUsed)

class GPUBenchmarkResults:
    def __init__(self, runtime, gpuMem, gpuUtil):
        self.runtime = runtime
        self.gpuMem = gpuMem
        self.gpuUtil = gpuUtil


class GPUMetadata(pytest_benchmark.stats.Metadata):
    def __init__(self, fixture, iterations, options):
        super().__init__(fixture, iterations, options)
        self.stats = GPUStats()

    def update(self, gpuBenchmarkResults):
        # Assume GPU metrics do not accumulate over multiple runs like runtime does?
        self.stats.update(gpuBenchmarkResults.runtime / self.iterations,
                          gpuBenchmarkResults.gpuMem,
                          gpuBenchmarkResults.gpuUtil)


class GPUStats(pytest_benchmark.stats.Stats):
    fields = (
        "min", "max", "mean", "stddev", "rounds", "median", "gpu_mem", "gpu_util", "iqr", "q1", "q3", "iqr_outliers", "stddev_outliers",
        "outliers", "ld15iqr", "hd15iqr", "ops", "total"
    )

    def __init__(self):
        super().__init__()
        self.gpuData = []

    def update(self, duration, gpuMem, gpuUtil):
        super().update(duration)
        self.gpuData.append((gpuMem, gpuUtil))

    def as_dict(self):
        return super().as_dict()

    @pytest_benchmark.utils.cached_property
    def gpu_mem(self):
        #for i in self.gpuData: print(i)
        return max([i[0] for i in self.gpuData])
    @pytest_benchmark.utils.cached_property
    def gpu_util(self):
        return max([i[1] for i in self.gpuData])


class GPUBenchmarkFixture(pytest_benchmark.fixture.BenchmarkFixture):

    def __init__(self, benchmarkFixtureInstance):
        self.__benchmarkFixtureInstance = benchmarkFixtureInstance
        self.__timeOnlyRunner = None

    def __getattr__(self, attr):
        return getattr(self.__benchmarkFixtureInstance, attr)

    # def _make_runner(self, function_to_benchmark, args, kwargs):
    #     origBenchmarkRunner = super()._make_runner(function_to_benchmark,
    #                                                args, kwargs)
    #     self.__timeOnlyRunner = origBenchmarkRunner

    #     def runner(loops_range, timer=self._timer):
    #         gpuPollObj = startGpuMetricPolling()
    #         try:
    #             origRetVal = origBenchmarkRunner(loops_range, timer)
    #         finally:
    #             stopGpuMetricPolling(gpuPollObj)

    #         if loops_range:
    #             return GPUBenchmarkResults(runtime=origRetVal,
    #                                        gpuMem=gpuPollObj.maxGpuMemUsed,
    #                                        gpuUtil=gpuPollObj.maxGpuUtil)
    #         else:
    #             return (GPUBenchmarkResults(runtime=origRetVal[0],
    #                                         gpuMem=gpuPollObj.maxGpuMemUsed,
    #                                         gpuUtil=gpuPollObj.maxGpuUtil),
    #                     origRetVal[1])
    #     return runner

    def _make_runner(self, function_to_benchmark, args, kwargs):
        timeBenchmarkRunner = super()._make_runner(function_to_benchmark,
                                                   args, kwargs)
        self.__timeOnlyRunner = timeBenchmarkRunner

        def runner(loops_range, timer=self._timer):
            # GPU polling increases runtime for code that uses the GPU, so
            # perform runs to measure runtime separately

            # runtime measurement
            timeRetVal = timeBenchmarkRunner(loops_range, timer)
            # GPU measurements
            gpuPollObj = startGpuMetricPolling()
            try:
                timeBenchmarkRunner(loops_range, timer)
            finally:
                stopGpuMetricPolling(gpuPollObj)

            if loops_range:
                return GPUBenchmarkResults(runtime=timeRetVal,
                                           gpuMem=gpuPollObj.maxGpuMemUsed,
                                           gpuUtil=gpuPollObj.maxGpuUtil)
            else:
                return (GPUBenchmarkResults(runtime=timeRetVal[0],
                                            gpuMem=gpuPollObj.maxGpuMemUsed,
                                            gpuUtil=gpuPollObj.maxGpuUtil),
                        timeRetVal[1])
        return runner

    def _calibrate_timer(self, runner):
        return super()._calibrate_timer(self.__timeOnlyRunner)

    def _make_stats(self, iterations):
        bench_stats = GPUMetadata(self, iterations=iterations, options={
            "disable_gc": self._disable_gc,
            "timer": self._timer,
            "min_rounds": self._min_rounds,
            "max_time": self._max_time,
            "min_time": self._min_time,
            "warmup": self._warmup,
        })
        self._add_stats(bench_stats)
        self.stats = bench_stats
        return bench_stats


class GPUBenchmarkSession(pytest_benchmark.session.BenchmarkSession):
    compared_mapping = None
    groups = None

    def __init__(self, benchmarkSession):
        self.__benchmarkSessionInstance = benchmarkSession
        self.compared_mapping = benchmarkSession.compared_mapping
        self.groups = benchmarkSession.groups

        origColumns = self.columns
        self.columns = []
        for c in ["min", "max", "mean", "stddev", "median", "iqr", "outliers", "ops", "gpu_mem", "rounds", "iterations"]:
            if (c == "gpu_mem") or (c in origColumns):
                self.columns.append(c)

    def __getattr__(self, attr):
        return getattr(self.__benchmarkSessionInstance, attr)

    def display(self, tr):
        if not self.groups:
            return
        tr.ensure_newline()
        results_table = GPUTableResults(
            columns=self.columns,
            sort=self.sort,
            histogram=self.histogram,
            name_format=self.name_format,
            logger=self.logger,
            scale_unit=partial(self.config.hook.pytest_benchmark_scale_unit, config=self.config),
        )
        results_table.display(tr, self.groups)
        self.check_regressions()
        self.display_cprofile(tr)



def pytest_configure(config):
    return
    config._gpubenchmarksession = None


@pytest.fixture(scope="function")
def gpubenchmark(benchmark):
    return GPUBenchmarkFixture(benchmark)


def pytest_sessionstart(session):
    # # Replace the pytest-benchmark hook function objects with the wrappers defined here
    # for hc in session.config.pluginmanager.get_hookcallers(pytest_benchmark.plugin):
    #     if hc.name == "pytest_report_header":
    #         #hc.spec.function = rapids_pytest_report_header
    #         for hi in hc.get_hookimpls():
    #             if hi.plugin_name == "benchmark":
    #                 hi.function = rapids_pytest_report_header

    session.config._benchmarksession_orig = session.config._benchmarksession
    session.config._gpubenchmarksession = \
        GPUBenchmarkSession(session.config._benchmarksession)
    session.config._benchmarksession = session.config._gpubenchmarksession

    # p = session.config.pluginmanager.get_plugin("benchmark")
    # p = session.config.pluginmanager.unregister(pytest_benchmark.plugin)
    # session.config.pluginmanager.unregister(name="benchmark")
    # #session.config.pluginmanager.register(pytest_benchmark.plugin)
    # p.pytest_report_header = rapids_pytest_report_header
    # session.config.pluginmanager.register(p, name="benchmark")


def DISABLED_pytest_plugin_registered(plugin, manager):
    pass
#     if plugin != pytest_benchmark.plugin:
#         return

    #plugin = config.pluginmanager.get_plugin("benchmark")
    #plugin.pytest_report_header = rapids_pytest_report_header

#     class BenchmarkFixtureOverride(plugin.BenchmarkFixture):
#
#         def __init__(self, benchmarkFixtureInstance):
#             self.__benchmarkFixtureInstance = benchmarkFixtureInstance
#
#         def __getattr__(self, attr):
#             return getattr(self.__benchmarkFixtureInstance, attr)
#
#         def _make_stats(self, iterations):
#             bench_stats = pytest_benchmark.stats.Metadata(
#                 self, iterations=iterations, options={
#                     "disable_gc": self._disable_gc,
#                     "timer": self._timer,
#                     "min_rounds": self._min_rounds,
#                     "max_time": self._max_time,
#                     "min_time": self._min_time,
#                     "something_else": 44,
#                     "warmup": self._warmup,
#                 })
#             self._add_stats(bench_stats)
#             self.stats = bench_stats
#             return bench_stats
#
#     global GPUBenchmarkFixture
#     GPUBenchmarkFixture = BenchmarkFixtureOverride

################################################################################
def pytest_report_header(config):
    #bs = config._benchmarksession
    return ("gpubenchmark: {version} (defaults:"
            " asv_output_dir=."
            ")").format(
        version=__version__
    )

def DISABLED_pytest_terminal_summary(terminalreporter):
    """
    try:
        terminalreporter.config._benchmarksession.display(terminalreporter)
    except PerformanceRegression:
        raise
    except Exception:
        terminalreporter.config._benchmarksession.logger.error("\n%s" % traceback.format_exc())
        raise
    """
    tr = terminalreporter
    if hasattr(tr.config, "_gpubenchmarksession") \
       and tr.config._gpubenchmarksession is not None:
        tr.ensure_newline()

        for b in tr.config._benchmarksession.benchmarks:
            bd = b.as_dict()
            tr.write_line("max GPU mem: %s" % bd["stats"]["gpu_mem"])

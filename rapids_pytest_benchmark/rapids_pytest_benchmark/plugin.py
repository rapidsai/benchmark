from functools import partial
import time
import platform

import pytest
import pytest_benchmark
import asvdb.utils as asvdbUtils
from asvdb import ASVDb, BenchmarkInfo, BenchmarkResult
from pynvml import smi
import psutil

from .gpu_metric_poller import startGpuMetricPolling, stopGpuMetricPolling
from .reporting import GPUTableResults


__version__ = "0.1"


def pytest_addoption(parser):
    group = parser.getgroup("benchmark")
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
    def __init__(self, fixture, iterations, options, fixtureParamNames=None):
        super().__init__(fixture, iterations, options)
        self.stats = GPUStats()
        self.fixture_param_names = fixtureParamNames

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

    def __init__(self, benchmarkFixtureInstance, fixtureParamNames=None):
        self.__benchmarkFixtureInstance = benchmarkFixtureInstance
        self.fixture_param_names = fixtureParamNames
        self.__timeOnlyRunner = None

    def __getattr__(self, attr):
        return getattr(self.__benchmarkFixtureInstance, attr)

    def _make_runner(self, function_to_benchmark, args, kwargs):
        # The benchmark runner returned from BenchmarkFixture._make_runner() is
        # a function that runs the function_to_benchmark loops_range times in a
        # loop and returns total timed duration.
        timeBenchmarkRunner = super()._make_runner(function_to_benchmark,
                                                   args, kwargs)
        self.__timeOnlyRunner = timeBenchmarkRunner

        def runner(loops_range, timer=self._timer):
            # GPU polling increases runtime for code that uses the GPU, so
            # perform runs to measure runtime separately.
            # runtime measurement
            timeRetVal = timeBenchmarkRunner(loops_range, timer)

            # GPU measurements
            gpuPollObj = startGpuMetricPolling()
            time.sleep(0.1)
            try:
                startTime = time.time()
                (_, result) = timeBenchmarkRunner(loops_range=0,
                                                  timer=timer)
                duration = time.time() - startTime
                # guarantee a minimum time has passed to ensure GPU metrics have
                # been taken
                if duration < 0.1:
                    time.sleep(0.1)

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
        bench_stats = GPUMetadata(self,
                                  iterations=iterations,
                                  options={
                                      "disable_gc": self._disable_gc,
                                      "timer": self._timer,
                                      "min_rounds": self._min_rounds,
                                      "max_time": self._max_time,
                                      "min_time": self._min_time,
                                      "warmup": self._warmup,
                                  },
                                  fixtureParamNames=self.fixture_param_names)
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


# def pytest_configure(config):
#     return


@pytest.fixture(scope="function")
def gpubenchmark(request, benchmark):
    # FIXME: if ASV output is enabled, enforce that fixture_param_names are set.
    # FIXME: if no params, do not enforce fixture_param_names check
    return GPUBenchmarkFixture(benchmark, request.node.keywords.get("fixture_param_names"))


################################################################################
def pytest_sessionstart(session):
    session.config._benchmarksession_orig = session.config._benchmarksession
    session.config._gpubenchmarksession = \
        GPUBenchmarkSession(session.config._benchmarksession)
    session.config._benchmarksession = session.config._gpubenchmarksession


def _ensureListLike(item):
    """
    Return the item if it is a list or tuple, otherwise add it to a list and
    return that.
    """
    return item if (isinstance(item, list) or isinstance(item, tuple)) \
                else [item]


def _getHierNameFromFullname(benchFullname):
    """
    Turn a bench name that potentiall looks like this:
       'foodir/bench_algos.py::BenchStuff::bench_bfs[1-2-False-True]'
    into this:
       'foodir.bench_algos.BenchStuff.bench_bfs'
    """
    benchFullname = benchFullname.partition("[")[0]  # strip any params
    (modName, _, benchName) = benchFullname.partition("::")

    if modName.endswith(".py"):
        modName = modName.partition(".")[0]

    modName = modName.replace("/", ".")
    benchName = benchName.replace("::", ".")

    return "%s.%s" % (modName, benchName)


def pytest_sessionfinish(session, exitstatus):
    if exitstatus != 0:
        return

    gpuBenchSess = session.config._gpubenchmarksession
    config = session.config
    asv_output_dir = config.getoption("benchmark_asv_output_dir", None)

    if asv_output_dir and gpuBenchSess.benchmarks:

        (commitHash, commitTime) = asvdbUtils.getCommitInfo()
        (repo, branch) = asvdbUtils.getRepoInfo()
        db = ASVDb(asv_output_dir, repo, [branch])

        smi.nvmlInit()
        # FIXME: get actual device number!
        gpuDeviceHandle = smi.nvmlDeviceGetHandleByIndex(0)
        uname = platform.uname()
        machineName = uname.machine
        cpuType = uname.processor
        arch = uname.machine
        pythonVer=platform.python_version()
        cudaVer = "unknown"
        osType = platform.linux_distribution()[0]
        gpuType = smi.nvmlDeviceGetName(gpuDeviceHandle).decode()
        ram = "%d" % psutil.virtual_memory().total

        suffixDict = dict(gpu_util="gpuutil",
                          gpu_mem="gpumem",
                          mean="time",
        )
        unitsDict = dict(gpu_util="percent",
                         gpu_mem="bytes",
                         mean="seconds",
        )

        bInfo = BenchmarkInfo(machineName=machineName,
                              cudaVer=cudaVer,
                              osType=osType,
                              pythonVer=pythonVer,
                              commitHash=commitHash,
                              commitTime=commitTime,
                              gpuType=gpuType,
                              cpuType=cpuType,
                              arch=arch,
                              ram=ram)

        for bench in gpuBenchSess.benchmarks:
            benchName = _getHierNameFromFullname(bench.fullname)
            # build the final params dict by extracting them from the
            # bench.params dictionary
            params = {}
            for (paramName, paramVal) in bench.params.items():
                # If the params are coming from a fixture, handle them
                # differently since they will (should be) stored in a special
                # variable accessible with the name of the fixture.
                #
                # NOTE: "fixture_param_names" must be manually set by the
                # benchmark author/user using the "request" fixture! (see below)
                #
                # @pytest.fixture(params=[1,2,3])
                # def someFixture(request):
                #     request.keywords["fixture_param_names"] = ["the_param_name"]
                if hasattr(bench, "fixture_param_names") and \
                   (bench.fixture_param_names is not None) and \
                   (paramName in bench.fixture_param_names):
                    fixtureName = paramName
                    paramNames = _ensureListLike(bench.fixture_param_names[fixtureName])
                    paramValues = _ensureListLike(paramVal)
                    for (pname, pval) in zip(paramNames, paramValues):
                        params[pname] = pval
                # otherwise, a benchmark/test will have params added to the
                # bench.params dict as a standard key:value (paramName:paramVal)
                else:
                    params[paramName] = paramVal

            bench.stats.mean
            getattr(bench.stats, "gpu_mem", None)
            getattr(bench.stats, "gpu_util", None)

            for statType in ["mean", "gpu_mem", "gpu_util"]:
                bn = "%s_%s" % (benchName, suffixDict[statType])
                val = getattr(bench.stats, statType, None)
                if val is not None:
                    bResult = BenchmarkResult(funcName=bn,
                                              argNameValuePairs=list(params.items()),
                                              result=val)
                    bResult.unit = unitsDict[statType]
                    db.addResult(bInfo, bResult)


def pytest_report_header(config):
    return ("rapids_pytest_benchmark: {version})").format(
        version=__version__
    )


def DISABLED_pytest_benchmark_scale_unit(config, unit, benchmarks, best, worst, sort):
    """
    Scale GPU memory and utilization measurements accordingly
    """
    return


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






########
    # # Replace the pytest-benchmark hook function objects with the wrappers defined here
    # for hc in session.config.pluginmanager.get_hookcallers(pytest_benchmark.plugin):
    #     if hc.name == "pytest_report_header":
    #         #hc.spec.function = rapids_pytest_report_header
    #         for hi in hc.get_hookimpls():
    #             if hi.plugin_name == "benchmark":
    #                 hi.function = rapids_pytest_report_header

    # p = session.config.pluginmanager.get_plugin("benchmark")
    # p = session.config.pluginmanager.unregister(pytest_benchmark.plugin)
    # session.config.pluginmanager.unregister(name="benchmark")
    # #session.config.pluginmanager.register(pytest_benchmark.plugin)
    # p.pytest_report_header = rapids_pytest_report_header
    # session.config.pluginmanager.register(p, name="benchmark")

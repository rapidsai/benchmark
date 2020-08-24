from functools import partial
import time
import platform
import ctypes
import argparse
import subprocess
import json

import pytest
from pytest_benchmark import stats as pytest_benchmark_stats
from pytest_benchmark import utils as pytest_benchmark_utils
from pytest_benchmark import fixture as pytest_benchmark_fixture
from pytest_benchmark import session as pytest_benchmark_session
from pytest_benchmark import compat as pytest_benchmark_compat
import asvdb.utils as asvdbUtils
from asvdb import ASVDb, BenchmarkInfo, BenchmarkResult
from pynvml import smi
import psutil

from . import __version__
from .gpu_metric_poller import startGpuMetricPolling, stopGpuMetricPolling
from .reporting import GPUTableResults

# FIXME: find a better place to do this and/or a better way
pytest_benchmark_utils.ALLOWED_COLUMNS.append("gpu_mem")
pytest_benchmark_utils.ALLOWED_COLUMNS.append("gpu_util")
pytest_benchmark_utils.ALLOWED_COLUMNS.append("gpu_rounds")


def pytest_addoption(parser):
    group = parser.getgroup("benchmark")
    # FIXME: add check for valid dir, similar to "parse_save()" in
    # pytest-benchmark

    # FIXME: when multi-GPU supported, update the help to mention that the user
    # can specify this option multiple times to observe multiple GPUs? This is
    # why action=append.
    group.addoption(
        "--benchmark-gpu-device",
        metavar="GPU_DEVICENO", default=[0], type=_parseSaveGPUDeviceNum,
        help="GPU device number to observe for GPU metrics."
    )
    group.addoption(
        "--benchmark-gpu-max-rounds", type=_parseGpuMaxRounds,
        help="Maximum number of rounds to run the test/benchmark during the "
        "GPU measurement phase. If not provided, will run the same number of "
        "rounds performed for the runtime measurement."
    )
    group.addoption(
        "--benchmark-gpu-disable", action="store_true", default=False,
        help="Do not perform GPU measurements when using the gpubenchmark "
        "fixture, only perform other enabled measurements."
    )
    group.addoption(
        "--benchmark-custom-metrics-disable", action="store_true", default=False,
        help="Do not perform custom metrics measurements when using the "
        "gpubenchmark fixture, only perform other enabled measurements."
    )
    group.addoption(
        "--benchmark-asv-output-dir",
        metavar="ASV_DB_DIR", default=None,
        help='ASV "database" directory to update with benchmark results.'
    )
    group.addoption(
        "--benchmark-asv-metadata",
        metavar="ASV_DB_METADATA",
        default={}, type=_parseSaveMetadata,
        help='Metadata to be included in the ASV report in JSON format. For example: '
        '{"machineName":"my_machine2000", "gpuType":"FastGPU3", "arch":"x86_64"}. If not '
        'provided, best-guess values will be derived from the environment. '
        'Valid metadata is: "machineName", "cudaVer", "osType", "pythonVer", '
        '"commitRepo", "commitBranch", "commitHash", "commitTime", "gpuType", '
        '"cpuType", "arch", "ram", "gpuRam", "requirements"'
    )


def _parseGpuMaxRounds(stringOpt):
    """
    Ensures opt passed is a number > 0
    """
    if not stringOpt:
        raise argparse.ArgumentTypeError("Cannot be empty")
    if stringOpt.isdecimal():
        num = int(stringOpt)
        if num == 0:
            raise argparse.ArgumentTypeError("Must be non-zero")
    else:
        raise argparse.ArgumentTypeError("Must be an int > 0")
    return num


def _parseSaveGPUDeviceNum(stringOpt):
    """
    Given a string like "0,1, 2" return [0, 1, 2]
    """
    if not stringOpt:
        raise argparse.ArgumentTypeError("Cannot be empty")
    retList = []

    for i in stringOpt.split(","):
        try:
            num = int(i.strip())
        except ValueError:
            raise argparse.ArgumentTypeError(f"must specify an int, got {i}")
        # FIXME: also check that this is a valid GPU device
        if num not in retList:
            retList.append(num)

    return retList


def _parseSaveMetadata(stringOpt):
    """
    Convert JSON input to Python dictionary
    """
    if not stringOpt:
        raise argparse.ArgumentTypeError("Cannot be empty")

    validVars = ["machineName", "cudaVer", "osType", "pythonVer",
                 "commitRepo", "commitBranch", "commitHash", "commitTime",
                 "gpuType", "cpuType", "arch", "ram", "gpuRam", "requirements"]

    retDict = json.loads(stringOpt)

    for key in retDict.keys():
        if key not in validVars:
            raise argparse.ArgumentTypeError(f'invalid metadata var: "{var}"')

    return retDict


class GPUBenchmarkResults:
    def __init__(self, gpuMem, gpuUtil):
        self.gpuMem = gpuMem
        self.gpuUtil = gpuUtil


class GPUMetadata(pytest_benchmark_stats.Metadata):
    def __init__(self, fixture, iterations, options, fixtureParamNames=None):
        super().__init__(fixture, iterations, options)
        # Use an overridden Stats object that also handles GPU metrics
        self.stats = GPUStats()
        # fixture_param_names is used for reporting, see pytest_sessionfinish()
        self.fixture_param_names = fixtureParamNames

    def updateGPUMetrics(self, gpuBenchmarkResults):
        self.stats.updateGPUMetrics(gpuBenchmarkResults)

    def updateCustomMetric(self, result, name, unitString):
        self.stats.updateCustomMetric(result, name, unitString)


class GPUStats(pytest_benchmark_stats.Stats):
    fields = (
        "min", "max", "mean", "stddev", "rounds", "gpu_rounds", "median", "gpu_mem", "gpu_util", "iqr", "q1", "q3", "iqr_outliers", "stddev_outliers",
        "outliers", "ld15iqr", "hd15iqr", "ops", "total"
    )

    def __init__(self):
        super().__init__()
        self.gpuData = []
        self.__customMetrics = {}


    def updateGPUMetrics(self, gpuBenchmarkResults):
        self.gpuData.append((gpuBenchmarkResults.gpuMem,
                             gpuBenchmarkResults.gpuUtil))


    def updateCustomMetric(self, result, name, unitString):
        self.__customMetrics[name] = (result, unitString)


    def getCustomMetricNames(self):
        return list(self.__customMetrics.keys())


    def getCustomMetric(self, name):
        return self.__customMetrics[name]


    # FIXME: this may not need to be here
    def as_dict(self):
        return super().as_dict()

    @pytest_benchmark_utils.cached_property
    def gpu_rounds(self):
        return len(self.gpuData)

    # Only the max values are available for GPU metrics, since the method with
    # which they're obtained (polling the GPU) lends itself to missing spikes,
    # and a min or average really isn't meaningful. The max returned here is
    # really a lower bound, to be read as "the [mem|gpu] usage was *at least* x"
    @pytest_benchmark_utils.cached_property
    def gpu_mem(self):
        return max([i[0] for i in self.gpuData])

    @pytest_benchmark_utils.cached_property
    def gpu_util(self):
        return max([i[1] for i in self.gpuData])


class GPUBenchmarkFixture(pytest_benchmark_fixture.BenchmarkFixture):

    def __init__(self, benchmarkFixtureInstance, fixtureParamNames=None,
                 gpuDeviceNums=None, gpuMaxRounds=None, gpuDisable=False,
                 customMetricsDisable=False):
        self.__benchmarkFixtureInstance = benchmarkFixtureInstance
        self.fixture_param_names = fixtureParamNames
        self.gpuDeviceNums = gpuDeviceNums or [0]
        self.gpuMaxRounds = gpuMaxRounds
        self.gpuDisable = gpuDisable
        self.customMetricsDisable = customMetricsDisable
        self.__timeOnlyRunner = None
        self.__customMetricsDict = {}


    def __getattr__(self, attr):
        """
        Any method or member that is not defined in this class will fall
        through and be accessed on self.__benchmarkFixtureInstance.  This
        allows this class to override anything on the previously-instantiated
        self.__benchmarkFixtureInstance without changing the code that
        instantiated it in pytest-benchmark.
        """
        return getattr(self.__benchmarkFixtureInstance, attr)


    def _make_gpu_runner(self, function_to_benchmark, args, kwargs):
        """
        Create a callable that will run the function_to_benchmark with the
        provided args and kwargs, and wrap it in calls to perform GPU
        measurements. The resulting callable will return a GPUBenchmarkResults
        obj containing the measurements.
        """
        def runner():
            gpuPollObj = startGpuMetricPolling(self.gpuDeviceNums)
            time.sleep(0.1)  # Helps ensure the polling process has started
            try:
                startTime = time.time()
                function_to_benchmark(*args, **kwargs)
                duration = time.time() - startTime
                # Guarantee a minimum time has passed to ensure GPU metrics
                # have been taken
                if duration < 0.1:
                    time.sleep(0.1)

            finally:
                stopGpuMetricPolling(gpuPollObj)

            return GPUBenchmarkResults(gpuMem=gpuPollObj.maxGpuMemUsed,
                                       gpuUtil=gpuPollObj.maxGpuUtil)
        return runner


    def _run_gpu_measurements(self, function_to_benchmark, args, kwargs):
        """
        Run as part of _raw() or _raw_pedantic() to perform GPU measurements.
        This only runs if benchmarks and gpu benchmarks are enabled.
        """
        if self.enabled and not(self.gpuDisable):
            gpuRunner = self._make_gpu_runner(function_to_benchmark, args, kwargs)

            # Get the number of rounds performed from the runtime measurement
            rounds = self.stats.stats.rounds
            assert rounds > 0  # FIXME: do we need this?

            if self.gpuMaxRounds is not None:
                rounds = min(rounds, self.gpuMaxRounds)

            for _ in pytest_benchmark_compat.XRANGE(rounds):
                self.stats.updateGPUMetrics(gpuRunner())

        # Set the "mode" (regular or pedantic) here rather than override another
        # method. This is needed since cleanup callbacks registered prior to the
        # class override dont see the new value and will print a warning saying
        # a benchmark was run without using a benchmark fixture. The warning is
        # printed based on if mode was ever set or not.
        self.__benchmarkFixtureInstance._mode = self._mode


    def _run_custom_measurements(self, function_result):
        if self.enabled and not(self.customMetricsDisable):
            for (metic_name, (metric_callable, metric_unit_string)) in \
                self.__customMetricsDict.items():
                self.stats.updateCustomMetric(
                    metric_callable(function_result),
                    metic_name, metric_unit_string)


    def _raw(self, function_to_benchmark, *args, **kwargs):
        """
        Run the time measurement as defined in pytest-benchmark, then run GPU
        metrics separately. Running separately ensures GPU monitoring does not
        affect runtime perf.
        """
        function_result = super()._raw(function_to_benchmark, *args, **kwargs)
        self._run_gpu_measurements(function_to_benchmark, args, kwargs)
        self._run_custom_measurements(function_result)
        return function_result


    def _raw_pedantic(self, target, args=(), kwargs=None, setup=None, rounds=1,
                      warmup_rounds=0, iterations=1):
        """
        Run the pedantic time measurement as defined in pytest-benchmark, then
        run GPU metrics separately. Running separately ensures GPU monitoring
        does not affect runtime perf.
        """
        result = super()._raw_pedantic(target, args, kwargs, setup, rounds,
                                       warmup_rounds, iterations)
        self._run_gpu_measurements(function_to_benchmark, args, kwargs)
        self._run_custom_measurements(function_result)
        return result


    def _make_stats(self, iterations):
        """
        Overridden method to create a stats object that can be used as-is by
        pytest-benchmark but also accepts GPU metrics.
        """
        if self.gpuDisable:
            return super()._make_stats(iterations)

        bench_stats = GPUMetadata(self,
                                  iterations=iterations,
                                  options={
                                      "disable_gc": self._disable_gc,
                                      "timer": self._timer,
                                      "min_rounds": self._min_rounds,
                                      "max_time": self._max_time,
                                      "min_time": self._min_time,
                                      "warmup": self._warmup,
                                      "gpu_max_rounds": self.gpuMaxRounds,
                                  },
                                  fixtureParamNames=self.fixture_param_names)
        self._add_stats(bench_stats)
        self.stats = bench_stats
        return bench_stats


    def addMetric(self, metric_callable, metric_name, metric_unit_string):
        """
        Adds a custom metric to the set of metrics gathered as part of this
        benchmark run.  When the benchmark is run, the metric_callable will also
        be run and the return value will be stored under the name metric_name.
        """
        self.__customMetricsDict[metric_name] = (metric_callable,
                                                 metric_unit_string)


class GPUBenchmarkSession(pytest_benchmark_session.BenchmarkSession):
    compared_mapping = None
    groups = None

    def __init__(self, benchmarkSession):
        self.__benchmarkSessionInstance = benchmarkSession
        self.compared_mapping = benchmarkSession.compared_mapping
        self.groups = benchmarkSession.groups

        # Add the GPU columns to the original list in the appropriate order
        # FIXME: this always adds gpu_* columns, even if the user specified a
        # list of columns that didn't include those.  This is because the
        # default list of columns is hardcoded in the pytest-benchmark option
        # parsing and cannot be overridden without changing pytest-benchmark (I
        # think?)
        origColumns = self.columns
        self.columns = []
        for c in ["min", "max", "mean", "stddev", "median", "iqr", "outliers", "ops", "gpu_mem", "rounds", "gpu_rounds", "iterations"]:
            # Always add gpu_mem (for now), and only add gpu_rounds if rounds was requested.
            if (c in origColumns) or \
               (c == "gpu_mem") or \
               ((c == "gpu_rounds") and ("rounds" in origColumns)):
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


@pytest.fixture(scope="function")
def gpubenchmark(request, benchmark):
    # FIXME: if ASV output is enabled, enforce that fixture_param_names are set.
    # FIXME: if no params, do not enforce fixture_param_names check
    gpuDeviceNums = request.config.getoption("benchmark_gpu_device")
    gpuMaxRounds = request.config.getoption("benchmark_gpu_max_rounds")
    gpuDisable = request.config.getoption("benchmark_gpu_disable")
    customMetricsDisable = request.config.getoption("benchmark_custom_metrics_disable")
    return GPUBenchmarkFixture(
        benchmark,
        fixtureParamNames=request.node.keywords.get("fixture_param_names"),
        gpuDeviceNums=gpuDeviceNums,
        gpuMaxRounds=gpuMaxRounds,
        gpuDisable=gpuDisable,
        customMetricsDisable=customMetricsDisable)


################################################################################
def pytest_sessionstart(session):
    session.config._benchmarksession_orig = session.config._benchmarksession
    session.config._gpubenchmarksession = \
        GPUBenchmarkSession(session.config._benchmarksession)
    session.config._benchmarksession = session.config._gpubenchmarksession


def _getOSName():
    try :
        binout = subprocess.check_output(
            ["bash", "-c",
             "source /etc/os-release && echo -n ${ID}-${VERSION_ID}"])
        return binout.decode()

    except subprocess.CalledProcessError:
        return None


def _getCudaVersion():
    """
    Get the CUDA version from the CUDA DLL/.so if possible, otherwise return
    None. (NOTE: is this better than screen scraping nvidia-smi?)
    """
    try :
        lib = ctypes.CDLL("libcudart.so")
        function = getattr(lib,"cudaRuntimeGetVersion")
        result = ctypes.c_int()
        resultPtr = ctypes.pointer(result)
        function(resultPtr)
        # The version is returned as (1000 major + 10 minor). For example, CUDA
        # 9.2 would be represented by 9020
        major = int(result.value / 1000)
        minor = int((result.value - (major * 1000)) / 10)
        return f"{major}.{minor}"
    # FIXME: do not use a catch-all handler
    except:
        return None


def _ensureListLike(item):
    """
    Return the item if it is a list or tuple, otherwise add it to a list and
    return that.
    """
    return item if (isinstance(item, list) or isinstance(item, tuple)) \
                else [item]


def _getHierBenchNameFromFullname(benchFullname):
    """
    Turn a bench name that potentially looks like this:
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
    gpuBenchSess = session.config._gpubenchmarksession
    config = session.config
    asvOutputDir = config.getoption("benchmark_asv_output_dir")
    asvMetadata = config.getoption("benchmark_asv_metadata")
    gpuDeviceNums = config.getoption("benchmark_gpu_device")

    if asvOutputDir and gpuBenchSess.benchmarks:

        # FIXME: do not lookup commit metadata if already specified on the
        # command line.
        (commitHash, commitTime) = asvdbUtils.getCommitInfo()
        (commitRepo, commitBranch) = asvdbUtils.getRepoInfo()

        # FIXME: do not make pynvml calls if all the metadata provided by pynvml
        # was specified on the command line.
        smi.nvmlInit()
        # only supporting 1 GPU
        gpuDeviceHandle = smi.nvmlDeviceGetHandleByIndex(gpuDeviceNums[0])

        uname = platform.uname()
        machineName = asvMetadata.get("machineName", uname.machine)
        cpuType = asvMetadata.get("cpuType", uname.processor)
        arch = asvMetadata.get("arch", uname.machine)
        pythonVer = asvMetadata.get("pythonVer",
            ".".join(platform.python_version_tuple()[:-1]))
        cudaVer = asvMetadata.get("cudaVer", _getCudaVersion() or "unknown")
        osType = asvMetadata.get("osType",
            _getOSName() or platform.linux_distribution()[0])
        gpuType = asvMetadata.get("gpuType",
            smi.nvmlDeviceGetName(gpuDeviceHandle).decode())
        ram = asvMetadata.get("ram", "%d" % psutil.virtual_memory().total)
        gpuRam = asvMetadata.get("gpuRam",
            "%d" % smi.nvmlDeviceGetMemoryInfo(gpuDeviceHandle).total)

        commitHash = asvMetadata.get("commitHash", commitHash)
        commitTime = asvMetadata.get("commitTime", commitTime)
        commitRepo = asvMetadata.get("commitRepo", commitRepo)
        commitBranch = asvMetadata.get("commitBranch", commitBranch)
        requirements = asvMetadata.get("requirements", "{}")

        suffixDict = dict(gpu_util="gpuutil",
                          gpu_mem="gpumem",
                          mean="time",
        )
        unitsDict = dict(gpu_util="percent",
                         gpu_mem="bytes",
                         mean="seconds",
        )

        db = ASVDb(asvOutputDir, commitRepo, [commitBranch])

        bInfo = BenchmarkInfo(machineName=machineName,
                              cudaVer=cudaVer,
                              osType=osType,
                              pythonVer=pythonVer,
                              commitHash=commitHash,
                              commitTime=commitTime,
                              branch=commitBranch,
                              gpuType=gpuType,
                              cpuType=cpuType,
                              arch=arch,
                              ram=ram,
                              gpuRam=gpuRam,
                              requirements=requirements)

        for bench in gpuBenchSess.benchmarks:
            benchName = _getHierBenchNameFromFullname(bench.fullname)
            # build the final params dict by extracting them from the
            # bench.params dictionary. Not all benchmarks are parameterized
            params = {}
            bench_params = bench.params.items() if bench.params is not None else []
            for (paramName, paramVal) in bench_params:
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

            resultList = []
            for statType in ["mean", "gpu_mem", "gpu_util"]:
                bn = "%s_%s" % (benchName, suffixDict[statType])
                val = getattr(bench.stats, statType, None)
                if val is not None:
                    bResult = BenchmarkResult(funcName=bn,
                                              argNameValuePairs=list(params.items()),
                                              result=val)
                    bResult.unit = unitsDict[statType]
                    resultList.append(bResult)

            # If there were any custom metrics, add each of those as well as an
            # individual result to the same bInfo isntance.
            for customMetricName in bench.stats.getCustomMetricNames():
                (result, unitString) = bench.stats.getCustomMetric(customMetricName)
                bn = "%s_%s" % (benchName, customMetricName)
                bResult = BenchmarkResult(funcName=bn,
                                          argNameValuePairs=list(params.items()),
                                          result=result)
                bResult.unit = unitString
                resultList.append(bResult)

            db.addResults(bInfo, resultList)


def pytest_report_header(config):
    return ("rapids_pytest_benchmark: {version}").format(
        version=__version__
    )


def DISABLED_pytest_benchmark_scale_unit(config, unit, benchmarks, best, worst, sort):
    """
    Scale GPU memory and utilization measurements accordingly
    """
    return

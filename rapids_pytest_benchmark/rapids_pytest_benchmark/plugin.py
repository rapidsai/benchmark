from functools import partial
import time
import platform
import ctypes
import argparse
import subprocess

import pytest
from pytest_benchmark import stats as pytest_benchmark_stats
from pytest_benchmark import utils as pytest_benchmark_utils
from pytest_benchmark import fixture as pytest_benchmark_fixture
from pytest_benchmark import session as pytest_benchmark_session
import asvdb.utils as asvdbUtils
from asvdb import ASVDb, BenchmarkInfo, BenchmarkResult
from pynvml import smi
import psutil

from .gpu_metric_poller import startGpuMetricPolling, stopGpuMetricPolling
from .reporting import GPUTableResults

# FIXME: find a better place to do this and/or a better way
pytest_benchmark_utils.ALLOWED_COLUMNS.append("gpu_mem")
pytest_benchmark_utils.ALLOWED_COLUMNS.append("gpu_util")


# FIXME: single-source this to stay syncd with the version in the packaging code
__version__ = "0.0.7"


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
        "--benchmark-asv-output-dir",
        metavar="ASV_DB_DIR", default=None,
        help='ASV "database" directory to update with benchmark results.'
    )
    group.addoption(
        "--benchmark-asv-metadata",
        metavar="ASV_DB_METADATA",
        default={}, type=_parseSaveMetadata,
        help='Metadata to be included in the ASV report. For example: '
        '"machineName=my_machine2000, gpuType=FastGPU3, arch=x86_64". If not '
        'provided, best-guess values will be derived from the environment. '
        'Valid metadata is: "machineName", "cudaVer", "osType", "pythonVer", '
        '"commitRepo", "commitBranch", "commitHash", "commitTime", "gpuType", '
        '"cpuType", "arch", "ram", "gpuRam"'
    )


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
    Given a string like "foo=bar, baz=44" return {'foo':'bar', 'baz':44}
    """
    if not stringOpt:
        raise argparse.ArgumentTypeError("Cannot be empty")

    validVars = ["machineName", "cudaVer", "osType", "pythonVer",
                 "commitRepo", "commitBranch", "commitHash", "commitTime",
                 "gpuType", "cpuType", "arch", "ram", "gpuRam"]
    retDict = {}

    for pair in stringOpt.split(","):
        (var, value) = [i.strip() for i in pair.split("=")]
        if var in validVars:
            retDict[var.strip()] = str(value.strip())
        else:
            raise argparse.ArgumentTypeError(f'invalid metadata var: "{var}"')

    return retDict


class GPUBenchmarkResults:
    def __init__(self, runtime, gpuMem, gpuUtil):
        self.runtime = runtime
        self.gpuMem = gpuMem
        self.gpuUtil = gpuUtil


class GPUMetadata(pytest_benchmark_stats.Metadata):
    def __init__(self, fixture, iterations, options, fixtureParamNames=None):
        super().__init__(fixture, iterations, options)
        self.stats = GPUStats()
        self.fixture_param_names = fixtureParamNames

    def update(self, gpuBenchmarkResults):
        # Assume GPU metrics do not accumulate over multiple runs like runtime does?
        self.stats.update(gpuBenchmarkResults.runtime / self.iterations,
                          gpuBenchmarkResults.gpuMem,
                          gpuBenchmarkResults.gpuUtil)


class GPUStats(pytest_benchmark_stats.Stats):
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

    @pytest_benchmark_utils.cached_property
    def gpu_mem(self):
        #for i in self.gpuData: print(i)
        return max([i[0] for i in self.gpuData])
    @pytest_benchmark_utils.cached_property
    def gpu_util(self):
        return max([i[1] for i in self.gpuData])


class GPUBenchmarkFixture(pytest_benchmark_fixture.BenchmarkFixture):

    def __init__(self, benchmarkFixtureInstance, fixtureParamNames=None,
                 gpuDeviceNums=None):
        self.__benchmarkFixtureInstance = benchmarkFixtureInstance
        self.fixture_param_names = fixtureParamNames
        self.gpuDeviceNums = gpuDeviceNums or [0]
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
            gpuPollObj = startGpuMetricPolling(self.gpuDeviceNums)
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

        # Set the "mode" (regular or pedantic) here rather than override another
        # method. This is needed since cleanup callbacks registered prior to the
        # class override dont see the new value and will print a warning saying
        # a benchmark was run without using a benchmark fixture. The warning is
        # printed based on if "mode" was ever set or not.
        self.__benchmarkFixtureInstance._mode = self._mode

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


class GPUBenchmarkSession(pytest_benchmark_session.BenchmarkSession):
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


@pytest.fixture(scope="function")
def gpubenchmark(request, benchmark):
    # FIXME: if ASV output is enabled, enforce that fixture_param_names are set.
    # FIXME: if no params, do not enforce fixture_param_names check
    gpuDeviceNums = request.config.getoption("benchmark_gpu_device")
    return GPUBenchmarkFixture(benchmark,
        fixtureParamNames=request.node.keywords.get("fixture_param_names"),
        gpuDeviceNums=gpuDeviceNums)


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
    Get the CUDA version from the DLL if possible, otherwise return None.
    (is this better than screen scraping nvidia-smi?)
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
                              gpuType=gpuType,
                              cpuType=cpuType,
                              arch=arch,
                              ram=ram,
                              gpuRam=gpuRam)

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
    return ("rapids_pytest_benchmark: {version}").format(
        version=__version__
    )


def DISABLED_pytest_benchmark_scale_unit(config, unit, benchmarks, best, worst, sort):
    """
    Scale GPU memory and utilization measurements accordingly
    """
    return

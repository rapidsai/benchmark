# Maintainer's guide

In no particular order, this covers implementation details and common maintenance tasks for the benchmarking tools. To request more information about a topic, please update this document with text that includes a `FIXME` label and a description of the the topic/information to be added, and a PR will be submitted with the new information.


## Extending rapids-pytest-benchmark
- This is covered in detail in the [`rapids-pytest-benchmark` README](rapids_pytest_benchmark/README.md).


## Proposal for how to integrate RMM logging for more accurate GPU mem usage (and leak detection!)
- The latest version of RMM has an API to enable logging of memory allocs and frees to a file.
- An issue with more details is [here](https://github.com/rapidsai/benchmark/issues/27)


## Working CB design doc
- There is a diagram proposing potential CB implementation details [here](https://docs.google.com/drawings/d/1LBxqMlJM0DObfSjnK-CK8c-MxAdiFGQOXl9tId3Daaw).
  - If S3 is used for storage, a separate diagram proposing potential implementation details is [here](https://docs.google.com/drawings/d/1Cd1QDry1THKmzHpHI8jQETsSVdpncG22xqy72yUWIR0)


## asvdb design decisions
- `[asvdb`](https://github.com/rapidsai/asvdb) is a Python module (and CLI tool) that encapsulates the implementation details of how ASV stores results, allowing other tools in a benchmarking workflow to take advantage of ASV reporting without having to implement their own ASV-compatible data generation.
- One approach looked into for this was to pull out the individual classes in `asv` responsible for reading and writing the ASV "database" and use those directly.
  - This was initially rejected because:
    - It would either require a copy-and-past fork of specific classes which are not optimized for this use case, or an added dependency on the entire `asv` package.
    - Since `asvdb` started out as a simple abstraction for a single operation (write results to disk for ASV to read), the added work of pulling in classes from another package didn't seem necessary for some simple JSON dumps.
  - **However**, now that `asvdb` has grown in scope, using classes from `asv` might be worth revisiting:
    - The classes to use would be:
      - `asv/config.py` - the ASV configuration (`asv.conf.json`)
      - `asv/benchmarks.py` - meta-data about the benchmarks being run (`results/benchmarks.json`)
      - `asv/machine.py` - meta-data about the benchmark machine (`<machine name>/machine.json`)
      - `asv/results.py` - all results for a single machine and commit hash
    - Using the classes from the `asv` project could also facilitate including `asvdb` as a utility in the `asv` project itself, which could greatly simplify things (community maintainers, free upgrades when ASV updates internals, etc.)

![asvdb-based benchmarking design diagram](https://docs.google.com/drawings/d/e/2PACX-1vRIxIV02BWh5tbJkF1fL368m0JvepZKqcD0oxYQNIQesgda1qsFo_zlmygRh5unfFOWwsTYGaYgUzmA/pub?w=960&h=720)


## Proposal for including automated notebook benchmarks
- Some teams still prefer to use notebooks for E2E benchmarks. Notebooks are nice because they're easily shared with the community, marketing, and customers, and help to highlight the performance advantages of RAPIDS in ways other benchmarks don't (they display images, real results, highlight our APIs, show plots, etc.)
- We can add notebooks to our benchmark runs as "just another source of benchmark data" by doing the following:
  - Create a new magic specifically for benchmarking
    - This magic would essentially do what our `gpubenchmark` fixture and other `rapids-pytest-benchmark` features do for python-based benchmarks (gather time and GPU metrics, push results to an ASV database)
  - Have the magic used in the `nbtest.sh` script
    - At the moment, the script ignores all magics since many are not compatible with a scripted run, but the `gpubenchmark` magic could be
  - From there, `asvdb`, ASV, and any other consumer of benchmarks results wouldn't know or care that the result didn't come from python or gbench.
  - This was demo'd here: https://nvmeet.webex.com/webappng/sites/nvmeet/recording/playback/ba0ab73c4a364959b0be6de41a40d289


## ops-utils repo tools
- For convenience, the benchmark jobs use tools in the [`ops-utils` repo](https://github.com/rapidsai/ops-utils/tree/master/benchmark), in particular the `updateJenkinsReport.py` script for creating the nightly overview report.
- Another useful script in this repo is one which can return the exact nightly conda package that was used with a particular commit, called [getNearestCondaPackages.py](https://github.com/rapidsai/ops-utils/blob/master/benchmark/getNearestCondaPackages.py)


## Jenkins jobs overview
- Currently defined [here](http://10.33.227.188/job/wip/job/benchmark-pipeline)
- Currently consists of:
  1) A job to backup the current ASV results dirs ("databases"). The job keeps the last 20 database updates backed up (which should represent 20 days worth if the benchmark pipeline ran once-per-day everyday). After 20, the job removes the oldest one before adding the newest one. This job is defined [here](http://10.33.227.188/job/wip/job/backup-benchmark-results)
  2) The individual benchmark jobs for the different repos. These currently run in parallel on a specific machine. Note that it's important to run on the same machine each time if possible, since different configurations can potentially invaludate benchmark results! Sometimes even different non-GPU aspects of a machine can come into play if code being benchmarked includes some CPU-based computations. In particaulr, some benchmarks need to compare results to baseline implementations that are CPU-based, so benchmarks like these definitely need to run in a CPU-controlled environment. The most up-to-date benchmark job is cuGraph's, which is defined [here](http://10.33.227.188/job/wip/job/cugraph-e2e-benchmarks)
     * As part of the cuGraph job, a custom HTML report is written to provide an easy way to see if a benchmark run failed (since the ASV report itself won't show that), as well as an easy way to go to either the ASV front end (if the jobs ran without errors to see the latest results), or to the individual error logs (if the jobs failed). cuGraph's custom report is [here](http://10.33.227.188:88/asv/cugraph-e2e)
  3) The publish job, which publishes the new results into an ASV report using the `asv publish` command on each repo's ASV database. This job is defined [here](http://10.33.227.188/job/wip/job/publish-benchmark-results)
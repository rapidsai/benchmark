# Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

import sys
import operator

from pytest_benchmark import table as pytest_benchmark_table
from pytest_benchmark import utils as pytest_benchmark_utils


NUMBER_FMT = pytest_benchmark_table.NUMBER_FMT
ALIGNED_NUMBER_FMT = pytest_benchmark_table.ALIGNED_NUMBER_FMT
INT_NUMBER_FMT = "{0:,d}" if sys.version_info[:2] > (2, 6) else "{0:d}"
ALIGNED_INT_NUMBER_FMT = "{0:>{1},d}{2:<{3}}" if sys.version_info[:2] > (2, 6) else "{0:>{1}d}{2:<{3}}"


class GPUTableResults(pytest_benchmark_table.TableResults):
    def display(self, tr, groups, progress_reporter=pytest_benchmark_utils.report_progress):
        tr.write_line("")
        tr.rewrite("Computing stats ...", black=True, bold=True)
        for line, (group, benchmarks) in progress_reporter(groups, tr, "Computing stats ... group {pos}/{total}"):
            benchmarks = sorted(benchmarks, key=operator.itemgetter(self.sort))
            for bench in benchmarks:
                bench["name"] = self.name_format(bench)

            worst = {}
            best = {}
            solo = len(benchmarks) == 1
            for line, prop in progress_reporter(("min", "max", "mean", "median", "iqr", "stddev", "gpu_mem", "gpu_leaked_mem", "ops"),
                                                tr, "{line}: {value}", line=line):
                # During a compare, current or previous results may not have gpu keys
                if prop not in bench:
                    continue

                if prop == "ops":
                    worst[prop] = min(bench[prop] for _, bench in progress_reporter(
                        benchmarks, tr, "{line} ({pos}/{total})", line=line) if prop in bench)
                    best[prop] = max(bench[prop] for _, bench in progress_reporter(
                        benchmarks, tr, "{line} ({pos}/{total})", line=line) if prop in bench)
                else:
                    worst[prop] = max(bench[prop] for _, bench in progress_reporter(
                        benchmarks, tr, "{line} ({pos}/{total})", line=line) if prop in bench)
                    best[prop] = min(bench[prop] for _, bench in progress_reporter(
                        benchmarks, tr, "{line} ({pos}/{total})", line=line) if prop in bench)

            for line, prop in progress_reporter(("outliers", "rounds", "iterations", "gpu_rounds"), tr, "{line}: {value}", line=line):
                if prop not in bench:
                    continue
                worst[prop] = max(benchmark[prop] for _, benchmark in progress_reporter(
                    benchmarks, tr, "{line} ({pos}/{total})", line=line))

            unit, adjustment = self.scale_unit(unit='seconds', benchmarks=benchmarks, best=best, worst=worst,
                                               sort=self.sort)
            ops_unit, ops_adjustment = self.scale_unit(unit='operations', benchmarks=benchmarks, best=best, worst=worst,
                                                       sort=self.sort)
            labels = {
                "name": "Name (time in {0}s, mem in bytes)".format(unit),
                "min": "Min",
                "max": "Max",
                "mean": "Mean",
                "stddev": "StdDev",
                "gpu_mem": "GPU mem",
                "gpu_leaked_mem": "GPU Leaked mem",
                "rounds": "Rounds",
                "gpu_rounds": "GPU Rounds",
                "iterations": "Iterations",
                "iqr": "IQR",
                "median": "Median",
                "outliers": "Outliers",
                "ops": "OPS ({0}ops/s)".format(ops_unit) if ops_unit else "OPS",
            }
            widths = {
                "name": 3 + max(len(labels["name"]), max(len(benchmark["name"]) for benchmark in benchmarks)),
                "rounds": 2 + max(len(labels["rounds"]), len(str(worst["rounds"]))),
                "iterations": 2 + max(len(labels["iterations"]), len(str(worst["iterations"]))),
                "outliers": 2 + max(len(labels["outliers"]), len(str(worst["outliers"]))),
                "ops": 2 + max(len(labels["ops"]), len(NUMBER_FMT.format(best["ops"] * ops_adjustment))),
            }
            # gpu_rounds may not be present if user passed --benchmark-gpu-disable
            if "gpu_rounds" in worst:
                widths["gpu_rounds"] = 2 + max(len(labels["gpu_rounds"]), len(str(worst["gpu_rounds"])))

            for prop in "min", "max", "mean", "stddev", "median", "iqr":
                widths[prop] = 2 + max(len(labels[prop]), max(
                    len(NUMBER_FMT.format(bench[prop] * adjustment))
                    for bench in benchmarks if prop in bench
                ))
            for prop in ["gpu_mem", "gpu_leaked_mem"]:
                if [b for b in benchmarks if prop in b]:
                    widths[prop] = 2 + max(len(labels[prop]), max(
                        len(INT_NUMBER_FMT.format(bench[prop]))
                        for bench in benchmarks if prop in bench
                    ))

            rpadding = 0 if solo else 10
            labels_line = labels["name"].ljust(widths["name"]) + "".join(
                labels[prop].rjust(widths[prop]) + (
                    " " * rpadding
                    #if prop not in ["outliers", "rounds", "iterations"]
                    if prop not in ["outliers", "iterations"]
                    else ""
                )
                for prop in self.columns if (prop in labels) and (prop in widths)
            )
            tr.rewrite("")
            tr.write_line(
                " benchmark{name}: {count} tests ".format(
                    count=len(benchmarks),
                    name="" if group is None else " {0!r}".format(group),
                ).center(len(labels_line), "-"),
                yellow=True,
            )
            tr.write_line(labels_line)
            tr.write_line("-" * len(labels_line), yellow=True)

            for bench in benchmarks:
                has_error = bench.get("has_error")
                tr.write(bench["name"].ljust(widths["name"]), red=has_error, invert=has_error)
                for prop in self.columns:
                    if not((prop in bench) and (prop in widths) and (prop in bench) and (prop in worst)):
                        continue

                    if prop in ("min", "max", "mean", "stddev", "median", "iqr"):
                        tr.write(
                            ALIGNED_NUMBER_FMT.format(
                                bench[prop] * adjustment,
                                widths[prop],
                                pytest_benchmark_table.compute_baseline_scale(best[prop], bench[prop], rpadding),
                                rpadding
                            ),
                            green=not solo and bench[prop] == best.get(prop),
                            red=not solo and bench[prop] == worst.get(prop),
                            bold=True,
                        )
                    elif prop in ("gpu_mem", "gpu_leaked_mem"):
                        tr.write(
                            ALIGNED_INT_NUMBER_FMT.format(
                                bench[prop],
                                widths[prop],
                                pytest_benchmark_table.compute_baseline_scale(best[prop], bench[prop], rpadding),
                                rpadding
                            ),
                            green=not solo and bench[prop] == best.get(prop),
                            red=not solo and bench[prop] == worst.get(prop),
                            bold=True,
                        )
                    elif prop == "ops":
                        tr.write(
                            ALIGNED_NUMBER_FMT.format(
                                bench[prop] * ops_adjustment,
                                widths[prop],
                                pytest_benchmark_table.compute_baseline_scale(best[prop], bench[prop], rpadding),
                                rpadding
                            ),
                            green=not solo and bench[prop] == best.get(prop),
                            red=not solo and bench[prop] == worst.get(prop),
                            bold=True,
                        )
                    else:
                        tr.write("{0:>{1}}".format(bench[prop], widths[prop]))
                tr.write("\n")
            tr.write_line("-" * len(labels_line), yellow=True)
            tr.write_line("")
            if self.histogram:
                # This import requires additional dependencies. Import it
                # here so reporting that does not use the histogram feature
                # need not install dependencies that will not be used.
                from pytest_benchmark import histogram as pytest_benchmark_histogram

                if len(benchmarks) > 75:
                    self.logger.warn("Group {0!r} has too many benchmarks. Only plotting 50 benchmarks.".format(group))
                    benchmarks = benchmarks[:75]

                output_file = pytest_benchmark_histogram.make_histogram(self.histogram, group, benchmarks, unit, adjustment)

                self.logger.info("Generated histogram: {0}".format(output_file), bold=True)

        tr.write_line("Legend:")
        tr.write_line("  Outliers: 1 Standard Deviation from Mean; "
                      "1.5 IQR (InterQuartile Range) from 1st Quartile and 3rd Quartile.")
        tr.write_line("  OPS: Operations Per Second, computed as 1 / Mean")

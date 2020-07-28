import os
import sys
import json
import subprocess
import argparse
import re
import platform
import psutil

from asvdb import ASVDb, BenchmarkInfo, BenchmarkResult
from pynvml import smi

# USAGE:
#
#   -d : JSON Result Directory
#   -n : Repository Name
#   -t : Target Directory for ASV JSON
#   -b : Branch Name

def build_argparse():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('-d', nargs=1, help='JSON Result Directory')
    parser.add_argument('-n', nargs=1, help='Repository Name')
    parser.add_argument('-t', nargs=1, help='Target Directory for JSON')
    parser.add_argument('-b', nargs=1, help='Branch Name')
    parser.add_argument('-r', nargs=1, help='Requirements metadata in JSON format', default=['{}'])
    return parser


def getCommandOutput(cmd):
    result = subprocess.run(cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            shell=True)
    stdout = result.stdout.decode().strip()
    if result.returncode == 0:
        return stdout

    stderr = result.stderr.decode().strip()
    raise RuntimeError("Problem running '%s' (STDOUT: '%s' STDERR: '%s')"
                       % (cmd, stdout, stderr))


def getSysInfo(requirements):
    # Use Node Label from Jenkins if possible
    label = os.environ.get('ASV_LABEL')
    uname = platform.uname()
    if label == None:
        label = uname.machine

    commitHash = getCommandOutput("git rev-parse HEAD")
    commitTime = getCommandOutput("git log -n1 --pretty=%%ct %s" % commitHash)
    commitTime = str(int(commitTime)*1000)  # ASV wants commit to be in milliseconds
    gpuDeviceNums = [0]
    gpuDeviceHandle = smi.nvmlDeviceGetHandleByIndex(gpuDeviceNums[0])

    bInfo = BenchmarkInfo(
                machineName=label,
                cudaVer=getCommandOutput("nvcc --version | grep release | awk '{print $5}' | tr -d ,"),
                osType="%s %s" % (uname.system, uname.release),
                pythonVer=platform.python_version(),
                commitHash=commitHash,
                commitTime=commitTime,
                gpuType=smi.nvmlDeviceGetName(gpuDeviceHandle).decode(),
                cpuType=uname.processor,
                arch=uname.machine,
                ram="%d" % psutil.virtual_memory().total,
                requirements=requirements
            )

    return bInfo

def genBenchmarkJSON(db, sys_info, fileList, repoName):
    pattern = re.compile(r"([^\/]+)")

    for file in fileList:
        with open(file, 'r') as in_file:
            tests = json.load(in_file)["benchmarks"]

        # Create dictionary reference for number of parameters
        num_params_dict = {}
        for each in tests:
            name_and_params = pattern.findall(each["name"])
            name = repoName + "." + name_and_params[0]
            name = name.replace("<","[").replace(">","]")
            test_params = name_and_params[1:]

            # Get max number of parameters for each benchmark
            if name not in num_params_dict:
                num_params_dict[name] = len(test_params)
            else:
                if len(test_params) > num_params_dict[name]:
                    num_params_dict[name] = len(test_params)

        for each in tests:
            #Get Benchmark Name and Test Parameters
            name_and_params = pattern.findall(each["name"])
            name = repoName + "." + name_and_params[0]
            name = name.replace("<","[").replace(">","]")
            test_params = name_and_params[1:]
            param_values = []

            for idx in range(num_params_dict[name]):
                if idx < len(test_params):
                    param_values.append((f"param{idx}", test_params[idx]))
                else:
                    param_values.append((f"param{idx}", "None"))

            #Get result
            if "real_time" in each:
                bench_result = each["real_time"]
            elif "rms" in each:
                bench_result = each["rms"]

            bResult = BenchmarkResult(funcName=name,
                                      argNameValuePairs=param_values,
                                      result=bench_result)
            
            db.addResult(sys_info, bResult)


def main(args):
    ns = build_argparse().parse_args(args)
    testResultDir = ns.d[0]
    repoName = ns.n[0]
    outputDir = ns.t[0]
    branchName = ns.b[0]
    requirements = json.loads(ns.r[0])

    gbenchFileList = os.listdir(testResultDir)
    repoUrl = getCommandOutput("git remote -v").split("\n")[-1].split()[1]

    db = ASVDb(outputDir, repoUrl, [branchName])

    for each in gbenchFileList:
        if not ".json" in each:
            gbenchFileList.remove(each)

    for i,val in enumerate(gbenchFileList):
        gbenchFileList[i] = f"{testResultDir}/{val}"

    smi.nvmlInit()
    system_info = getSysInfo(requirements)
    genBenchmarkJSON(db, system_info, gbenchFileList, repoName)


if __name__ == '__main__':
    main(sys.argv[1:])

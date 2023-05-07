# Copyright 2023 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Start nodes.
# > bazel run -c opt //examples/python/utils:nodectl -- up
#
# Run this example script.
# > bazel run -c opt //examples/python:test

import argparse
import json

import jax
import jax.numpy as jnp
import numpy as np

import spu.utils.distributed as ppd

parser = argparse.ArgumentParser(description="distributed driver.")
parser.add_argument("-c", "--config", default="examples/python/conf/mal3pc.json")
args = parser.parse_args()

with open(args.config, "r") as file:
    conf = json.load(file)

print(conf["devices"]["SPU"]["config"]["runtime_config"]["protocol"])

ppd.init(conf["nodes"], conf["devices"])


def test(x, y, count):
    result = 0

    def for_loop(i, result):
        return result + (x[i] * y[i])

    result = jax.lax.fori_loop(0, count, for_loop, result)
    return result


def data_from_alice(count):
    np.random.seed(0xDEADBEEF)
    return np.random.randint(100, size=count)


def data_from_bob(count):
    np.random.seed(0xC0FFEE)
    return np.random.randint(100, size=count)


def run_on_spu():
    count = 10
    x = ppd.device("P1")(data_from_alice)(count)
    y = ppd.device("P2")(data_from_bob)(count)

    result = ppd.device("SPU")(test, static_argnums=(2,))(x, y, count)

    result = ppd.get(result)
    print(result)


def run_on_cpu():
    count = 10
    x = jnp.array(data_from_alice(count))
    y = jnp.array(data_from_bob(count))

    result = test(x, y, count)

    print(result)


"""
x = DeviceArray([88, 97, 57, 98, 81, 66, 29, 18, 85, 89], dtype=int32)
y = DeviceArray([89, 32, 68, 70, 39, 37, 22, 15, 93, 13], dtype=int32)
"""

if __name__ == "__main__":
    run_on_spu()
    run_on_cpu()

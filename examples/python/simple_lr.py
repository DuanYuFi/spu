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
# > bazel run -c opt //examples/python:simple_lr
#
# Run in malicious setting.
# > bazel run -c opt //examples/python:simple_lr -- -c examples/python/conf/mal3pc.json

import argparse
import json

import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, jit, vmap
from jax import random

from time import time

import spu.utils.distributed as ppd

parser = argparse.ArgumentParser(description="distributed driver.")
parser.add_argument("-c", "--config", default="examples/python/conf/3pc.json")
args = parser.parse_args()

with open(args.config, "r") as file:
    conf = json.load(file)

print(conf["devices"]["SPU"]["config"]["runtime_config"]["protocol"])

ppd.init(conf["nodes"], conf["devices"])


# 定义一个简单的线性函数 y = ax + b
def linear_function(params, x):
    a, b = params
    return a * x + b


# 定义均方误差损失函数
def loss_function(params, x, y):
    y_pred = linear_function(params, x)
    return jnp.mean((y - y_pred) ** 2)


key = random.PRNGKey(0)


def points_from_alice():
    x = random.uniform(key, (10,), minval=0, maxval=10)
    return x


def points_from_bob():
    x = random.uniform(key, (10,), minval=0, maxval=10)
    a_true = 2.5
    b_true = 5
    y = a_true * x + b_true + random.normal(key, (10,)) * 0.5
    return y


learning_rate = 0.01
num_epochs = 100


def run_on_cpu():
    start = time()

    x = points_from_alice()
    y = points_from_bob()

    # 初始化参数
    params = jnp.array([1.0, 0.0])

    # 损失函数关于参数的梯度
    grad_loss = grad(loss_function)

    # 训练模型

    for epoch in range(num_epochs):
        grads = grad_loss(params, x, y)
        params -= learning_rate * grads

    end = time()
    print("Trained parameters: a =", params[0], ", b =", params[1])
    print("Time cost on CPU: ", end - start)


def init_params():
    return jnp.array([1.0, 0.0])


def fit(params, grads, learning_rate):
    return params - learning_rate * grads


def run_on_spu():
    start = time()

    x = ppd.device("P1")(points_from_alice)()
    y = ppd.device("P2")(points_from_bob)()

    params = jnp.array([1.0, 0.0])

    grad_loss = grad(loss_function)

    for epoch in range(num_epochs):
        grads = ppd.device("SPU")(grad_loss)(params, x, y)
        params = ppd.device("SPU")(fit, static_argnums=(2,))(
            params, grads, learning_rate
        )

    params = ppd.get(params)

    end = time()

    print("Trained parameters: a =", params[0], ", b =", params[1])
    print("Time cost on SPU: ", end - start)


if __name__ == "__main__":
    run_on_spu()
    run_on_cpu()

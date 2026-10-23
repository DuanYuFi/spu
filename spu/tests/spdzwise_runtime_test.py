# Copyright 2026 The SPU Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# https://www.apache.org/licenses/LICENSE-2.0

"""SPDZwise RING Python IO/runtime and minimal PPML acceptance fixtures.

Run against a freshly built extension: python -m unittest spu.tests.spdzwise_runtime_test
"""

import json
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from spu import api, libspu
from spu.utils.simulation import PropagatingThread, Simulator, sim_jax


def config():
    result = libspu.RuntimeConfig(
        protocol=libspu.ProtocolKind.SPDZWISE, field=libspu.FieldType.FM64
    )
    result.spdzwise_config.domain = libspu.SpdzWiseConfig.RING
    result.share_max_chunk_size = 32
    return result


class SpdzWiseRuntimeTest(unittest.TestCase):
    def test_spdzwise_domain_config(self):
        config = libspu.RuntimeConfig(
            protocol=libspu.ProtocolKind.SPDZWISE,
            field=libspu.FieldType.FM64,
        )
        self.assertEqual(config.spdzwise_config.domain, libspu.SpdzWiseConfig.FIELD)
        self.assertEqual(config.spdzwise_config.check_threshold, 5_000_000)
        for domain in (libspu.SpdzWiseConfig.FIELD, libspu.SpdzWiseConfig.RING):
            config.spdzwise_config.domain = domain
            config.spdzwise_config.check_threshold = 10_000_000
            restored = libspu.RuntimeConfig()
            self.assertTrue(restored.ParseFromString(config.SerializeToString()))
            self.assertEqual(restored.spdzwise_config.domain, domain)
            self.assertEqual(restored.spdzwise_config.check_threshold, 10_000_000)
            copied = libspu.RuntimeConfig(config)
            self.assertEqual(copied.spdzwise_config.domain, domain)
            self.assertEqual(copied.spdzwise_config.check_threshold, 10_000_000)
        self.assertTrue(
            config.ParseFromJsonString(
                json.dumps(
                    {
                        "protocol": "SPDZWISE",
                        "field": "FM64",
                        "spdzwise_config": {"domain": "FIELD"},
                    }
                )
            )
        )
        self.assertEqual(config.spdzwise_config.domain, libspu.SpdzWiseConfig.FIELD)

        self.assertEqual(config.spdzwise_config.check_threshold, 5_000_000)
        self.assertTrue(
            config.ParseFromJsonString(
                json.dumps(
                    {
                        "protocol": "SPDZWISE",
                        "field": "FM64",
                        "spdzwise_config": {
                            "domain": "RING",
                            "check_threshold": 123,
                        },
                    }
                )
            )
        )
        self.assertEqual(config.spdzwise_config.check_threshold, 123)

    def test_small_check_threshold_runtime(self):
        cfg = config()
        cfg.spdzwise_config.check_threshold = 2
        x = np.array([-3, -1, 0, 5], np.int32)
        actual = sim_jax(Simulator(3, cfg), lambda a: (a * a) * a)(x)
        np.testing.assert_array_equal(actual, x**3)

    def test_io_rank_metadata(self):
        io = api.Io(3, config())
        x = np.array([-3, 0, 1, 5], np.int64)
        shares = io.make_shares(x, libspu.Visibility.VIS_SECRET)
        np.testing.assert_array_equal(io.reconstruct(shares), x)
        with self.assertRaises(RuntimeError):
            io.reconstruct([shares[1], shares[0], shares[2]])

    def test_repeated_runtime_and_export_accessors(self):
        cfg = config()
        io = api.Io(3, cfg)
        x = np.array([-3, -1, 0, 5], np.int64)
        shares = io.make_shares(x, libspu.Visibility.VIS_SECRET)
        desc = libspu.link.Desc()
        for rank in range(3):
            desc.add_party(str(rank), f"thread_{rank}")
        code = b"""
func.func @main(%x: tensor<4x!pphlo.secret<i64>>) -> tensor<4x!pphlo.secret<i64>> {
  %y = pphlo.multiply %x, %x : tensor<4x!pphlo.secret<i64>>
  return %y : tensor<4x!pphlo.secret<i64>>
}"""

        def run(rank):
            rt = api.Runtime(libspu.link.create_mem(desc, rank), cfg)
            rt.set_var("x", shares[rank])
            rt.run(
                libspu.Executable(
                    name="square", input_names=["x"], output_names=["y"], code=code
                )
            )
            first = rt.get_var("y")
            self.assertEqual(rt.get_var_chunk_count("y"), len(first.share_chunks))
            self.assertEqual(rt.get_var_meta("y").SerializeToString(), first.meta)
            rt.run(
                libspu.Executable(
                    name="square_again", input_names=["y"], output_names=["z"], code=code
                )
            )
            return first, rt.get_var("z")

        jobs = [PropagatingThread(target=run, args=(rank,)) for rank in range(3)]
        for job in jobs:
            job.start()
        outputs = [job.join() for job in jobs]
        for index, expected in enumerate((x**2, x**4)):
            np.testing.assert_array_equal(
                io.reconstruct([output[index] for output in outputs]), expected
            )

    def test_lr_inference_and_gradient_update(self):
        def step(x, y, w):
            pred = jax.nn.sigmoid(x @ w)
            return pred, w - 0.1 * (x.T @ (pred - y)) / x.shape[0]

        x = np.array([[0.2, -0.1], [-0.3, 0.4], [0.1, 0.2]], np.float32)
        y = np.array([0, 1, 1], np.float32)
        w = np.array([0.25, -0.5], np.float32)
        for fraction_bits in (0, 16):
            with self.subTest(fraction_bits=fraction_bits):
                cfg = config()
                cfg.fxp_fraction_bits = fraction_bits
                actual = sim_jax(Simulator(3, cfg), step)(x, y, w)
                for got, expected in zip(actual, step(x, y, w)):
                    np.testing.assert_allclose(got, expected, atol=0.01, rtol=0.01)

    def test_relu_logsoftmax_and_batchnorm(self):
        def nn(x, w, var):
            hidden = jax.nn.relu(x @ w)
            normalized = hidden * jax.lax.rsqrt(var + 1e-3)
            return jax.nn.log_softmax(normalized)

        args = (
            np.array([[0.2, -0.1], [0.1, 0.3]], np.float32),
            np.array([[0.5, -0.2], [0.3, 0.4]], np.float32),
            np.array([0.5, 1.0], np.float32),
        )
        actual = sim_jax(Simulator(3, config()), nn)(*args)
        np.testing.assert_allclose(actual, nn(*args), atol=0.02, rtol=0.02)

    def test_conv_batchnorm_relu(self):
        def cnn(x, weights, variance):
            y = jax.lax.conv_general_dilated(
                x,
                weights,
                (1, 1),
                "VALID",
                dimension_numbers=("NHWC", "HWIO", "NHWC"),
            )
            return jax.nn.relu(y * jax.lax.rsqrt(variance + 1e-3))

        args = (
            np.arange(9, dtype=np.float32).reshape(1, 3, 3, 1) / 10,
            np.array([0.1, -0.2, 0.3, 0.4], np.float32).reshape(2, 2, 1, 1),
            np.array([0.5], np.float32),
        )
        actual = sim_jax(Simulator(3, config()), cnn)(*args)
        np.testing.assert_allclose(actual, cnn(*args), atol=0.02, rtol=0.02)


if __name__ == "__main__":
    unittest.main()

load("@rules_python//python:defs.bzl", "py_binary")

package(default_visibility = ["//visibility:public"])

py_binary(
    name = "camera_test",
    srcs = ["camera_test.py"],
    deps = [
        "//cyber/proto:unit_test_py_pb2",
        "//cyber/python/cyber_py3:cyber",
        "//modules/drivers/proto:sensor_image_py_pb2",
    ],
)

py_binary(
    name = "add_noise_test",
    srcs = ["add_noise_test.py"],
    deps = [
        "//cyber/proto:unit_test_py_pb2",
        "//cyber/python/cyber_py3:cyber",
        "//modules/drivers/proto:sensor_image_py_pb2",
    ],
)

py_binary(
    name = "lidar_test",
    srcs = ["lidar_test.py"],
    deps = [
        "//cyber/proto:unit_test_py_pb2",
        "//cyber/python/cyber_py3:cyber",
        "//modules/drivers/proto:pointcloud_py_pb2",
        "//modules/drivers/proto:sensor_image_py_pb2",
    ],
)
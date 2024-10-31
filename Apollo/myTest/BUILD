load("@rules_python//python:defs.bzl", "py_binary")

package(default_visibility = ["//visibility:public"])

py_binary(
    name = "camera",
    srcs = ["camera.py"],
    deps = [
        "//cyber/proto:unit_test_py_pb2",
        "//cyber/python/cyber_py3:cyber",
        "//modules/drivers/proto:sensor_image_py_pb2",
    ],
)
py_binary(
    name = "lidar",
    srcs = ["lidar.py"],
    deps = [
        "//cyber/proto:unit_test_py_pb2",
        "//cyber/python/cyber_py3:cyber",
        "//modules/drivers/proto:pointcloud_py_pb2",
        "//modules/drivers/proto:sensor_image_py_pb2",
        "//modules/myTest:lidar_operation"
    ],
)
py_binary(
    name = "add_noise",
    srcs = ["add_noise.py"],
    deps = [
        "//cyber/proto:unit_test_py_pb2",
        "//cyber/python/cyber_py3:cyber",
        "//modules/drivers/proto:sensor_image_py_pb2",
    ],
)
py_binary(
    name = "occlusion",
    srcs = ["occlusion.py"],
    deps = [
        "//cyber/proto:unit_test_py_pb2",
        "//cyber/python/cyber_py3:cyber",
        "//modules/drivers/proto:sensor_image_py_pb2",
    ],
)
py_library(
    name = "lidar_operation",
    srcs = ["lidar_operation.py"],
    deps = [
        "//cyber/proto:unit_test_py_pb2",
        "//cyber/python/cyber_py3:cyber",
        "//modules/drivers/proto:pointcloud_py_pb2",
        "//modules/drivers/proto:sensor_image_py_pb2",
    ],
)
py_library(
    name = "lidar_operation_vis",
    srcs = ["lidar_operation_vis.py"],
    deps = [
        "//cyber/proto:unit_test_py_pb2",
        "//cyber/python/cyber_py3:cyber",
        "//modules/drivers/proto:pointcloud_py_pb2",
        "//modules/drivers/proto:sensor_image_py_pb2",
    ],
)


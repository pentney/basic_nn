http_archive(
    name = "com_github_grpc_grpc",
    urls = [
        "https://github.com/grpc/grpc/archive/YOUR_GRPC_COMMIT_SHA.tar.gz",
    ],
    strip_prefix = "grpc-YOUR_GRPC_COMMIT_SHA",
)
load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")
grpc_deps()
load("@com_github_grpc_grpc//bazel:grpc_extra_deps.bzl", "grpc_extra_deps")
grpc_extra_deps()

cc_library(
  name = "fastmath",
  hdrs = ["fastmath.h"],
)

cc_library(
  name = "nn",
  srcs = ["nn.cc"],
  hdrs = ["nn.h"],
  deps = [
       ":fastmath",
  ]
)

cc_library(
  name = "dataset",
  srcs = ["dataset.cc"],
  hdrs = ["dataset.h"],
)
  
cc_library(
  name = "gradient_test",
  hdrs = ["gradient_test.h"],
)

cc_library(
  name = "csv",
  hdrs = ["csv.h"],
)

cc_binary(
  name = "nn_main",
  srcs = ["nn_main.cc"],
  deps = [
     ":csv",
     ":fastmath",
     ":dataset",
     ":nn",
  ],
  linkopts = ["-pthread"],
)

cc_test(
   name = "gradient_test_test",
   srcs = ["gradient_test_test.cc"],
   deps = [
        ":gradient_test",
   	"@gtest//:main",
   ]
)

cc_test(
   name = "dataset_test",
   srcs = ["dataset_test.cc"],
   deps = [
        ":dataset",
        "@gtest//:main",
   ],
)

cc_test(
   name = "nn_test",
   srcs = ["nn_test.cc"],
   deps = [
        ":gradient_test",
   	":nn",
	"@gtest//:main",
   ]
)
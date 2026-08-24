#include <hip/hip_runtime.h>
#include <rccl/rccl.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#define HIP_CHECK(cmd) do { \
  hipError_t e = (cmd); \
  if (e != hipSuccess) { \
    std::cerr << "HIP failure: " << hipGetErrorString(e) << std::endl; \
    std::exit(2); \
  } \
} while (0)

#define RCCL_CHECK(cmd) do { \
  ncclResult_t r = (cmd); \
  if (r != ncclSuccess) { \
    std::cerr << "RCCL failure: " << ncclGetErrorString(r) << std::endl; \
    std::exit(3); \
  } \
} while (0)

static int env_int(const char* name) {
  const char* value = std::getenv(name);
  if (!value) {
    std::cerr << "Missing environment variable " << name << std::endl;
    std::exit(4);
  }
  return std::stoi(value);
}

int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "usage: " << argv[0]
              << " UNIQUE_ID_FILE LOCAL_GPUS ELEMENTS ITERATIONS" << std::endl;
    return 1;
  }

  const std::filesystem::path id_path(argv[1]);
  const int local_gpus = std::stoi(argv[2]);
  const size_t elements = std::stoull(argv[3]);
  const int iterations = std::stoi(argv[4]);
  const int node_rank = env_int("SLURM_PROCID");
  const int nodes = env_int("SLURM_NTASKS");
  const int world = nodes * local_gpus;
  const size_t bytes = elements * sizeof(float);

  int visible_gpus = 0;
  HIP_CHECK(hipGetDeviceCount(&visible_gpus));
  if (visible_gpus < local_gpus) {
    std::cerr << "rank " << node_rank << " sees only " << visible_gpus
              << " GPUs, expected " << local_gpus << std::endl;
    return 5;
  }

  ncclUniqueId id;
  if (node_rank == 0) {
    RCCL_CHECK(ncclGetUniqueId(&id));
    const auto tmp_path = id_path.string() + ".tmp";
    {
      std::ofstream out(tmp_path, std::ios::binary | std::ios::trunc);
      out.write(reinterpret_cast<const char*>(&id), sizeof(id));
      out.flush();
      if (!out) {
        std::cerr << "Failed to write " << tmp_path << std::endl;
        return 6;
      }
    }
    std::filesystem::rename(tmp_path, id_path);
  } else {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::seconds(60);
    while ((!std::filesystem::exists(id_path) ||
            std::filesystem::file_size(id_path) != sizeof(id)) &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (!std::filesystem::exists(id_path)) {
      std::cerr << "Timed out waiting for " << id_path << std::endl;
      return 7;
    }
    std::ifstream in(id_path, std::ios::binary);
    in.read(reinterpret_cast<char*>(&id), sizeof(id));
    if (!in) {
      std::cerr << "Failed to read " << id_path << std::endl;
      return 8;
    }
  }

  std::vector<ncclComm_t> comms(local_gpus);
  std::vector<hipStream_t> streams(local_gpus);
  std::vector<float*> send(local_gpus);
  std::vector<float*> recv(local_gpus);

  RCCL_CHECK(ncclGroupStart());
  for (int gpu = 0; gpu < local_gpus; ++gpu) {
    HIP_CHECK(hipSetDevice(gpu));
    HIP_CHECK(hipStreamCreate(&streams[gpu]));
    HIP_CHECK(hipMalloc(&send[gpu], bytes));
    HIP_CHECK(hipMalloc(&recv[gpu], bytes));
    const int global_rank = node_rank * local_gpus + gpu;
    std::vector<float> input(elements, static_cast<float>(global_rank + 1));
    HIP_CHECK(hipMemcpy(send[gpu], input.data(), bytes, hipMemcpyHostToDevice));
    RCCL_CHECK(ncclCommInitRank(
        &comms[gpu], world, id, global_rank));
  }
  RCCL_CHECK(ncclGroupEnd());

  const auto start = std::chrono::steady_clock::now();
  for (int iteration = 0; iteration < iterations; ++iteration) {
    RCCL_CHECK(ncclGroupStart());
    for (int gpu = 0; gpu < local_gpus; ++gpu) {
      HIP_CHECK(hipSetDevice(gpu));
      RCCL_CHECK(ncclAllReduce(send[gpu], recv[gpu], elements, ncclFloat,
                               ncclSum, comms[gpu], streams[gpu]));
    }
    RCCL_CHECK(ncclGroupEnd());
  }

  const float expected = static_cast<float>(world * (world + 1) / 2);
  bool valid = true;
  for (int gpu = 0; gpu < local_gpus; ++gpu) {
    HIP_CHECK(hipSetDevice(gpu));
    HIP_CHECK(hipStreamSynchronize(streams[gpu]));
    float result = 0.0f;
    HIP_CHECK(hipMemcpy(&result, recv[gpu], sizeof(float),
                        hipMemcpyDeviceToHost));
    if (std::fabs(result - expected) > 0.01f) {
      std::cerr << "rank " << node_rank << " gpu " << gpu << " got "
                << result << ", expected " << expected << std::endl;
      valid = false;
    }
  }
  const double seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - start).count();

  for (int gpu = 0; gpu < local_gpus; ++gpu) {
    HIP_CHECK(hipSetDevice(gpu));
    RCCL_CHECK(ncclCommDestroy(comms[gpu]));
    HIP_CHECK(hipFree(send[gpu]));
    HIP_CHECK(hipFree(recv[gpu]));
    HIP_CHECK(hipStreamDestroy(streams[gpu]));
  }

  std::cout << "node_rank=" << node_rank << " world_gpus=" << world
            << " bytes=" << bytes << " iterations=" << iterations
            << " seconds=" << seconds << " validation="
            << (valid ? "OK" : "FAILED") << std::endl;
  return valid ? 0 : 9;
}

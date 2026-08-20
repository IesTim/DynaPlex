#include "dynaplex/torchavailability.h"
#include <iostream>
#if DP_TORCH_AVAILABLE
#include <torch/torch.h>
#endif

namespace DynaPlex ::TorchAvailability {

    bool TorchAvailable()
    {
#if DP_TORCH_AVAILABLE
        // DynaPlex already parallelizes across samples/trajectories with its own
        // thread pool (sized to hardware concurrency). Without this, every one of
        // those worker threads independently invokes libtorch's own default
        // intra-op thread pool (also sized to hardware concurrency) whenever it
        // calls into the network - nesting two full-width thread pools and causing
        // severe oversubscription (observed: ~8000+ live threads on a 128-core
        // machine). Networks here are tiny (few small linear layers), so per-call
        // intra-op parallelism buys nothing anyway; disabling it lets our own
        // outer-level parallelism do the work without contention.
        torch::set_num_threads(1);
        torch::set_num_interop_threads(1);
        return true;
#else


        return false;
#endif	
    }

    std::string TorchVersion()
    {
        std::string result;

#if DP_TORCH_AVAILABLE
        result = "DynaPlex: torch available, Version ";
        result += std::to_string(TORCH_VERSION_MAJOR) + ".";
        result += std::to_string(TORCH_VERSION_MINOR) + ".";
        result += std::to_string(TORCH_VERSION_PATCH) + "\t";
        if (torch::cuda::is_available())
        {
       //     result += "cuda available. ";
        }
        else
        {
      //      result += "cuda not available. ";
        }       
#else
        result = "DynaPlex: torch not available. To make available, set dynaplex_enable_pytorch to true and dynaplex_pytorch_path to an appropriate path, e.g. in CMakeUserPresets.txt";
#endif	

        return result;
    }
}
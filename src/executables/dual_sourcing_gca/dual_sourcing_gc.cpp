#include <iostream>
#include "dynaplex/dynaplexprovider.h"

using namespace DynaPlex;

int main() {
    // load mdp config
    auto& dp = DynaPlexProvider::Get();
    auto& system = dp.System();

    VarGroup mdp_config = VarGroup::LoadFromFile(system.filepath("mdp_config_examples", "dual_sourcing_backlog", "mdp_config_0.json"));
    DynaPlex::MDP mdp = dp.GetMDP(mdp_config);
    system << "MDP loaded: dual_sourcing_backlog" << std::endl;

    // load dcl config
    VarGroup dcl_config = VarGroup::LoadFromFile(system.filepath("mdp_config_examples", "dual_sourcing_backlog", "dcl_config_0.json"));

    int64_t num_gens;
    dcl_config.Get("num_gens", num_gens);

    // Initial CDI policy
    auto initial_policy = mdp->GetPolicy("cdi");
    system << "Initial policy: CDI" << std::endl;

    // output path
    std::string action_repr;
    mdp_config.Get("action_representation", action_repr);
    auto path = system.filepath("dual_sourcing_backlog", "GCA-DS_" + action_repr);

    system << "Action representation: " << action_repr << std::endl;
    system << "Training for " << num_gens << " generations..." << std::endl;

    // Train
    auto dcl = dp.GetDCL(mdp, initial_policy, dcl_config);
    dcl.TrainPolicy();

    // Save policy
    auto final_policy = dcl.GetPolicy(num_gens);
    dp.SavePolicy(final_policy, path);

    system << "Training complete. Policy saved." << std::endl;

    return 0;
}
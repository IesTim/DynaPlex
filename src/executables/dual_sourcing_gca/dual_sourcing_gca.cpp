#include <iostream>
#include "dynaplex/dynaplexprovider.h"
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>

using namespace DynaPlex;

int main(int argc, char *argv[])
{

    std::string dcl_config_name = "dcl_config_0.json";
    std::string dcl_config_base = "dcl_config_0";

    std::string mdp_config_name = "mdp_config_0.json";
    std::string mdp_config_base = "mdp_config_0";

    if (argc > 1)
        dcl_config_name = argv[1];
    dcl_config_base = dcl_config_name.substr(0, dcl_config_name.find('.'));
    if (argc > 2)
        mdp_config_name = argv[2];
    mdp_config_base = mdp_config_name.substr(0, mdp_config_name.find('.'));

    auto &dp = DynaPlexProvider::Get();
    auto &system = dp.System();

    // load mdp config
    VarGroup mdp_config = VarGroup::LoadFromFile(system.filepath("mdp_config_examples", "dual_sourcing_backlog", mdp_config_name));
    DynaPlex::MDP mdp = dp.GetMDP(mdp_config);
    system << "MDP loaded: dual_sourcing_backlog" << std::endl;

    // load dcl config
    VarGroup dcl_config = VarGroup::LoadFromFile(system.filepath("mdp_config_examples", "dual_sourcing_backlog", dcl_config_name));

    int64_t num_gens;
    dcl_config.Get("num_gens", num_gens);

    // Initial CDI policy
    auto initial_policy = mdp->GetPolicy("cdi");
    system << "Initial policy: CDI" << std::endl;

    // output path
    std::string action_repr;
    mdp_config.Get("action_representation", action_repr);

    auto now = std::chrono::system_clock::now();
    std::time_t time_t_now = std::chrono::system_clock::to_time_t(now);
    std::tm tm_now{};
#ifdef _WIN32
    localtime_s(&tm_now, &time_t_now);
#else
    localtime_r(&time_t_now, &tm_now);
#endif
    std::ostringstream ts;
    ts << std::put_time(&tm_now, "%Y%m%d_%H%M%S");
    std::string timestamp = ts.str();
    std::string run_name = action_repr + "_" + mdp_config_base + "_" + dcl_config_base + "_" + timestamp;

    auto path = system.filepath("dual_sourcing_backlog", "runs", run_name, "policy_final");

    VarGroup run_info;
    run_info.Add("action_representation", action_repr);
    run_info.Add("mdp_config", mdp_config_name);
    run_info.Add("dcl_config", dcl_config_name);
    run_info.Add("timestamp", timestamp);
    run_info.Add("num_gens", num_gens);
    run_info.Add("mdp_identifier", mdp->Identifier());

    run_info.SaveToFile(system.filepath("dual_sourcing", "runs", run_name, "run_info.json"), 4);

    system << "Action representation: " << action_repr << std::endl;
    system << "Training for " << num_gens << " generations..." << std::endl;
    system << "Output path: GCA-DS" + action_repr + "_" + mdp_config_base + "_" + dcl_config_base << std::endl;

    // Train
    auto dcl = dp.GetDCL(mdp, initial_policy, dcl_config);
    dcl.TrainPolicy();

    // Save policy
    auto final_policy = dcl.GetPolicy(num_gens);
    dp.SavePolicy(final_policy, path);

    system << "Training complete. Policy saved." << std::endl;

    return 0;
}
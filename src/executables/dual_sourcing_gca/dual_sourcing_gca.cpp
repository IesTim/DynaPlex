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

    // Save cost to JSON for between generation evaluation
    system << "Evaluating per-generation costs..." << std::endl;

    VarGroup eval_config = VarGroup::LoadFromFile(system.filepath("mdp_config_examples", "dual_sourcing_backlog", "eval_config_0.json"));
    VarGroup eval_sim_config;
    eval_config.Get("simulation", eval_sim_config);

    VarGroup instances_config = VarGroup::LoadFromFile(system.filepath("mdp_config_examples", "dual_sourcing_backlog", "instances_config.json"));
    std::vector<VarGroup> instances;
    instances_config.Get("instances", instances);

    int64_t train_l_max;
    int64_t periods;
    eval_config.Get("train_l_max", train_l_max);
    eval_config.Get("periods_per_trajectory", periods);

    std::vector<VarGroup> generation_results;

    for (int64_t gen = 1; gen < num_gens; gen++)
    {
        auto gen_policy = dcl.GetPolicy(gen);
        std::vector<VarGroup> instance_costs;

        for (auto &instance : instances)
        {
            std::string inst_name;
            instance.Get("name", inst_name);

            VarGroup inst_config;
            inst_config.Add("id", std::string("dual_sourcing_backlog"));
            inst_config.Add("K", int64_t(2));

            int64_t l_e, l_r;
            instance.Get("l_e", l_e);
            instance.Get("l_r", l_r);
            inst_config.Add("l_min", l_e);
            inst_config.Add("l_max", train_l_max);

            double mu, sigma, h, b, c_r, c_e;
            instance.Get("mu", mu);
            instance.Get("sigma", sigma);
            instance.Get("h", h);
            instance.Get("b", b);
            instance.Get("c_r", c_r);
            instance.Get("c_e", c_e);

            inst_config.Add("min_h", h);
            inst_config.Add("max_h", h);
            inst_config.Add("min_b", b);
            inst_config.Add("max_b", b);
            inst_config.Add("min_c", c_r);
            inst_config.Add("max_c", c_e);
            inst_config.Add("min_mu", mu);
            inst_config.Add("max_mu", mu);
            inst_config.Add("action_representation", action_repr);
            inst_config.Add("discount_factor", 1.0);

            DynaPlex::MDP inst_mdp = dp.GetMDP(inst_config);
            auto comparer = dp.GetPolicyComparer(inst_mdp, eval_sim_config);
            auto result = comparer.Assess(gen_policy);

            double cost;
            result.Get("mean", cost);
            double cost_per_period = cost / static_cast<double>(periods);

            VarGroup inst_result;
            inst_result.Add("instance", inst_name);
            inst_result.Add("cost", cost_per_period);
            instance_costs.push_back(inst_result);
        }

        VarGroup gen_result;
        gen_result.Add("generation", gen);
        gen_result.Add("instance_cost", instance_costs);
        generation_results.push_back(gen_result);

        system << "  Generation " << gen << " evaluated." << std::endl;
    }

    VarGroup gen_output;
    gen_output.Add("action_representation", action_repr);
    gen_output.Add("mdp_config", mdp_config);
    gen_output.Add("dcl_config", dcl_config);
    gen_output.Add("generations", generation_results);

    auto gen_path = system.filepath("dual_sourcing_backlog", "generations_costs_" + action_repr + "_" + mdp_config_base + "_" + dcl_config_base + ".json");
    gen_output.SaveToFile(gen_path, 4);
    system << "Per-generation costs saved." << std::endl;

    // Save policy
    auto final_policy = dcl.GetPolicy(num_gens);
    dp.SavePolicy(final_policy, path);

    system << "Training complete. Policy saved." << std::endl;

    return 0;
}
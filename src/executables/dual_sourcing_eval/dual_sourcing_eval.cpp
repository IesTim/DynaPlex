#include <iostream>
#include <string>
#include "dynaplex/dynaplexprovider.h"
#include "dynaplex/modelling/discretedist.h"
#include <cmath>

using namespace DynaPlex;

VarGroup BuildInstanceConfig(const VarGroup& instance) { 
    VarGroup config;
    config.Add("id", "dual_sourcing_backlog");
    config.Add("K", int64_t(2));

    int64_t l_e, l_r;
    instance.Get("l_e", l_e);
    instance.Get("l_r", l_r);
    config.Add("l_min", l_e);
    config.Add("l_max", l_r);

    double mu, sigma, h, b, c_r, c_e;
    instance.Get("mu", mu);
    instance.Get("sigma", sigma);
    instance.Get("h", h);
    instance.Get("b", b);
    instance.Get("c_r", c_r);
    instance.Get("c_e", c_e);
    
    config.Add("min_h", h); config.Add("max_h", h);
    config.Add("min_b", b); config.Add("max_b", b);
    config.Add("min_c", c_r); config.Add("max_c", c_e);
    config.Add("min_mu", mu); config.Add("max_mu", mu);
    config.Add("action_representation", std::string("flat_joint"));
    config.Add("discount_factor", 1.0);

    return config;
}

double EvaluatePolicy(DynaPlex::MDP& mdp, DynaPlex::Policy& policy, const VarGroup& tuning_config) {
    auto& dp = DynaPlexProvider::Get();
    auto comparer = dp.GetPolicyComparer(mdp, tuning_config);
    auto result = comparer.Assess(policy);
    double cost;
    result.Get("mean", cost);
    return cost;
}

auto add_gap = [&](const std::string& key, VarGroup instance_result) {
    double cdi_cost;
    VarGroup cdi_res;
    instance_result.Get("CDI", cdi_res);
    cdi_res.Get("cost", cdi_cost);

    VarGroup res;
    instance_result.Get(key, res);
    double cost;
    res.Get("cost", cost);
    double gap = (cost - cdi_cost) / cdi_cost * 100.0;
    res.Add("gap_vs_CDI_pct", gap);
    instance_result.Set(key, res);
};

void RunEval(const std::string& eval_config_name) {
    auto& dp = DynaPlexProvider::Get();
    auto& system = dp.System();

    VarGroup instances_config = VarGroup::LoadFromFile(system.filepath("mdp_config_examples", "dual_sourcing_backlog", "instances_config.json"));

    std::vector<VarGroup> instances;
    instances_config.Get("instances", instances);

    VarGroup tuned = VarGroup::LoadFromFile(system.filepath("dual_sourcing_backlog", "tuned_heuristic_params.json"));

    std::vector<VarGroup> tuned_instances;
    tuned.Get("tuned_policies", tuned_instances);

    VarGroup eval_config = VarGroup::LoadFromFile(system.filepath("mdp_config_examples", "dual_sourcing_backlog", eval_config_name));

    VarGroup sim_config;
    eval_config.Get("simulation", sim_config);

    std::string path_flat, path_sequential, path_multi;
    eval_config.Get("gca_flat_joint", path_flat);
    eval_config.Get("gca_sequential", path_sequential);
    eval_config.Get("gca_multi_discrete", path_multi);

    system << "Starting evaluation mode..." << std::endl;

    std::vector<VarGroup> results;

    for (size_t i = 0; i < instances.size(); i++) {
        auto& instance = instances[i];
        auto& tuned_inst = tuned_instances[i];

        std::string name;
        instance.Get("name", name);
        system << "Evaluating instance: " << name << std::endl;

        VarGroup mdp_config = BuildInstanceConfig(instance);
        DynaPlex::MDP mdp = dp.GetMDP(mdp_config);

        VarGroup instance_result;
        instance_result.Add("name", name);

        // cdi
        {
            VarGroup policy_config;
            VarGroup cdi_params;
            tuned_inst.Get("CDI", cdi_params);
            int64_t S_r, S_e;
            cdi_params.Get("S_r", S_r);
            cdi_params.Get("S_e", S_e);

            policy_config.Add("id", std::string("cdi"));
            policy_config.Add("S_r", S_r);
            policy_config.Add("S_e", S_e);
            auto policy = mdp->GetPolicy(policy_config);
            double cost = EvaluatePolicy(mdp, policy, sim_config);

            VarGroup res;
            res.Add("cost", cost);
            res.Add("S_r", S_r);
            res.Add("S_e", S_e);
            instance_result.Add("CDI", res);
            system << "  CDI cost: " << cost << std::endl; 
        }
        

        // di
        { 
            VarGroup policy_config;
            VarGroup di_params;
            tuned_inst.Get("DI", di_params);
            int64_t S;
            di_params.Get("S", S);

            policy_config.Add("id", std::string("di"));
            policy_config.Add("S", S);
            auto policy = mdp->GetPolicy(policy_config);
            double cost = EvaluatePolicy(mdp, policy, sim_config);

            VarGroup res;
            res.Add("cost", cost);
            res.Add("S", S);
            instance_result.Add("DI", res);
            system << "  DI cost: " << cost << std::endl;
        }
        

        // si
        {
            VarGroup policy_config;
            VarGroup si_params;
            tuned_inst.Get("SI", si_params);
            int64_t S;
            si_params.Get("S", S);

            policy_config.Add("id", std::string("si"));
            policy_config.Add("S", S);
            auto policy = mdp->GetPolicy(policy_config);
            double cost = EvaluatePolicy(mdp, policy, sim_config);

            VarGroup res;
            res.Add("cost", cost);
            res.Add("S", S);
            instance_result.Add("SI", res);
            system << "  SI cost: " << cost << std::endl;
        }
        


        // tbs
        {
            VarGroup policy_config;
            VarGroup tbs_params;
            tuned_inst.Get("TBS", tbs_params);
            int64_t S_e, c;
            tbs_params.Get("S_e", S_e);
            tbs_params.Get("c", c);

            policy_config.Add("id", std::string("tbs"));
            policy_config.Add("S_e", S_e);
            policy_config.Add("c", c);
            auto policy = mdp->GetPolicy(policy_config);
            double cost = EvaluatePolicy(mdp, policy, sim_config);

            VarGroup res;
            res.Add("cost", cost);
            res.Add("S_e", S_e);
            res.Add("c", c);
            instance_result.Add("TBS", res);
            system << "  TBS cost: " << cost << std::endl;
        }
        


        // GCA-DS flat
        {
            VarGroup mdp_flat = BuildInstanceConfig(instance);
            mdp_flat.Set("action_representation", std::string("flat_joint"));
            DynaPlex::MDP mdp_f = dp.GetMDP(mdp_flat);
            auto full_path = system.filepath("dual_sourcing_backlog", path_flat);
            auto policy = dp.LoadPolicy(mdp_f, full_path);
            double cost = EvaluatePolicy(mdp_f, policy, sim_config);

            VarGroup res;
            res.Add("cost", cost);
            instance_result.Add("GCA_flat_joint", res);
            system << "  GCA flat_joint cost: " << cost << std::endl;
        }
        


        //  GCA-DS sequential
        {
            VarGroup mdp_seq = BuildInstanceConfig(instance);
            mdp_seq.Set("action_representation", std::string("sequential"));
            DynaPlex::MDP mdp_s = dp.GetMDP(mdp_seq);
            auto full_path = system.filepath("dual_sourcing_backlog", path_sequential);
            auto policy = dp.LoadPolicy(mdp_s, full_path);
            double cost = EvaluatePolicy(mdp_s, policy, sim_config);

            VarGroup res;
            res.Add("cost", cost);
            instance_result.Add("GCA_sequential", res);
            system << "  GCA sequential cost: " << cost << std::endl;
        }
        
        // Compute gaps vs CDI
        add_gap("DI", instance_result);
        add_gap("SI", instance_result);
        add_gap("TBS", instance_result);
        add_gap("GCA_flat_joint", instance_result);
        add_gap("GCA_sequential", instance_result);

        results.push_back(instance_result);
        system << "  Done." << std::endl;
    }

    // Save results
    VarGroup output;
    output.Add("eval_results", results);
    auto out_path = system.filepath("dual_sourcing_backlog", "eval_results.json");
    output.SaveToFile(out_path);
    system << "Eval complete. Results saved." << std::endl;
}

int main(int argc, char* argv[]) {
    auto& dp = DynaPlexProvider::Get();
    auto& system = dp.System();

    if (argc < 3) {
        system << "Usage: dual_sourcing_eval <mode> <config>" << std::endl;
        system << "Modes: eval | sweep | convergence | horizon" << std::endl;
        return 1;
    }

    std::string mode = argv[1];
    std::string config_name = argv[2];

    if (mode == "eval")
        RunEval(config_name);

    return 0;
}
#include <iostream>
#include <string>
#include <cmath>
#include "dynaplex/dynaplexprovider.h"
#include "dynaplex/modelling/discretedist.h"

using namespace DynaPlex;

VarGroup BuildInstanceConfig(const VarGroup &instance, int64_t train_l_max, const std::string &action_representation = "flat_joint")
{
    VarGroup config;
    config.Add("id", std::string("dual_sourcing_backlog"));
    config.Add("K", int64_t(2));

    int64_t l_e, l_r;
    instance.Get("l_e", l_e);
    instance.Get("l_r", l_r);
    config.Add("l_min", l_e);
    config.Add("l_max", train_l_max);

    double mu, sigma, h, b, c_r, c_e;
    instance.Get("mu", mu);
    instance.Get("sigma", sigma);
    instance.Get("h", h);
    instance.Get("b", b);
    instance.Get("c_r", c_r);
    instance.Get("c_e", c_e);

    config.Add("min_h", h);
    config.Add("max_h", h);
    config.Add("min_b", b);
    config.Add("max_b", b);
    config.Add("min_c", c_r);
    config.Add("max_c", c_e);
    config.Add("min_mu", mu);
    config.Add("max_mu", mu);
    config.Add("action_representation", action_representation);
    config.Add("discount_factor", 1.0);

    return config;
}

double EvaluatePolicy(DynaPlex::MDP &mdp, DynaPlex::Policy &policy, const VarGroup &sim_config)
{
    auto &dp = DynaPlexProvider::Get();
    auto comparer = dp.GetPolicyComparer(mdp, sim_config);
    auto result = comparer.Assess(policy);
    double cost;
    result.Get("mean", cost);
    int64_t periods;
    sim_config.Get("periods_per_trajectory", periods);
    return cost / static_cast<double>(periods);
}

double ComputeGap(double policy_cost, double cdi_cost)
{
    return (policy_cost - cdi_cost) / cdi_cost * 100.0;
}

bool EvaluateGCA(DynaPlex::MDP &mdp, const std::string &weights_path, const VarGroup &sim_config, double &cost_out)
{
    auto &dp = DynaPlexProvider::Get();
    auto &system = dp.System();
    try
    {
        auto full_path = system.filepath("dual_sourcing_backlog", weights_path);
        auto policy = dp.LoadPolicy(mdp, full_path);
        cost_out = EvaluatePolicy(mdp, policy, sim_config);
        return true;
    }
    catch (const DynaPlex::Error &e)
    {
        system << "  Skipped " << weights_path << " (" << e.what() << ")" << std::endl;
        return false;
    }
}

void RunEval(const std::string &eval_config_name)
{
    auto &dp = DynaPlexProvider::Get();
    auto &system = dp.System();

    // Load eval config
    VarGroup eval_config = VarGroup::LoadFromFile(system.filepath("mdp_config_examples", "dual_sourcing_backlog", eval_config_name));

    VarGroup sim_config;
    eval_config.Get("simulation", sim_config);

    int64_t train_l_max;
    eval_config.Get("train_l_max", train_l_max);

    std::string path_flat, path_sequential;
    eval_config.Get("gca_flat_joint", path_flat);
    eval_config.Get("gca_sequential", path_sequential);

    // Load instances
    VarGroup instances_config = VarGroup::LoadFromFile(system.filepath("mdp_config_examples","dual_sourcing_backlog", "instances_config.json"));
    std::vector<VarGroup> instances;
    instances_config.Get("instances", instances);

    // Load tuned heuristic parameters
    VarGroup tuned = VarGroup::LoadFromFile(system.filepath("dual_sourcing_backlog", "tuned_heuristic_params.json"));
    std::vector<VarGroup> tuned_instances;
    tuned.Get("tuned_policies", tuned_instances);

    system << "Starting eval mode on " << instances.size() << " instances..." << std::endl;

    std::vector<VarGroup> results;

    for (size_t i = 0; i < instances.size(); i++)
    {
        auto &instance = instances[i];
        auto &tuned_inst = tuned_instances[i];

        std::string name;
        instance.Get("name", name);
        system << "Evaluating instance: " << name << std::endl;

        VarGroup mdp_config = BuildInstanceConfig(instance, train_l_max, "flat_joint");
        DynaPlex::MDP mdp = dp.GetMDP(mdp_config);

        VarGroup instance_result;
        instance_result.Add("name", name);

        instance_result.Add("parameters", instance);

        // -----------------------------------------------
        // Evaluate heuristics
        // -----------------------------------------------
        double cdi_cost = 0.0;

        {
            VarGroup cdi_params;
            tuned_inst.Get("CDI", cdi_params);
            int64_t S_r, S_e;
            cdi_params.Get("S_r", S_r);
            cdi_params.Get("S_e", S_e);
            VarGroup policy_config;
            policy_config.Add("id", std::string("cdi"));
            policy_config.Add("S_r", S_r);
            policy_config.Add("S_e", S_e);
            auto policy = mdp->GetPolicy(policy_config);
            cdi_cost = EvaluatePolicy(mdp, policy, sim_config);
            VarGroup res;
            res.Add("cost", cdi_cost);
            res.Add("gap_vs_CDI_pct", 0.0);
            res.Add("S_r", S_r);
            res.Add("S_e", S_e);
            instance_result.Add("CDI", res);
            system << "  CDI cost: " << cdi_cost << std::endl;
        }

        {
            VarGroup di_params;
            tuned_inst.Get("DI", di_params);
            int64_t S;
            di_params.Get("S", S);
            VarGroup policy_config;
            policy_config.Add("id", std::string("di"));
            policy_config.Add("S", S);
            auto policy = mdp->GetPolicy(policy_config);
            double cost = EvaluatePolicy(mdp, policy, sim_config);
            VarGroup res;
            res.Add("cost", cost);
            res.Add("gap_vs_CDI_pct", ComputeGap(cost, cdi_cost));
            res.Add("S", S);
            instance_result.Add("DI", res);
            system << "  DI cost: " << cost << std::endl;
        }

        {
            VarGroup si_params;
            tuned_inst.Get("SI", si_params);
            int64_t S;
            si_params.Get("S", S);
            VarGroup policy_config;
            policy_config.Add("id", std::string("si"));
            policy_config.Add("S", S);
            auto policy = mdp->GetPolicy(policy_config);
            double cost = EvaluatePolicy(mdp, policy, sim_config);
            VarGroup res;
            res.Add("cost", cost);
            res.Add("gap_vs_CDI_pct", ComputeGap(cost, cdi_cost));
            res.Add("S", S);
            instance_result.Add("SI", res);
            system << "  SI cost: " << cost << std::endl;
        }

        {
            VarGroup tbs_params;
            tuned_inst.Get("TBS", tbs_params);
            int64_t S_e, c;
            tbs_params.Get("S_e", S_e);
            tbs_params.Get("c", c);
            VarGroup policy_config;
            policy_config.Add("id", std::string("tbs"));
            policy_config.Add("S_e", S_e);
            policy_config.Add("c", c);
            auto policy = mdp->GetPolicy(policy_config);
            double cost = EvaluatePolicy(mdp, policy, sim_config);
            VarGroup res;
            res.Add("cost", cost);
            res.Add("gap_vs_CDI_pct", ComputeGap(cost, cdi_cost));
            res.Add("S_e", S_e);
            res.Add("c", c);
            instance_result.Add("TBS", res);
            system << "  TBS cost: " << cost << std::endl;
        }

        // Evaluate GCA-DS flat_joint
        {
            VarGroup mdp_flat = BuildInstanceConfig(instance, train_l_max, "flat_joint");
            DynaPlex::MDP mdp_f = dp.GetMDP(mdp_flat);
            double cost = 0.0;
            if (EvaluateGCA(mdp_f, path_flat, sim_config, cost))
            {
                VarGroup res;
                res.Add("cost", cost);
                res.Add("gap_vs_CDI_pct", ComputeGap(cost, cdi_cost));
                instance_result.Add("GCA_flat_joint", res);
                system << "  GCA flat_joint cost: " << cost << std::endl;
            }
        }

        // Evaluate GCA-DS sequential
        {
            VarGroup mdp_seq = BuildInstanceConfig(instance, train_l_max, "sequential");
            DynaPlex::MDP mdp_s = dp.GetMDP(mdp_seq);
            double cost = 0.0;
            if (EvaluateGCA(mdp_s, path_sequential, sim_config, cost))
            {
                VarGroup res;
                res.Add("cost", cost);
                res.Add("gap_vs_CDI_pct", ComputeGap(cost, cdi_cost));
                instance_result.Add("GCA_sequential", res);
                system << "  GCA sequential cost: " << cost << std::endl;
            }
        }

        results.push_back(instance_result);
        system << "  Instance " << name << " done." << std::endl;
    }

    // Save results
    VarGroup output;
    output.Add("experiment", std::string("benchmark_comparison"));
    output.Add("instances", results);

    auto out_path = system.filepath("dual_sourcing_backlog", "benchmark_results.json");
    output.SaveToFile(out_path, 4);
    system << "Eval complete. Results saved." << std::endl;
}

// Parameter evaluation
void RunSweep(const std::string &eval_config_name)
{
    auto &dp = DynaPlexProvider::Get();
    auto &system = dp.System();
    system << "Sweep mode not yet implemented." << std::endl;
}

// Horizon evaluation
void RunHorizon(const std::string &eval_config_name)
{
    auto &dp = DynaPlexProvider::Get();
    auto &system = dp.System();
    system << "Horizon mode not yet implemented." << std::endl;
}

// Structural comparison
void RunAnalyze(const std::string &eval_config_name)
{
    auto &dp = DynaPlexProvider::Get();
    auto &system = dp.System();
    system << "Analyze mode not yet implemented." << std::endl;
}

int main(int argc, char *argv[])
{
    auto &dp = DynaPlexProvider::Get();
    auto &system = dp.System();

    if (argc < 3)
    {
        system << "Usage: dual_sourcing_eval <mode> <config>" << std::endl;
        system << "Modes: eval | sweep | horizon | analyze" << std::endl;
        return 1;
    }

    std::string mode = argv[1];
    std::string config_name = argv[2];

    if (mode == "eval") RunEval(config_name);
    else if (mode == "sweep") RunSweep(config_name);
    else if (mode == "horizon") RunHorizon(config_name);
    else if (mode == "analyze") RunAnalyze(config_name);
    else
    {
        system << "Unknown mode: " << mode << std::endl;
        return 1;
    }

    return 0;
}
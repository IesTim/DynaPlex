#include <iostream>
#include <string>
#include <cmath>
#include "dynaplex/dynaplexprovider.h"
#include "dynaplex/modelling/discretedist.h"
#include "../../lib/models/models/dual_sourcing_backlog/tuning_utils.h"

using namespace DynaPlex;

VarGroup BuildInstanceConfig(const VarGroup &instance, int64_t train_l_max, double train_max_mu, double train_max_b, double train_max_c, double train_min_h, const std::string &action_representation = "flat_joint")
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

    config.Add("min_h", train_min_h);
    config.Add("max_h", h);
    config.Add("min_b", b);
    config.Add("max_b", train_max_b);
    config.Add("min_c", c_r);
    config.Add("max_c", train_max_c);
    config.Add("min_mu", mu);
    config.Add("max_mu", train_max_mu);
    config.Add("action_representation", action_representation);
    config.Add("discount_factor", 1.0);

    return config;
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
        auto full_path = system.filepath("dual_sourcing", "runs", weights_path, "policy_final");
        auto policy = dp.LoadPolicy(mdp, full_path);
        cost_out = EvaluatePolicyTuning(mdp, policy, sim_config);
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
    VarGroup eval_config = VarGroup::LoadFromFile(system.filepath("mdp_config_examples", "dual_sourcing_backlog", "configs", eval_config_name));

    VarGroup sim_config;
    eval_config.Get("simulation", sim_config);

    int64_t train_l_max;
    eval_config.Get("train_l_max", train_l_max);

    double train_max_mu;
    eval_config.Get("train_max_mu", train_max_mu);

    double train_max_b;
    eval_config.Get("train_max_b", train_max_b);

    double train_max_c;
    eval_config.Get("train_max_c", train_max_c);

    double train_min_h;
    eval_config.Get("train_min_h", train_min_h);

    std::string path_flat, path_sequential;
    eval_config.Get("gca_flat_joint", path_flat);
    eval_config.Get("gca_sequential", path_sequential);

    // Load instances
    VarGroup instances_config = VarGroup::LoadFromFile(system.filepath("mdp_config_examples", "dual_sourcing_backlog", "configs", "instances_config.json"));
    std::vector<VarGroup> instances;
    instances_config.Get("instances", instances);

    // Load tuned heuristic parameters
    VarGroup tuned = VarGroup::LoadFromFile(system.filepath("dual_sourcing", "tuning", "tuned_heuristic_params.json"));
    std::vector<VarGroup> tuned_policies;
    tuned.Get("tuned_policies", tuned_policies);

    system << "Starting eval mode on " << instances.size() << " instances..." << std::endl;

    std::vector<VarGroup> results;

    for (size_t i = 0; i < instances.size(); i++)
    {
        auto &instance = instances[i];
        auto &tuned_policy = tuned_policies[i];

        std::string name;
        instance.Get("name", name);
        system << "Evaluating instance: " << name << std::endl;

        VarGroup mdp_config = BuildInstanceConfig(instance, train_l_max, train_max_mu, train_max_b, train_max_c, train_min_h, "flat_joint");
        DynaPlex::MDP mdp = dp.GetMDP(mdp_config);

        VarGroup instance_result;
        instance_result.Add("name", name);

        instance_result.Add("parameters", instance);

        // Evaluate heuristics
        double cdi_cost = 0.0;
        {
            VarGroup cdi_params;
            tuned_policy.Get("CDI", cdi_params);
            int64_t S_r, S_e;
            cdi_params.Get("S_r", S_r);
            cdi_params.Get("S_e", S_e);
            VarGroup policy_config;
            policy_config.Add("id", std::string("cdi"));
            policy_config.Add("S_r", S_r);
            policy_config.Add("S_e", S_e);
            auto policy = mdp->GetPolicy(policy_config);
            cdi_cost = EvaluatePolicyTuning(mdp, policy, sim_config);
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
            tuned_policy.Get("DI", di_params);
            int64_t S;
            di_params.Get("S", S);
            VarGroup policy_config;
            policy_config.Add("id", std::string("di"));
            policy_config.Add("S", S);
            auto policy = mdp->GetPolicy(policy_config);
            double cost = EvaluatePolicyTuning(mdp, policy, sim_config);
            VarGroup res;
            res.Add("cost", cost);
            res.Add("gap_vs_CDI_pct", ComputeGap(cost, cdi_cost));
            res.Add("S", S);
            instance_result.Add("DI", res);
            system << "  DI cost: " << cost << std::endl;
        }

        {
            VarGroup si_params;
            tuned_policy.Get("SI", si_params);
            int64_t S;
            si_params.Get("S", S);
            VarGroup policy_config;
            policy_config.Add("id", std::string("si"));
            policy_config.Add("S", S);
            auto policy = mdp->GetPolicy(policy_config);
            double cost = EvaluatePolicyTuning(mdp, policy, sim_config);
            VarGroup res;
            res.Add("cost", cost);
            res.Add("gap_vs_CDI_pct", ComputeGap(cost, cdi_cost));
            res.Add("S", S);
            instance_result.Add("SI", res);
            system << "  SI cost: " << cost << std::endl;
        }

        {
            VarGroup tbs_params;
            tuned_policy.Get("TBS", tbs_params);
            int64_t S_e, c;
            tbs_params.Get("S_e", S_e);
            tbs_params.Get("c", c);
            VarGroup policy_config;
            policy_config.Add("id", std::string("tbs"));
            policy_config.Add("S_e", S_e);
            policy_config.Add("c", c);
            auto policy = mdp->GetPolicy(policy_config);
            double cost = EvaluatePolicyTuning(mdp, policy, sim_config);
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
            VarGroup mdp_flat = BuildInstanceConfig(instance, train_l_max, train_max_mu, train_max_b, train_max_c, train_min_h, "flat_joint");
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
            VarGroup mdp_seq = BuildInstanceConfig(instance, train_l_max, train_max_mu, train_max_b, train_max_c, train_min_h, "sequential");
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

    auto out_path = system.filepath("dual_sourcing", "evaluation", "benchmark_results.json");
    output.SaveToFile(out_path, 4);
    system << "Eval complete. Results saved." << std::endl;
}

// Parameter evaluation
void RunParameterEvaluation(const std::string &eval_config_name)
{
    auto &dp = DynaPlexProvider::Get();
    auto &system = dp.System();

    // Load config
    VarGroup eval_config = VarGroup::LoadFromFile(system.filepath("mdp_config_examples", "dual_sourcing_backlog", "configs", eval_config_name));

    // Read parameter settings
    std::string parameter;
    double parameter_min, parameter_max, parameter_step;
    double train_min, train_max;
    eval_config.Get("parameter", parameter);
    eval_config.Get("parameter_min", parameter_min);
    eval_config.Get("parameter_max", parameter_max);
    eval_config.Get("parameter_step", parameter_step);
    eval_config.Get("train_min", train_min);
    eval_config.Get("train_max", train_max);

    VarGroup base_instance;
    eval_config.Get("base_instance", base_instance);

    int64_t train_l_max;
    eval_config.Get("train_l_max", train_l_max);

    double train_max_mu;
    eval_config.Get("train_max_mu", train_max_mu);

    double train_max_b;
    eval_config.Get("train_max_b", train_max_b);
    
    double train_max_c;
    eval_config.Get("train_max_c", train_max_c);

    double train_min_h;
    eval_config.Get("train_min_h", train_min_h);

    VarGroup sim_config, tuning_config;
    eval_config.Get("simulation", sim_config);
    eval_config.Get("tuning", tuning_config);

    std::string path_flat;
    eval_config.Get("gca_flat_joint", path_flat);

    system << "Starting parameter evaluation mode: " << parameter << " from " << parameter_min << " to " << parameter_max << " step " << parameter_step << std::endl;

    std::vector<VarGroup> parameter_results;

    // Generate parameter values
    for (double val = parameter_min; val <= parameter_max + 1e-9; val += parameter_step)
    {
        system << "Evaluating " << parameter
               << " = " << val << std::endl;

        VarGroup instance = base_instance;
        instance.Set(parameter, val);

        // Update sigma if mu changes
        if (parameter == "mu")
        {
            double base_mu, base_sigma;
            base_instance.Get("mu", base_mu);
            base_instance.Get("sigma", base_sigma);
            double cv = base_sigma / base_mu;
            instance.Set("sigma", val * cv);
        }

        VarGroup mdp_config = BuildInstanceConfig(instance, train_l_max, train_max_mu, train_max_b, train_max_c, train_min_h, "flat_joint");
        DynaPlex::MDP mdp = dp.GetMDP(mdp_config);

        double mu, sigma, h, b, c_r, c_e;
        int64_t l_e, l_r;
        instance.Get("mu", mu);
        instance.Get("sigma", sigma);
        instance.Get("h", h);
        instance.Get("b", b);
        instance.Get("c_r", c_r);
        instance.Get("c_e", c_e);
        instance.Get("l_e", l_e);
        instance.Get("l_r", l_r);

        auto dist = DiscreteDist::GetAdanEenigeResingDist(mu, sigma);
        auto demand_over_lr = DiscreteDist::GetZeroDist();
        for (int64_t i = 0; i <= l_r; i++)
            demand_over_lr = demand_over_lr.Add(dist);
        int64_t max_val = demand_over_lr.Fractile(b / (b + h));

        // Tune CDI
        VarGroup cdi_result = TuneCDI(mdp, tuning_config,
                                      mu, sigma, b, h, l_r, l_e, max_val);

        // Evaluate CDI
        double cdi_cost = 0.0;
        {
            int64_t S_r, S_e;
            cdi_result.Get("S_r", S_r);
            cdi_result.Get("S_e", S_e);
            VarGroup policy_config;
            policy_config.Add("id", std::string("cdi"));
            policy_config.Add("S_r", S_r);
            policy_config.Add("S_e", S_e);
            auto policy = mdp->GetPolicy(policy_config);
            cdi_cost = EvaluatePolicyTuning(mdp, policy, sim_config);
        }

        // Evaluate GCA-DS
        double gca_cost = 0.0;
        bool gca_success = EvaluateGCA(mdp, path_flat, sim_config, gca_cost);

        VarGroup point;
        point.Add("value", val);
        point.Add("in_distribution",
                  val >= train_min && val <= train_max);
        point.Add("CDI_cost", cdi_cost);
        point.Add("CDI_S_r", int64_t(0));
        point.Add("CDI_S_e", int64_t(0));
        cdi_result.Get("S_r", point);
        if (gca_success)
        {
            point.Add("GCA_flat_cost", gca_cost);
            point.Add("gap_vs_CDI_pct", ComputeGap(gca_cost, cdi_cost));
        }
        parameter_results.push_back(point);

        system << "  CDI: " << cdi_cost;
        if (gca_success)
            system << "  GCA: " << gca_cost
                   << "  gap: " << ComputeGap(gca_cost, cdi_cost) << "%";
        system << std::endl;
    }

    // Save results
    VarGroup output;
    output.Add("experiment", std::string("robustness_parameters"));
    output.Add("parameter", parameter);
    output.Add("train_min", train_min);
    output.Add("train_max", train_max);
    output.Add("parameter_points", parameter_results);

    auto out_path = system.filepath("dual_sourcing", "evaluation", "parameter_" + parameter + ".json");
    output.SaveToFile(out_path, 4);
    system << "Parameter evaluation complete. Results saved." << std::endl;
}

// Horizon evaluation
// template <typename StateType>
// double SimulatePeriod(DynaPlex::MDP &mdp, DynaPlex::Policy &policy, StateType &state, DynaPlex::RNG &rng)
// {
//     double cost = 0.0;

//     while (mdp->GetStateCategory(state).IsAwaitAction())
//     {
//         int64_t action = policy->GetAction(state);
//         cost += mdp->ModifyStateWithAction(state, action);
//     }

//     auto event = mdp->GetEvent(state, rng);
//     cost += mdp->ModifyStateWithEvent(state, event);

//     return cost;
// }

// void RunHorizon(const std::string &eval_config_name)
// {
//     auto &dp = DynaPlexProvider::Get();
//     auto &system = dp.System();

//     VarGroup horizon_config = VarGroup::LoadFromFile(system.filepath("mdp_config_examples", "dual_sourcing_backlog", "configs", eval_config_name));

//     VarGroup instance;
//     horizon_config.Get("instance", instance);

//     int64_t train_l_max;
//     horizon_config.Get("train_l_max", train_l_max);

//     int64_t num_trajectories, periods_per_trajectory;
//     horizon_config.Get("number_of_trajectories", num_trajectories);
//     horizon_config.Get("periods_per_trajectory", periods_per_trajectory);

//     VarGroup tuning_config;
//     horizon_config.Get("tuning", tuning_config);

//     std::string gca_policy_path;
//     std::string action_repr;
//     horizon_config.Get("gca_policy", gca_policy_path);
//     horizon_config.Get("action_representation", action_repr);

//     double mu, sigma, h, b, c_r, c_e;
//     int64_t l_e, l_r;
//     instance.Get("mu", mu);
//     instance.Get("sigma", sigma);
//     instance.Get("h", h);
//     instance.Get("b", b);
//     instance.Get("c_r", c_r);
//     instance.Get("c_e", c_e);
//     instance.Get("l_e", l_e);
//     instance.Get("l_r", l_r);

//     system << "Starting horizon mode..." << std::endl;
//     system << "Action representation: " << action_repr << std::endl;
//     system << "Trajectories: " << num_trajectories
//            << " Periods: " << periods_per_trajectory << std::endl;

//     VarGroup mdp_config = BuildInstanceConfig(instance, train_l_max, action_repr);
//     DynaPlex::MDP mdp = dp.GetMDP(mdp_config);

//     auto dist_bound = DiscreteDist::GetAdanEenigeResingDist(mu, sigma);
//     auto demand_bound = DiscreteDist::GetZeroDist();
//     for (int64_t i = 0; i <= l_r; i++)
//         demand_bound = demand_bound.Add(dist_bound);
//     int64_t max_val = demand_bound.Fractile(b / (b + h));

//     system << "Tuning clairvoyant CDI..." << std::endl;
//     VarGroup cdi_result = TuneCDI(mdp, tuning_config, mu, sigma, b, h, l_r, l_e, max_val);

//     int64_t S_r, S_e;
//     cdi_result.Get("S_r", S_r);
//     cdi_result.Get("S_e", S_e);

//     VarGroup cdi_policy_config;
//     cdi_policy_config.Add("id", std::string("cdi"));
//     cdi_policy_config.Add("S_r", S_r);
//     cdi_policy_config.Add("S_e", S_e);
//     auto cdi_policy = mdp->GetPolicy(cdi_policy_config);

//     auto full_path = system.filepath("dual_sourcing", "runs", gca_policy_path, "policy_final");
//     auto gca_policy = dp.LoadPolicy(mdp, full_path);

//     system << "Running simulation..." << std::endl;

//     std::vector<double> gca_costs(periods_per_trajectory, 0.0);
//     std::vector<double> cdi_costs(periods_per_trajectory, 0.0);

//     // Simulation
//     for (int64_t traj = 0; traj < num_trajectories; traj++)
//     {
//         DynaPlex::RNG rng_gca{true, traj};
//         DynaPlex::RNG rng_cdi{true, traj};

//         auto gca_state = mdp->GetInitialState();
//         auto cdi_state = mdp->GetInitialState();

//         for (int64_t t = 0; t < periods_per_trajectory; t++)
//         {
//             gca_costs[t] += SimulatePeriod(mdp, gca_policy, gca_state, rng_gca);
//             cdi_costs[t] += SimulatePeriod(mdp, cdi_policy, cdi_state, rng_cdi);
//         }
//     }

//     std::vector<VarGroup> periods_output;
//     for (int64_t t = 0; t < periods_per_trajectory; t++)
//     {
//         double gca_avg = gca_costs[t] / num_trajectories;
//         double cdi_avg = cdi_costs[t] / num_trajectories;

//         VarGroup p;
//         p.Add("period", t + 1);
//         p.Add("GCA_cost", gca_avg);
//         p.Add("CDI_cost", cdi_avg);
//         p.Add("gap_pct", ComputeGap(gca_avg, cdi_avg));
//         periods_output.push_back(p);
//     }

//     VarGroup output;
//     output.Add("experiment", std::string("online_estimation"));
//     output.Add("instance", instance);
//     output.Add("action_representation", action_repr);
//     output.Add("num_trajectories", num_trajectories);
//     output.Add("CDI_params", cdi_result);
//     output.Add("periods", periods_output);

//     auto out_path = system.filepath("dual_sourcing", "evaluation", "horizon_results.json");
//     output.SaveToFile(out_path, 4);
//     system << "Horizon complete. Results saved." << std::endl;
// }

void RunConvergence(const std::string &eval_config_name)
{
    auto& dp = DynaPlexProvider::Get();
    auto& system = dp.System();

    VarGroup eval_config = VarGroup::LoadFromFile(system.filepath("mdp_config_examples", "dual_sourcing_backlog", "configs", eval_config_name));

    VarGroup sim_config;
    eval_config.Get("simulation", sim_config);

    int64_t periods;
    sim_config.Get("periods_per_trajectory", periods);

    int64_t train_l_max;
    eval_config.Get("train_l_max", train_l_max);

    double train_max_mu;
    eval_config.Get("train_max_mu", train_max_mu);

    double train_max_b;
    eval_config.Get("train_max_b", train_max_b);

    double train_max_c;
    eval_config.Get("train_max_c", train_max_c);

    double train_min_h;
    eval_config.Get("train_min_h", train_min_h);

    VarGroup conv_config;
    eval_config.Get("convergence", conv_config);

    std::string run_info_path;
    conv_config.Get("run_info_path", run_info_path);

    VarGroup run_info = VarGroup::LoadFromFile(system.filepath(run_info_path));

    std::string mdp_identifier, action_repr;
    run_info.Get("mdp_identifier", mdp_identifier);
    run_info.Get("action_representation", action_repr);

    VarGroup instances_config = VarGroup::LoadFromFile(system.filepath("mdp_config_examples", "dual_sourcing_backlog", "configs", "instances_config.json"));
    std::vector<VarGroup> instances;
    instances_config.Get("instances", instances);

    VarGroup tuned = VarGroup::LoadFromFile(system.filepath("dual_sourcing", "tuning", "tuned_heuristic_params.json"));
    std::vector<VarGroup> tuned_instances;
    tuned.Get("tuned_policies", tuned_instances);

    std::vector<VarGroup> generation_results;

    int64_t gen = 1;
    while (true) {
        auto gen_weights_path = system.filepath(mdp_identifier, "dcl_policy_gen" + std::to_string(gen));
        std::vector<VarGroup> instance_results;
        bool any_loaded = false;

        for(size_t i = 0; i < instances.size(); i++) {
            auto& instance = instances[i];
            auto& tuned_inst = tuned_instances[i];

            std::string inst_name;
            instance.Get("name", inst_name);

            VarGroup inst_config = BuildInstanceConfig(instance, train_l_max, train_max_mu, train_max_b, train_max_c, train_min_h, action_repr);
            DynaPlex::MDP inst_mdp = dp.GetMDP(inst_config);

            DynaPlex::Policy gen_policy;
            try {
                gen_policy = dp.LoadPolicy(inst_mdp, gen_weights_path);
                any_loaded = true;
            } catch (const DynaPlex::Error& e) {
                    system << "LoadPolicy failed: " << e.what() << std::endl;
                goto save_results;
            }

            auto comparer = dp.GetPolicyComparer(inst_mdp, sim_config);
            auto result = comparer.Assess(gen_policy);
            double cost;
            result.Get("mean", cost);
            double cost_per_period = cost / static_cast<double>(periods);

            double cdi_cost = 0.0;
            {
                VarGroup cdi_params;
                tuned_inst.Get("CDI", cdi_params);
                int64_t S_r, S_e;
                cdi_params.Get("S_r", S_r);
                cdi_params.Get("S_e", S_e);
                VarGroup cdi_policy_config;
                cdi_policy_config.Add("id", std::string("cdi"));
                cdi_policy_config.Add("S_r", S_r);
                cdi_policy_config.Add("S_e", S_e);
                auto cdi_policy = inst_mdp->GetPolicy(cdi_policy_config);
                auto cdi_result = comparer.Assess(cdi_policy);
                double cdi_raw;
                cdi_result.Get("mean", cdi_raw);
                cdi_cost = cdi_raw / static_cast<double>(periods);
            }

            VarGroup inst_result;
            inst_result.Add("instance", inst_name);
            inst_result.Add("cost", cost_per_period);
            inst_result.Add("gap_vs_CDI_pct", ComputeGap(cost_per_period, cdi_cost));
            instance_results.push_back(inst_result);
        }

        VarGroup gen_result;
        gen_result.Add("generation", gen);
        gen_result.Add("instances", instance_results);
        generation_results.push_back(gen_result);
        gen++;
    }

    save_results:
    VarGroup output;
    output.Add("experiment", std::string("convergence"));
    output.Add("action_representation", action_repr);
    output.Add("generations_evaluated", gen - 1);
    output.Add("generations", generation_results);

    auto out_path = system.filepath("dual_sourcing", "evaluation", "convergence_" + action_repr + ".json");
    output.SaveToFile(out_path, 4);
}

int main(int argc, char *argv[])
{
    auto &dp = DynaPlexProvider::Get();
    auto &system = dp.System();

    if (argc < 3)
    {
        system << "Usage: dual_sourcing_eval <mode> <config>" << std::endl;
        system << "Modes: eval | parameter | horizon | convergence" << std::endl;
        return 1;
    }

    std::string mode = argv[1];
    std::string config_name = argv[2];

    if (mode == "eval")
        RunEval(config_name);
    else if (mode == "parameter")
        RunParameterEvaluation(config_name);
    // else if (mode == "horizon")
    //     RunHorizon(config_name);
    else if (mode == "convergence")
        RunConvergence(config_name);
    else
    {
        system << "Unknown mode: " << mode << std::endl;
        return 1;
    }

    return 0;
}
#include <iostream>
#include <limits>
#include <string>
#define DP_TORCH_AVAILABLE 1
#include "dynaplex/dynaplexprovider.h"
#include "dynaplex/modelling/discretedist.h"
#include "../../lib/models/models/dual_sourcing_backlog/tuning_utils.h"
#include "../../lib/nn/nn_policy.h"
#include <torch/torch.h>

using namespace DynaPlex;

VarGroup BuildInstanceConfig(const VarGroup& instance,
    const std::string& action_representation = "sequential")
{
    VarGroup config;
    config.Add("id", std::string("dual_sourcing_backlog"));
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

    config.Add("min_h", h);   config.Add("max_h", h);
    config.Add("min_b", b);   config.Add("max_b", b);
    config.Add("min_c", c_r); config.Add("max_c", c_e);
    config.Add("min_mu", mu); config.Add("max_mu", mu);
    config.Add("action_representation", action_representation);
    config.Add("discount_factor", 1.0);

    return config;
}

double EvaluateCDI(DynaPlex::MDP& mdp, int64_t S_r, int64_t S_e,
    const VarGroup& sim_config)
{
    VarGroup policy_config;
    policy_config.Add("id", std::string("cdi"));
    policy_config.Add("S_r", S_r);
    policy_config.Add("S_e", S_e);
    auto policy = mdp->GetPolicy(policy_config);

    auto& dp = DynaPlexProvider::Get();
    auto comparer = dp.GetPolicyComparer(mdp, sim_config);
    auto result = comparer.Assess(policy);
    double cost;
    result.Get("mean", cost);
    int64_t periods;
    sim_config.Get("periods_per_trajectory", periods);
    return cost / static_cast<double>(periods);
}

VarGroup GridSearchCDI(DynaPlex::MDP& mdp,
    const VarGroup& sim_config,
    int64_t max_val,
    const DynaPlex::System& system)
{
    double best_cost = std::numeric_limits<double>::infinity();
    int64_t best_S_r = 0, best_S_e = 0;

    system << "  Running grid search over S_r in [0," << max_val
           << "] and S_e in [0,S_r]..." << std::endl;

    int64_t total = (max_val + 1) * (max_val + 2) / 2;
    int64_t count = 0;

    for (int64_t S_r = 0; S_r <= max_val; S_r++)
    {
        for (int64_t S_e = 0; S_e <= S_r; S_e++)
        {
            double cost = EvaluateCDI(mdp, S_r, S_e, sim_config);
            if (cost < best_cost)
            {
                best_cost = cost;
                best_S_r = S_r;
                best_S_e = S_e;
            }
            count++;
            if (count % 100 == 0)
                system << "  Progress: " << count << "/" 
                       << total << "\r";
        }
    }
    system << std::endl;

    VarGroup result;
    result.Add("S_r", best_S_r);
    result.Add("S_e", best_S_e);
    result.Add("cost", best_cost);
    return result;
}

// Spot-checks the trained network's prescribed action across a sweep of a single instance
// parameter (holding everything else fixed), to see whether it varies sensibly with that
// parameter or collapses to a near-constant output regardless of context.
void DebugSweepNetworkActions(const DynaPlex::System& system, const std::string& run_path, const std::string& sweep_type)
{
    auto& dp = DynaPlexProvider::Get();

    struct Point { double mu, b; int64_t l_e, l_r; std::string label; };
    std::vector<Point> points;
    if (sweep_type == "b")
    {
        for (double b : {4.0, 9.0, 25.0, 49.0, 75.0, 99.0})
            points.push_back({4.0, b, 1, 2, "b=" + std::to_string(b)});
    }
    else if (sweep_type == "mu")
    {
        for (double mu : {2.0, 4.0, 6.0, 8.0, 10.0, 12.0})
            points.push_back({mu, 9.0, 1, 2, "mu=" + std::to_string(mu)});
    }
    else if (sweep_type == "l")
    {
        for (int64_t l_r : {2, 3, 4, 5})
            points.push_back({4.0, 9.0, 1, l_r, "l_r=" + std::to_string(l_r)});
    }
    else
        throw DynaPlex::Error("Unknown sweep_type: " + sweep_type);

    // Structural l_max must match what the network was trained with (max_lr sizes the
    // pipeline/state_vector). Wide-distribution training always uses l_max=5.
    int64_t train_l_max = 5;

    for (auto& pt : points)
    {
        VarGroup mdp_config;
        mdp_config.Add("id", std::string("dual_sourcing_backlog"));
        mdp_config.Add("K", int64_t(2));
        mdp_config.Add("l_min", int64_t(1));
        mdp_config.Add("l_max", train_l_max);
        mdp_config.Add("min_h", 1.0); mdp_config.Add("max_h", 1.0);
        mdp_config.Add("min_b", pt.b); mdp_config.Add("max_b", pt.b);
        mdp_config.Add("min_c", 1.0); mdp_config.Add("max_c", 1.001);
        mdp_config.Add("min_mu", pt.mu); mdp_config.Add("max_mu", pt.mu);
        mdp_config.Add("action_representation", std::string("sequential"));
        mdp_config.Add("discount_factor", 1.0);
        mdp_config.Add("max_order_size", int64_t(270));
        VarGroup fixed_instance;
        fixed_instance.Add("h", 1.0);
        fixed_instance.Add("b", pt.b);
        fixed_instance.Add("mu", pt.mu);
        fixed_instance.Add("sigma", pt.mu / 2.0);
        fixed_instance.Add("l_e", pt.l_e);
        fixed_instance.Add("l_r", pt.l_r);
        std::vector<double> costs = {1.0, 1.0};
        fixed_instance.Add("costs", costs);
        mdp_config.Add("fixed_instance", fixed_instance);

        DynaPlex::MDP mdp = dp.GetMDP(mdp_config);

        VarGroup cdi_config;
        cdi_config.Add("id", std::string("cdi"));
        cdi_config.Add("S_r", int64_t(11));
        cdi_config.Add("S_e", int64_t(6));
        auto driving_cdi = mdp->GetPolicy(cdi_config);

        auto full_path = system.filepath("dual_sourcing", "runs", run_path, "policy_final");
        auto network_policy = dp.LoadPolicy(mdp, full_path);

        DynaPlex::Trajectory traj(0);
        traj.RNGProvider.SeedEventStreams(false, 15112017, 0);
        mdp->InitiateState({ &traj, 1 });

        std::vector<int64_t> snapshot_periods = { 100, 150, 200 };
        std::vector<int64_t> network_actions;
        for (int64_t target_period : snapshot_periods)
        {
            while (traj.PeriodCount < target_period)
            {
                if (mdp->IncorporateUntilAction({ &traj, 1 }, target_period))
                    mdp->IncorporateAction({ &traj, 1 }, driving_cdi);
            }
            network_policy->SetAction({ &traj, 1 });
            network_actions.push_back(traj.NextAction);
            // advance the trajectory using CDI, not the network, so later snapshots stay
            // in a realistic, well-behaved state regardless of what the network prescribes.
            mdp->IncorporateAction({ &traj, 1 }, driving_cdi);
        }

        system << pt.label << ": network_actions=";
        for (auto a : network_actions) system << a << " ";
        system << std::endl;
    }
}

void DebugCompareActionsAcrossPolicies(const DynaPlex::System& system, const std::string& run_path)
{
    auto& dp = DynaPlexProvider::Get();

    VarGroup instances_config = VarGroup::LoadFromFile(system.filepath("mdp_config_examples", "dual_sourcing_backlog", "configs", "instances_config.json"));
    std::vector<VarGroup> instances;
    instances_config.Get("instances", instances);

    VarGroup tuned = VarGroup::LoadFromFile(system.filepath("dual_sourcing", "tuning", "tuned_heuristic_params.json"));
    std::vector<VarGroup> tuned_policies;
    tuned.Get("tuned_policies", tuned_policies);

    // Spread across the range: instance_1 (b=9, best CDI(6,11) match, worst network gap),
    // instance_7 (b=9, l_r=4, worst network gap overall),
    // instance_15 (b=99, mu=8, worst CDI(6,11) mismatch, relatively best network gap).
    std::vector<int64_t> instance_indices = { 0, 6, 14 };

    for (int64_t idx : instance_indices)
    {
        auto& instance = instances[idx];
        auto& tuned_policy = tuned_policies[idx];
        std::string name;
        instance.Get("name", name);

        int64_t l_e, l_r;
        instance.Get("l_e", l_e);
        instance.Get("l_r", l_r);
        double mu, sigma, h, b, c_r, c_e;
        instance.Get("mu", mu);
        instance.Get("sigma", sigma);
        instance.Get("h", h);
        instance.Get("b", b);
        instance.Get("c_r", c_r);
        instance.Get("c_e", c_e);

        VarGroup mdp_config;
        mdp_config.Add("id", std::string("dual_sourcing_backlog"));
        mdp_config.Add("K", int64_t(2));
        mdp_config.Add("l_min", l_e);
        mdp_config.Add("l_max", int64_t(5));
        mdp_config.Add("min_h", h); mdp_config.Add("max_h", h);
        mdp_config.Add("min_b", b); mdp_config.Add("max_b", b);
        mdp_config.Add("min_c", c_r); mdp_config.Add("max_c", c_e);
        mdp_config.Add("min_mu", mu); mdp_config.Add("max_mu", mu);
        mdp_config.Add("action_representation", std::string("sequential"));
        mdp_config.Add("discount_factor", 1.0);
        mdp_config.Add("max_order_size", int64_t(270));
        VarGroup fixed_instance;
        fixed_instance.Add("h", h);
        fixed_instance.Add("b", b);
        fixed_instance.Add("mu", mu);
        fixed_instance.Add("sigma", sigma);
        fixed_instance.Add("l_e", l_e);
        fixed_instance.Add("l_r", l_r);
        std::vector<double> costs = {c_e, c_r};
        fixed_instance.Add("costs", costs);
        mdp_config.Add("fixed_instance", fixed_instance);

        DynaPlex::MDP mdp = dp.GetMDP(mdp_config);

        VarGroup cdi_params;
        tuned_policy.Get("CDI", cdi_params);
        int64_t own_S_r, own_S_e;
        cdi_params.Get("S_r", own_S_r);
        cdi_params.Get("S_e", own_S_e);

        VarGroup own_cdi_config;
        own_cdi_config.Add("id", std::string("cdi"));
        own_cdi_config.Add("S_r", own_S_r);
        own_cdi_config.Add("S_e", own_S_e);
        auto own_cdi_policy = mdp->GetPolicy(own_cdi_config);

        VarGroup mismatched_cdi_config;
        mismatched_cdi_config.Add("id", std::string("cdi"));
        mismatched_cdi_config.Add("S_r", int64_t(11));
        mismatched_cdi_config.Add("S_e", int64_t(6));
        auto mismatched_cdi_policy = mdp->GetPolicy(mismatched_cdi_config);

        auto full_path = system.filepath("dual_sourcing", "runs", run_path, "policy_final");
        auto network_policy = dp.LoadPolicy(mdp, full_path);

        system << "=== " << name << " (own CDI S_e=" << own_S_e << ",S_r=" << own_S_r << ") ===" << std::endl;

        DynaPlex::Trajectory traj(0);
        traj.RNGProvider.SeedEventStreams(false, 15112017, idx);
        mdp->InitiateState({ &traj, 1 });

        std::vector<int64_t> snapshot_periods = { 100, 150, 200 };
        int64_t last_period = 0;
        for (int64_t target_period : snapshot_periods)
        {
            while (traj.PeriodCount < target_period)
            {
                if (mdp->IncorporateUntilAction({ &traj, 1 }, target_period))
                {
                    mdp->IncorporateAction({ &traj, 1 }, own_cdi_policy);
                }
            }
            // At this snapshot state (AwaitAction), query each policy without mutating state.
            own_cdi_policy->SetAction({ &traj, 1 });
            int64_t own_action = traj.NextAction;
            mismatched_cdi_policy->SetAction({ &traj, 1 });
            int64_t mismatched_action = traj.NextAction;
            network_policy->SetAction({ &traj, 1 });
            int64_t network_action = traj.NextAction;

            system << "  period " << target_period << ": own_CDI_action=" << own_action
                   << " mismatched_CDI(6,11)_action=" << mismatched_action
                   << " network_action=" << network_action << std::endl;

            // Advance using own CDI to reach the next snapshot.
            traj.NextAction = own_action;
            mdp->IncorporateAction({ &traj, 1 });
            last_period = target_period;
        }
    }
}

void DebugEvalMismatchedCdiAcrossInstances(const DynaPlex::System& system)
{
    auto& dp = DynaPlexProvider::Get();

    VarGroup instances_config = VarGroup::LoadFromFile(system.filepath("mdp_config_examples", "dual_sourcing_backlog", "configs", "instances_config.json"));
    std::vector<VarGroup> instances;
    instances_config.Get("instances", instances);

    VarGroup tuned = VarGroup::LoadFromFile(system.filepath("dual_sourcing", "tuning", "tuned_heuristic_params.json"));
    std::vector<VarGroup> tuned_policies;
    tuned.Get("tuned_policies", tuned_policies);

    // The warmup_policy / fallback_policy used throughout the wide-distribution runs -
    // tuned only for instance_1 (S_e=6, S_r=11).
    int64_t warmup_S_e = 6, warmup_S_r = 11;

    VarGroup sim_config;
    sim_config.Add("warmup_periods", int64_t(100));
    sim_config.Add("periods_per_trajectory", int64_t(5000));
    sim_config.Add("number_of_trajectories", int64_t(200));

    double sum_mismatch_gap = 0.0;
    int64_t count = 0;

    for (size_t i = 0; i < instances.size(); i++)
    {
        auto& instance = instances[i];
        auto& tuned_policy = tuned_policies[i];
        std::string name;
        instance.Get("name", name);

        int64_t l_e, l_r;
        instance.Get("l_e", l_e);
        instance.Get("l_r", l_r);
        double mu, sigma, h, b, c_r, c_e;
        instance.Get("mu", mu);
        instance.Get("sigma", sigma);
        instance.Get("h", h);
        instance.Get("b", b);
        instance.Get("c_r", c_r);
        instance.Get("c_e", c_e);

        VarGroup mdp_config;
        mdp_config.Add("id", std::string("dual_sourcing_backlog"));
        mdp_config.Add("K", int64_t(2));
        mdp_config.Add("l_min", l_e);
        mdp_config.Add("l_max", l_r);
        mdp_config.Add("min_h", h); mdp_config.Add("max_h", h);
        mdp_config.Add("min_b", b); mdp_config.Add("max_b", b);
        mdp_config.Add("min_c", c_r); mdp_config.Add("max_c", c_e);
        mdp_config.Add("min_mu", mu); mdp_config.Add("max_mu", mu);
        mdp_config.Add("action_representation", std::string("sequential"));
        mdp_config.Add("discount_factor", 1.0);
        mdp_config.Add("max_order_size", int64_t(270));
        VarGroup fixed_instance;
        fixed_instance.Add("h", h);
        fixed_instance.Add("b", b);
        fixed_instance.Add("mu", mu);
        fixed_instance.Add("sigma", sigma);
        fixed_instance.Add("l_e", l_e);
        fixed_instance.Add("l_r", l_r);
        std::vector<double> costs = {c_e, c_r};
        fixed_instance.Add("costs", costs);
        mdp_config.Add("fixed_instance", fixed_instance);

        DynaPlex::MDP mdp = dp.GetMDP(mdp_config);

        VarGroup cdi_params;
        tuned_policy.Get("CDI", cdi_params);
        int64_t own_S_r, own_S_e;
        cdi_params.Get("S_r", own_S_r);
        cdi_params.Get("S_e", own_S_e);
        double own_cdi_cost = EvaluateCDI(mdp, own_S_r, own_S_e, sim_config);

        double mismatched_cdi_cost = EvaluateCDI(mdp, warmup_S_r, warmup_S_e, sim_config);

        double gap = (mismatched_cdi_cost - own_cdi_cost) / own_cdi_cost * 100.0;
        sum_mismatch_gap += gap;
        count++;

        system << name << ": own_tuned_CDI(S_e=" << own_S_e << ",S_r=" << own_S_r << ")=" << own_cdi_cost
               << " mismatched_CDI(6,11)=" << mismatched_cdi_cost
               << " mismatch_gap=" << gap << "%" << std::endl;
    }

    system << "=== Average mismatch gap (CDI(6,11) vs each instance's own tuned CDI) across " << count << " instances: " << (sum_mismatch_gap / count) << "% ===" << std::endl;
}

void DebugEvalWideAcrossInstances(const DynaPlex::System& system, const std::string& run_path_arg)
{
    auto& dp = DynaPlexProvider::Get();

    VarGroup instances_config = VarGroup::LoadFromFile(system.filepath("mdp_config_examples", "dual_sourcing_backlog", "configs", "instances_config.json"));
    std::vector<VarGroup> instances;
    instances_config.Get("instances", instances);

    VarGroup tuned = VarGroup::LoadFromFile(system.filepath("dual_sourcing", "tuning", "tuned_heuristic_params.json"));
    std::vector<VarGroup> tuned_policies;
    tuned.Get("tuned_policies", tuned_policies);

    std::string run_path = run_path_arg.empty()
        ? "sequential_mdp_config_wide_sequential_dcl_config_minimal_cdiwarmup_20260807_155645"
        : run_path_arg;
    int64_t max_order_size = 270;
    int64_t train_l_max = 5;

    VarGroup sim_config;
    sim_config.Add("warmup_periods", int64_t(100));
    sim_config.Add("periods_per_trajectory", int64_t(5000));
    sim_config.Add("number_of_trajectories", int64_t(200));

    double sum_gap = 0.0;
    int64_t count = 0;

    for (size_t i = 0; i < instances.size(); i++)
    {
        auto& instance = instances[i];
        auto& tuned_policy = tuned_policies[i];
        std::string name;
        instance.Get("name", name);

        int64_t l_e, l_r;
        instance.Get("l_e", l_e);
        instance.Get("l_r", l_r);
        double mu, sigma, h, b, c_r, c_e;
        instance.Get("mu", mu);
        instance.Get("sigma", sigma);
        instance.Get("h", h);
        instance.Get("b", b);
        instance.Get("c_r", c_r);
        instance.Get("c_e", c_e);

        VarGroup mdp_config;
        mdp_config.Add("id", std::string("dual_sourcing_backlog"));
        mdp_config.Add("K", int64_t(2));
        mdp_config.Add("l_min", l_e);
        mdp_config.Add("l_max", train_l_max);
        mdp_config.Add("min_h", h); mdp_config.Add("max_h", h);
        mdp_config.Add("min_b", b); mdp_config.Add("max_b", b);
        mdp_config.Add("min_c", c_r); mdp_config.Add("max_c", c_e);
        mdp_config.Add("min_mu", mu); mdp_config.Add("max_mu", mu);
        mdp_config.Add("action_representation", std::string("sequential"));
        mdp_config.Add("discount_factor", 1.0);
        mdp_config.Add("max_order_size", max_order_size);
        VarGroup fixed_instance;
        fixed_instance.Add("h", h);
        fixed_instance.Add("b", b);
        fixed_instance.Add("mu", mu);
        fixed_instance.Add("sigma", sigma);
        fixed_instance.Add("l_e", l_e);
        fixed_instance.Add("l_r", l_r);
        std::vector<double> costs = {c_e, c_r};
        fixed_instance.Add("costs", costs);
        mdp_config.Add("fixed_instance", fixed_instance);

        DynaPlex::MDP mdp = dp.GetMDP(mdp_config);

        VarGroup cdi_params;
        tuned_policy.Get("CDI", cdi_params);
        double tuned_cdi_cost;
        cdi_params.Get("cost", tuned_cdi_cost);
        int64_t S_r, S_e;
        cdi_params.Get("S_r", S_r);
        cdi_params.Get("S_e", S_e);

        VarGroup policy_config;
        policy_config.Add("id", std::string("cdi"));
        policy_config.Add("S_r", S_r);
        policy_config.Add("S_e", S_e);
        auto cdi_policy = mdp->GetPolicy(policy_config);
        auto& dp2 = DynaPlexProvider::Get();
        auto comparer = dp2.GetPolicyComparer(mdp, sim_config);
        auto cdi_result = comparer.Assess(cdi_policy);
        double cdi_cost_here;
        cdi_result.Get("mean", cdi_cost_here);

        auto full_path = system.filepath("dual_sourcing", "runs", run_path, "policy_final");
        auto gca_policy = dp2.LoadPolicy(mdp, full_path);
        auto gca_result = comparer.Assess(gca_policy);
        double gca_cost;
        gca_result.Get("mean", gca_cost);

        double gap = (gca_cost - cdi_cost_here) / cdi_cost_here * 100.0;
        sum_gap += gap;
        count++;

        system << name << ": CDI(tuned_json)=" << tuned_cdi_cost
               << " CDI(recomputed)=" << cdi_cost_here
               << " GCA-DS(wide)=" << gca_cost
               << " gap=" << gap << "%" << std::endl;
    }

    system << "=== Average gap vs CDI across " << count << " instances: " << (sum_gap / count) << "% ===" << std::endl;
}

void DebugEvalCdiWarmupVsCdi(const DynaPlex::System& system)
{
    auto& dp = DynaPlexProvider::Get();

    VarGroup instance;
    instance.Add("name", std::string("instance_1"));
    instance.Add("l_e", int64_t(1));
    instance.Add("l_r", int64_t(2));
    instance.Add("mu", 4.0);
    instance.Add("sigma", 2.0);
    instance.Add("h", 1.0);
    instance.Add("b", 9.0);
    instance.Add("c_r", 0.0);
    instance.Add("c_e", 1.0);

    VarGroup sim_config;
    sim_config.Add("warmup_periods", int64_t(100));
    sim_config.Add("periods_per_trajectory", int64_t(5000));
    sim_config.Add("number_of_trajectories", int64_t(1000));

    int64_t tuned_S_r = 11, tuned_S_e = 6;
    double tuned_cdi_cost = 10.551646200000006;

    struct RunSpec { std::string label; std::string run_path; int64_t max_order_size; };
    std::vector<RunSpec> runs = {
        {"Stage1 (max_order_size=20, CDI-warmup)", "sequential_mdp_config_minimal_dcl_config_minimal_cdiwarmup_20260807_150230", 20},
        {"Stage2 (max_order_size=270, CDI-warmup)", "sequential_mdp_config_minimal_fullaction_dcl_config_minimal_cdiwarmup_20260807_153708", 270},
        {"Stage5 (max_order_size=270, CDI-warmup, Num_Promising_Actions=60)", "sequential_mdp_config_minimal_fullaction_dcl_config_minimal_cdiwarmup_wideactions_20260807_163351", 270},
        {"Stage6 (max_order_size=270, CDI-warmup, argmax-gated fallback)", "sequential_mdp_config_minimal_fullaction_dcl_config_minimal_cdiwarmup_fallback_20260807_185323", 270},
    };

    for (auto& run : runs)
    {
        VarGroup mdp_config;
        mdp_config.Add("id", std::string("dual_sourcing_backlog"));
        mdp_config.Add("K", int64_t(2));
        mdp_config.Add("l_min", int64_t(1));
        mdp_config.Add("l_max", int64_t(2));
        mdp_config.Add("min_h", 1.0); mdp_config.Add("max_h", 1.0);
        mdp_config.Add("min_b", 9.0); mdp_config.Add("max_b", 9.0);
        mdp_config.Add("min_c", 0.0); mdp_config.Add("max_c", 1.0);
        mdp_config.Add("min_mu", 4.0); mdp_config.Add("max_mu", 4.0);
        mdp_config.Add("action_representation", std::string("sequential"));
        mdp_config.Add("discount_factor", 1.0);
        mdp_config.Add("max_order_size", run.max_order_size);
        VarGroup fixed_instance;
        fixed_instance.Add("h", 1.0);
        fixed_instance.Add("b", 9.0);
        fixed_instance.Add("mu", 4.0);
        fixed_instance.Add("sigma", 2.0);
        fixed_instance.Add("l_e", int64_t(1));
        fixed_instance.Add("l_r", int64_t(2));
        std::vector<double> costs = {1.0, 0.0};
        fixed_instance.Add("costs", costs);
        mdp_config.Add("fixed_instance", fixed_instance);

        DynaPlex::MDP mdp = dp.GetMDP(mdp_config);

        double cdi_cost = EvaluateCDI(mdp, tuned_S_r, tuned_S_e, sim_config);

        auto full_path = system.filepath("dual_sourcing", "runs", run.run_path, "policy_final");
        auto policy = dp.LoadPolicy(mdp, full_path);
        auto comparer = dp.GetPolicyComparer(mdp, sim_config);
        auto result = comparer.Assess(policy);
        double gca_cost_total;
        result.Get("mean", gca_cost_total);
        int64_t periods;
        sim_config.Get("periods_per_trajectory", periods);
        double gca_cost = gca_cost_total / static_cast<double>(periods);

        double gap_vs_tuned = (gca_cost - tuned_cdi_cost) / tuned_cdi_cost * 100.0;
        double gap_vs_local_cdi = (gca_cost - cdi_cost) / cdi_cost * 100.0;

        system << "=== " << run.label << " ===" << std::endl;
        system << "  CDI (tuned, S_e=" << tuned_S_e << ", S_r=" << tuned_S_r << ") cost, recomputed here: " << cdi_cost << std::endl;
        system << "  CDI (tuned_heuristic_params.json) cost: " << tuned_cdi_cost << std::endl;
        system << "  GCA-DS cost: " << gca_cost << std::endl;
        system << "  Gap vs tuned CDI (json): " << gap_vs_tuned << "%" << std::endl;
        system << "  Gap vs CDI (recomputed here, same eval run): " << gap_vs_local_cdi << "%" << std::endl;
    }
}

// Tests whether a trained network actually *uses* the instance-parameter input features, or
// whether it's insensitive to them (i.e. it never received any training signal that varied
// them, because it was trained on a single fixed instance - a network can memorize good
// state-dependent behavior for one instance's dynamics without ever learning to condition on
// mu/b/l/c at all). Feeds the SAME hand-built feature vector, differing only in the
// instance-parameter positions, directly to the network's forward pass, bypassing MDP/state
// construction entirely so there is no ambiguity about what varied.
void DebugFeatureSensitivityProbe(const DynaPlex::System& system, const std::string& run_path)
{
    auto& dp = DynaPlexProvider::Get();

    // MDP matching the run's training structure (sequential, l_max=2, instance_1-shaped).
    VarGroup mdp_config;
    mdp_config.Add("id", std::string("dual_sourcing_backlog"));
    mdp_config.Add("K", int64_t(2));
    mdp_config.Add("l_min", int64_t(1));
    mdp_config.Add("l_max", int64_t(2));
    mdp_config.Add("min_h", 1.0); mdp_config.Add("max_h", 1.0);
    mdp_config.Add("min_b", 9.0); mdp_config.Add("max_b", 9.0);
    mdp_config.Add("min_c", 0.0); mdp_config.Add("max_c", 1.0);
    mdp_config.Add("min_mu", 4.0); mdp_config.Add("max_mu", 4.0);
    mdp_config.Add("action_representation", std::string("sequential"));
    mdp_config.Add("discount_factor", 1.0);
    mdp_config.Add("max_order_size", int64_t(270));
    VarGroup fixed_instance;
    fixed_instance.Add("h", 1.0);
    fixed_instance.Add("b", 9.0);
    fixed_instance.Add("mu", 4.0);
    fixed_instance.Add("sigma", 2.0);
    fixed_instance.Add("l_e", int64_t(1));
    fixed_instance.Add("l_r", int64_t(2));
    std::vector<double> costs = {1.0, 0.0};
    fixed_instance.Add("costs", costs);
    mdp_config.Add("fixed_instance", fixed_instance);

    DynaPlex::MDP mdp = dp.GetMDP(mdp_config);

    auto full_path = system.filepath("dual_sourcing", "runs", run_path, "policy_final");
    auto policy = dp.LoadPolicy(mdp, full_path);
    auto as_nn_policy = std::dynamic_pointer_cast<NN_Policy>(policy);
    if (!as_nn_policy)
        throw DynaPlex::Error("DebugFeatureSensitivityProbe: policy is not an NN_Policy.");

    // Feature layout for this MDP (K=2, l_max=2), matching GetFeatures() exactly:
    // [state_vector[0]/scale, state_vector[1]/scale, total_inv/scale, mu_hat, sigma_hat,
    //  h, b/(b+h), c[0], l[0], c[1], l[1], current_source]
    struct Point { std::string label; double mu_hat, b, h; int64_t l0, l1; };
    std::vector<Point> points = {
        {"instance_1's real params (mu=4,b=9,l=1/2)", 4.0, 9.0, 1.0, 1, 2},
        {"mu_hat perturbed to 12 (3x real)",           12.0, 9.0, 1.0, 1, 2},
        {"b perturbed to 99 (11x real)",                4.0, 99.0, 1.0, 1, 2},
        {"l perturbed to 1/5 (much longer)",            4.0, 9.0, 1.0, 1, 5},
        {"all three perturbed together",               12.0, 99.0, 1.0, 1, 5},
    };

    // A representative mid-trajectory state: modest pipeline, some inventory, source 0.
    double sv0 = 2.0, sv1 = 3.0, total_inv = 5.0, current_source = 0.0;
    double c0 = 1.0, c1 = 0.0;

    for (auto& pt : points)
    {
        double scale = std::max(pt.mu_hat, 0.1);
        std::vector<float> feats = {
            static_cast<float>(sv0 / scale),
            static_cast<float>(sv1 / scale),
            static_cast<float>(total_inv / scale),
            static_cast<float>(pt.mu_hat),
            static_cast<float>(pt.mu_hat / 2.0), // sigma_hat, kept proportional
            static_cast<float>(pt.h),
            static_cast<float>(pt.b / (pt.b + pt.h)),
            static_cast<float>(c0),
            static_cast<float>(pt.l0),
            static_cast<float>(c1),
            static_cast<float>(pt.l1),
            static_cast<float>(current_source)
        };

        torch::Tensor input = torch::from_blob(feats.data(), { 1, static_cast<int64_t>(feats.size()) }, torch::kFloat32).clone();
        torch::NoGradGuard no_grad;
        torch::Tensor output = as_nn_policy->neural_network->forward(input);
        int64_t argmax_action = output.argmax(1).item<int64_t>();

        system << pt.label << ": argmax_action=" << argmax_action << std::endl;
    }
}

// Tunes BaseStockPolicy's "S" for a single K=1 instance via LineSearch, for use as the
// warmup/fallback heuristic and as a benchmark in K=1 dual_sourcing_backlog experiments.
void DebugTuneK1BaseStock(const DynaPlex::System& system, double mu, double sigma, double h, double b, int64_t l)
{
    auto& dp = DynaPlexProvider::Get();

    // K=1 fixed_instance mode isn't supported by mdp.cpp (its constructor hardcodes reading
    // both l_e and l_r regardless of K) - use a degenerate (effectively fixed) general-mode
    // range instead, which works generically for any K.
    VarGroup mdp_config;
    mdp_config.Add("id", std::string("dual_sourcing_backlog"));
    mdp_config.Add("K", int64_t(1));
    mdp_config.Add("l_min", l);
    mdp_config.Add("l_max", l + 1);
    mdp_config.Add("min_h", h); mdp_config.Add("max_h", h);
    mdp_config.Add("min_b", b); mdp_config.Add("max_b", b);
    mdp_config.Add("min_c", 1.0); mdp_config.Add("max_c", 1.001);
    mdp_config.Add("min_mu", mu); mdp_config.Add("max_mu", mu);
    mdp_config.Add("action_representation", std::string("sequential"));
    mdp_config.Add("discount_factor", 1.0);
    mdp_config.Add("max_order_size", int64_t(270));

    DynaPlex::MDP mdp = dp.GetMDP(mdp_config);

    VarGroup tuning_config;
    tuning_config.Add("warmup_periods", int64_t(100));
    tuning_config.Add("periods_per_trajectory", int64_t(5000));
    tuning_config.Add("number_of_trajectories", int64_t(500));

    double fractile = b / (b + h);
    int64_t start_val = NewsvendorFractile(mu, sigma, fractile, l);
    int64_t max_val = start_val * 4 + 20;

    VarGroup policy_config;
    policy_config.Add("id", std::string("base_stock"));
    int64_t best_S = LineSearch(mdp, tuning_config, policy_config, "S", start_val, max_val);

    policy_config.Set("S", best_S);
    auto policy = mdp->GetPolicy(policy_config);
    double cost = EvaluatePolicyTuning(mdp, policy, tuning_config);

    system << "K=1 instance mu=" << mu << " sigma=" << sigma << " h=" << h << " b=" << b << " l=" << l
           << ": best_S=" << best_S << " cost=" << cost << std::endl;
}

// Sanity check: evaluates AdaptiveBaseStockPolicy (the behavior policy that drove gen1 of the
// K=1 narrow-mu run) against tuned fixed-S base_stock on the same instance, to check whether the
// behavior policy itself was sound (as opposed to the divergence originating in DCL training).
void DebugEvalAdaptiveBaseStock(const DynaPlex::System& system, double mu, double sigma, double h, double b, int64_t l, int64_t tuned_S)
{
    auto& dp = DynaPlexProvider::Get();

    VarGroup mdp_config;
    mdp_config.Add("id", std::string("dual_sourcing_backlog"));
    mdp_config.Add("K", int64_t(1));
    mdp_config.Add("l_min", l);
    mdp_config.Add("l_max", l + 1);
    mdp_config.Add("min_h", h); mdp_config.Add("max_h", h);
    mdp_config.Add("min_b", b); mdp_config.Add("max_b", b);
    mdp_config.Add("min_c", 1.0); mdp_config.Add("max_c", 1.001);
    mdp_config.Add("min_mu", mu); mdp_config.Add("max_mu", mu);
    mdp_config.Add("action_representation", std::string("sequential"));
    mdp_config.Add("discount_factor", 1.0);
    mdp_config.Add("max_order_size", int64_t(60));

    DynaPlex::MDP mdp = dp.GetMDP(mdp_config);

    VarGroup sim_config;
    sim_config.Add("warmup_periods", int64_t(100));
    sim_config.Add("periods_per_trajectory", int64_t(5000));
    sim_config.Add("number_of_trajectories", int64_t(500));

    VarGroup bs_config;
    bs_config.Add("id", std::string("base_stock"));
    bs_config.Add("S", tuned_S);
    auto bs_policy = mdp->GetPolicy(bs_config);
    double bs_cost = EvaluatePolicyTuning(mdp, bs_policy, sim_config);

    VarGroup adaptive_config;
    adaptive_config.Add("id", std::string("adaptive_base_stock"));
    auto adaptive_policy = mdp->GetPolicy(adaptive_config);
    double adaptive_cost = EvaluatePolicyTuning(mdp, adaptive_policy, sim_config);

    double gap = (adaptive_cost - bs_cost) / bs_cost * 100.0;
    system << "base_stock (S=" << tuned_S << ") cost: " << bs_cost << std::endl;
    system << "adaptive_base_stock cost: " << adaptive_cost << std::endl;
    system << "Gap vs base_stock: " << gap << "%" << std::endl;
}

// K=2 analogue of DebugTuneK1BaseStock: tunes CDI(S_r, S_e) on a fixed K=2 instance
// via fixed_instance mode (properly supported for K=2, unlike K=1's degenerate-range workaround).
void DebugTuneK2CDI(const DynaPlex::System& system, double mu, double sigma, double h, double b,
    int64_t l_e, int64_t l_r, double c_e, int64_t max_order_size, double inventory_cap_multiplier,
    double c_r)
{
    auto& dp = DynaPlexProvider::Get();

    VarGroup mdp_config;
    mdp_config.Add("id", std::string("dual_sourcing_backlog"));
    mdp_config.Add("K", int64_t(2));
    mdp_config.Add("l_min", l_e);
    mdp_config.Add("l_max", l_r);
    mdp_config.Add("min_h", h); mdp_config.Add("max_h", h);
    mdp_config.Add("min_b", b); mdp_config.Add("max_b", b);
    mdp_config.Add("min_c", c_r); mdp_config.Add("max_c", c_e);
    mdp_config.Add("min_mu", mu); mdp_config.Add("max_mu", mu);
    mdp_config.Add("action_representation", std::string("sequential"));
    mdp_config.Add("discount_factor", 1.0);
    mdp_config.Add("max_order_size", max_order_size);
    if (inventory_cap_multiplier > 0.0)
        mdp_config.Add("inventory_cap_multiplier", inventory_cap_multiplier);

    VarGroup fixed_instance;
    fixed_instance.Add("h", h);
    fixed_instance.Add("b", b);
    fixed_instance.Add("mu", mu);
    fixed_instance.Add("sigma", sigma);
    fixed_instance.Add("l_e", l_e);
    fixed_instance.Add("l_r", l_r);
    std::vector<double> costs = { c_e, c_r };
    fixed_instance.Add("costs", costs);
    mdp_config.Add("fixed_instance", fixed_instance);

    DynaPlex::MDP mdp = dp.GetMDP(mdp_config);

    VarGroup tuning_config;
    tuning_config.Add("warmup_periods", int64_t(100));
    tuning_config.Add("periods_per_trajectory", int64_t(5000));
    tuning_config.Add("number_of_trajectories", int64_t(500));

    auto result = TuneCDI(mdp, tuning_config, mu, sigma, b, h, l_r, l_e, max_order_size);
    system << "K=2 instance mu=" << mu << " sigma=" << sigma << " h=" << h << " b=" << b
           << " l_e=" << l_e << " l_r=" << l_r << " c_e=" << c_e << ": " << result.Dump() << std::endl;
}

// Evaluates a trained K=2 policy against tuned CDI, on the same fixed instance. struct_l_min/
// struct_l_max fix the MDP's structural state_vector/feature dimensionality (must match whatever
// range the network was actually trained on); l_e/l_r are the REALIZED lead times of the specific
// held-out instance being tested, pinned via fixed_instance. If struct_l_min/struct_l_max are 0,
// they default to l_e/l_r (matching the old fixed-instance-only behavior).
void DebugEvalK2FixedVsCDI(const DynaPlex::System& system, const std::string& run_path,
    double mu, double sigma, double h, double b, int64_t l_e, int64_t l_r, double c_e,
    int64_t max_order_size, int64_t S_r, int64_t S_e, double inventory_cap_multiplier,
    int64_t struct_l_min, int64_t struct_l_max, double c_r, const std::string& action_representation,
    int64_t number_of_trajectories, int64_t periods_per_trajectory)
{
    auto& dp = DynaPlexProvider::Get();
    if (struct_l_min <= 0) struct_l_min = l_e;
    if (struct_l_max <= 0) struct_l_max = l_r;

    VarGroup mdp_config;
    mdp_config.Add("id", std::string("dual_sourcing_backlog"));
    mdp_config.Add("K", int64_t(2));
    mdp_config.Add("l_min", struct_l_min);
    mdp_config.Add("l_max", struct_l_max);
    mdp_config.Add("min_h", h); mdp_config.Add("max_h", h);
    mdp_config.Add("min_b", b); mdp_config.Add("max_b", b);
    mdp_config.Add("min_c", c_r); mdp_config.Add("max_c", c_e);
    mdp_config.Add("min_mu", mu); mdp_config.Add("max_mu", mu);
    mdp_config.Add("action_representation", action_representation);
    mdp_config.Add("discount_factor", 1.0);
    mdp_config.Add("max_order_size", max_order_size);
    if (inventory_cap_multiplier > 0.0)
        mdp_config.Add("inventory_cap_multiplier", inventory_cap_multiplier);

    VarGroup fixed_instance;
    fixed_instance.Add("h", h);
    fixed_instance.Add("b", b);
    fixed_instance.Add("mu", mu);
    fixed_instance.Add("sigma", sigma);
    fixed_instance.Add("l_e", l_e);
    fixed_instance.Add("l_r", l_r);
    std::vector<double> costs = { c_e, c_r };
    fixed_instance.Add("costs", costs);
    mdp_config.Add("fixed_instance", fixed_instance);

    DynaPlex::MDP mdp = dp.GetMDP(mdp_config);

    VarGroup sim_config;
    sim_config.Add("warmup_periods", int64_t(100));
    sim_config.Add("periods_per_trajectory", periods_per_trajectory);
    sim_config.Add("number_of_trajectories", number_of_trajectories);

    VarGroup cdi_config;
    cdi_config.Add("id", std::string("cdi"));
    cdi_config.Add("S_r", S_r);
    cdi_config.Add("S_e", S_e);
    auto cdi_policy = mdp->GetPolicy(cdi_config);
    double cdi_cost = EvaluatePolicyTuning(mdp, cdi_policy, sim_config);

    auto full_path = system.filepath("dual_sourcing", "runs", run_path, "policy_final");
    auto network_policy = dp.LoadPolicy(mdp, full_path);
    double gca_cost = EvaluatePolicyTuning(mdp, network_policy, sim_config);

    double gap = (gca_cost - cdi_cost) / cdi_cost * 100.0;

    system << "CDI (S_r=" << S_r << ", S_e=" << S_e << ") cost: " << cdi_cost << std::endl;
    system << "GCA-DS cost: " << gca_cost << std::endl;
    system << "Gap vs CDI: " << gap << "%" << std::endl;
}

// Like DebugEvalK2FixedVsCDI, but loads a specific generation's checkpoint (dcl_policy_gen<N>)
// instead of the run's final policy - enabling live per-generation evaluation while a run is
// still in progress. Per-generation checkpoints are saved under a path keyed by the *training*
// mdp's Identifier() (see PolicyTrainer::PathToPolicy), not under the run's own timestamped
// directory, so this reads run_info.json (written by dual_sourcing_gca at launch) to recover
// that identifier rather than requiring the caller to know DynaPlex's internal hashing scheme.
void DebugEvalK2GenVsCDI(const DynaPlex::System& system, const std::string& run_path, int64_t generation,
    double mu, double sigma, double h, double b, int64_t l_e, int64_t l_r, double c_e,
    int64_t max_order_size, int64_t S_r, int64_t S_e, double inventory_cap_multiplier,
    int64_t struct_l_min, int64_t struct_l_max, double c_r, const std::string& action_representation,
    int64_t number_of_trajectories, int64_t periods_per_trajectory)
{
    auto& dp = DynaPlexProvider::Get();
    if (struct_l_min <= 0) struct_l_min = l_e;
    if (struct_l_max <= 0) struct_l_max = l_r;

    VarGroup run_info = VarGroup::LoadFromFile(system.filepath("dual_sourcing", "runs", run_path, "run_info.json"));
    std::string mdp_identifier;
    run_info.Get("mdp_identifier", mdp_identifier);

    VarGroup mdp_config;
    mdp_config.Add("id", std::string("dual_sourcing_backlog"));
    mdp_config.Add("K", int64_t(2));
    mdp_config.Add("l_min", struct_l_min);
    mdp_config.Add("l_max", struct_l_max);
    mdp_config.Add("min_h", h); mdp_config.Add("max_h", h);
    mdp_config.Add("min_b", b); mdp_config.Add("max_b", b);
    mdp_config.Add("min_c", c_r); mdp_config.Add("max_c", c_e);
    mdp_config.Add("min_mu", mu); mdp_config.Add("max_mu", mu);
    mdp_config.Add("action_representation", action_representation);
    mdp_config.Add("discount_factor", 1.0);
    mdp_config.Add("max_order_size", max_order_size);
    if (inventory_cap_multiplier > 0.0)
        mdp_config.Add("inventory_cap_multiplier", inventory_cap_multiplier);

    VarGroup fixed_instance;
    fixed_instance.Add("h", h);
    fixed_instance.Add("b", b);
    fixed_instance.Add("mu", mu);
    fixed_instance.Add("sigma", sigma);
    fixed_instance.Add("l_e", l_e);
    fixed_instance.Add("l_r", l_r);
    std::vector<double> costs = { c_e, c_r };
    fixed_instance.Add("costs", costs);
    mdp_config.Add("fixed_instance", fixed_instance);

    DynaPlex::MDP mdp = dp.GetMDP(mdp_config);

    VarGroup sim_config;
    sim_config.Add("warmup_periods", int64_t(100));
    sim_config.Add("periods_per_trajectory", periods_per_trajectory);
    sim_config.Add("number_of_trajectories", number_of_trajectories);

    VarGroup cdi_config;
    cdi_config.Add("id", std::string("cdi"));
    cdi_config.Add("S_r", S_r);
    cdi_config.Add("S_e", S_e);
    auto cdi_policy = mdp->GetPolicy(cdi_config);
    double cdi_cost = EvaluatePolicyTuning(mdp, cdi_policy, sim_config);

    auto full_path = system.filepath(mdp_identifier, "dcl_policy_gen" + std::to_string(generation));
    auto network_policy = dp.LoadPolicy(mdp, full_path);
    double gca_cost = EvaluatePolicyTuning(mdp, network_policy, sim_config);

    double gap = (gca_cost - cdi_cost) / cdi_cost * 100.0;

    system << "GEN=" << generation << " CDI cost: " << cdi_cost
        << " GCA-DS cost: " << gca_cost << " Gap vs CDI: " << gap << "%" << std::endl;
}

// Evaluates a trained K=1 policy against tuned base_stock, on the same fixed instance the
// tune_k1_base_stock diagnostic tunes S for.
void DebugEvalK1FixedVsBaseStock(const DynaPlex::System& system, const std::string& run_path,
    double mu, double sigma, double h, double b, int64_t l, int64_t tuned_S, int64_t max_order_size,
    double inventory_cap_multiplier)
{
    auto& dp = DynaPlexProvider::Get();

    VarGroup mdp_config;
    mdp_config.Add("id", std::string("dual_sourcing_backlog"));
    mdp_config.Add("K", int64_t(1));
    mdp_config.Add("l_min", l);
    mdp_config.Add("l_max", l + 1);
    mdp_config.Add("min_h", h); mdp_config.Add("max_h", h);
    mdp_config.Add("min_b", b); mdp_config.Add("max_b", b);
    mdp_config.Add("min_c", 1.0); mdp_config.Add("max_c", 1.001);
    mdp_config.Add("min_mu", mu); mdp_config.Add("max_mu", mu);
    mdp_config.Add("action_representation", std::string("sequential"));
    mdp_config.Add("discount_factor", 1.0);
    mdp_config.Add("max_order_size", max_order_size);
    if (inventory_cap_multiplier > 0.0)
        mdp_config.Add("inventory_cap_multiplier", inventory_cap_multiplier);

    DynaPlex::MDP mdp = dp.GetMDP(mdp_config);

    VarGroup sim_config;
    sim_config.Add("warmup_periods", int64_t(100));
    sim_config.Add("periods_per_trajectory", int64_t(5000));
    sim_config.Add("number_of_trajectories", int64_t(1000));

    VarGroup bs_config;
    bs_config.Add("id", std::string("base_stock"));
    bs_config.Add("S", tuned_S);
    auto bs_policy = mdp->GetPolicy(bs_config);
    double bs_cost = EvaluatePolicyTuning(mdp, bs_policy, sim_config);

    auto full_path = system.filepath("dual_sourcing", "runs", run_path, "policy_final");
    auto gca_policy = dp.LoadPolicy(mdp, full_path);
    double gca_cost = EvaluatePolicyTuning(mdp, gca_policy, sim_config);

    double gap = (gca_cost - bs_cost) / bs_cost * 100.0;
    system << "base_stock (S=" << tuned_S << ") cost: " << bs_cost << std::endl;
    system << "GCA-DS cost: " << gca_cost << std::endl;
    system << "Gap vs base_stock: " << gap << "%" << std::endl;
}

// Traces actions chosen by a trained K=2 policy vs CDI, at both decision stages
// (source 0 = expedited, source 1 = regular) per snapshot period, along a CDI-driven trajectory.
void DebugTraceK2Actions(const DynaPlex::System& system, const std::string& run_path,
    double mu, double sigma, double h, double b, int64_t l_e, int64_t l_r, double c_e,
    int64_t max_order_size, int64_t S_r, int64_t S_e)
{
    auto& dp = DynaPlexProvider::Get();

    VarGroup mdp_config;
    mdp_config.Add("id", std::string("dual_sourcing_backlog"));
    mdp_config.Add("K", int64_t(2));
    mdp_config.Add("l_min", l_e);
    mdp_config.Add("l_max", l_r);
    mdp_config.Add("min_h", h); mdp_config.Add("max_h", h);
    mdp_config.Add("min_b", b); mdp_config.Add("max_b", b);
    mdp_config.Add("min_c", 0.0); mdp_config.Add("max_c", c_e);
    mdp_config.Add("min_mu", mu); mdp_config.Add("max_mu", mu);
    mdp_config.Add("action_representation", std::string("sequential"));
    mdp_config.Add("discount_factor", 1.0);
    mdp_config.Add("max_order_size", max_order_size);

    VarGroup fixed_instance;
    fixed_instance.Add("h", h);
    fixed_instance.Add("b", b);
    fixed_instance.Add("mu", mu);
    fixed_instance.Add("sigma", sigma);
    fixed_instance.Add("l_e", l_e);
    fixed_instance.Add("l_r", l_r);
    std::vector<double> costs = { c_e, 0.0 };
    fixed_instance.Add("costs", costs);
    mdp_config.Add("fixed_instance", fixed_instance);

    DynaPlex::MDP mdp = dp.GetMDP(mdp_config);

    VarGroup cdi_config;
    cdi_config.Add("id", std::string("cdi"));
    cdi_config.Add("S_r", S_r);
    cdi_config.Add("S_e", S_e);
    auto cdi_policy = mdp->GetPolicy(cdi_config);

    auto full_path = system.filepath("dual_sourcing", "runs", run_path, "policy_final");
    auto network_policy = dp.LoadPolicy(mdp, full_path);

    DynaPlex::Trajectory traj(0);
    traj.RNGProvider.SeedEventStreams(false, 15112017, 0);
    mdp->InitiateState({ &traj, 1 });

    std::vector<int64_t> snapshot_periods = { 50, 100, 150, 200, 250, 300 };
    for (int64_t target_period : snapshot_periods)
    {
        while (traj.PeriodCount < target_period)
        {
            if (mdp->IncorporateUntilAction({ &traj, 1 }, target_period))
            {
                if (traj.PeriodCount >= target_period)
                    break;
                mdp->IncorporateAction({ &traj, 1 }, cdi_policy);
            }
        }

        // Source 0 (expedited) decision.
        cdi_policy->SetAction({ &traj, 1 });
        int64_t cdi_q_e = traj.NextAction;
        network_policy->SetAction({ &traj, 1 });
        int64_t net_q_e = traj.NextAction;

        mdp->IncorporateAction({ &traj, 1 }, cdi_policy);

        // Source 1 (regular) decision - state now reflects cdi_q_e already placed.
        cdi_policy->SetAction({ &traj, 1 });
        int64_t cdi_q_r = traj.NextAction;
        network_policy->SetAction({ &traj, 1 });
        int64_t net_q_r = traj.NextAction;

        system << "period " << target_period
               << ": CDI(q_e=" << cdi_q_e << ", q_r=" << cdi_q_r << ")"
               << "  Network(q_e=" << net_q_e << ", q_r=" << net_q_r << ")" << std::endl;

        mdp->IncorporateAction({ &traj, 1 }, cdi_policy);
    }
}

// Traces actions chosen by a trained K=1 policy vs base_stock, at several points along a
// base_stock-driven trajectory.
void DebugTraceK1Actions(const DynaPlex::System& system, const std::string& run_path,
    double mu, double sigma, double h, double b, int64_t l, int64_t tuned_S, int64_t max_order_size)
{
    auto& dp = DynaPlexProvider::Get();

    VarGroup mdp_config;
    mdp_config.Add("id", std::string("dual_sourcing_backlog"));
    mdp_config.Add("K", int64_t(1));
    mdp_config.Add("l_min", l);
    mdp_config.Add("l_max", l + 1);
    mdp_config.Add("min_h", h); mdp_config.Add("max_h", h);
    mdp_config.Add("min_b", b); mdp_config.Add("max_b", b);
    mdp_config.Add("min_c", 1.0); mdp_config.Add("max_c", 1.001);
    mdp_config.Add("min_mu", mu); mdp_config.Add("max_mu", mu);
    mdp_config.Add("action_representation", std::string("sequential"));
    mdp_config.Add("discount_factor", 1.0);
    mdp_config.Add("max_order_size", max_order_size);

    DynaPlex::MDP mdp = dp.GetMDP(mdp_config);

    VarGroup bs_config;
    bs_config.Add("id", std::string("base_stock"));
    bs_config.Add("S", tuned_S);
    auto bs_policy = mdp->GetPolicy(bs_config);

    auto full_path = system.filepath("dual_sourcing", "runs", run_path, "policy_final");
    auto network_policy = dp.LoadPolicy(mdp, full_path);

    DynaPlex::Trajectory traj(0);
    traj.RNGProvider.SeedEventStreams(false, 15112017, 0);
    mdp->InitiateState({ &traj, 1 });

    std::vector<int64_t> snapshot_periods = { 50, 100, 150, 200, 250, 300 };
    for (int64_t target_period : snapshot_periods)
    {
        while (traj.PeriodCount < target_period)
        {
            if (mdp->IncorporateUntilAction({ &traj, 1 }, target_period))
            {
                if (traj.PeriodCount >= target_period)
                    break;
                mdp->IncorporateAction({ &traj, 1 }, bs_policy);
            }
        }
        bs_policy->SetAction({ &traj, 1 });
        int64_t bs_action = traj.NextAction;
        network_policy->SetAction({ &traj, 1 });
        int64_t network_action = traj.NextAction;

        system << "period " << target_period << ": base_stock_action=" << bs_action
               << " network_action=" << network_action << std::endl;

        mdp->IncorporateAction({ &traj, 1 }, bs_policy);
    }
}

int main(int argc, char* argv[])
{
    auto& dp = DynaPlexProvider::Get();
    auto& system = dp.System();

    if (argc > 1 && std::string(argv[1]) == "eval_cdiwarmup")
    {
        DebugEvalCdiWarmupVsCdi(system);
        return 0;
    }
    if (argc > 1 && std::string(argv[1]) == "eval_wide")
    {
        std::string run_path_arg = argc > 2 ? std::string(argv[2]) : std::string();
        DebugEvalWideAcrossInstances(system, run_path_arg);
        return 0;
    }
    if (argc > 1 && std::string(argv[1]) == "eval_mismatch")
    {
        DebugEvalMismatchedCdiAcrossInstances(system);
        return 0;
    }
    if (argc > 1 && std::string(argv[1]) == "feature_sensitivity")
    {
        std::string run_path_arg = argc > 2 ? std::string(argv[2]) : std::string();
        DebugFeatureSensitivityProbe(system, run_path_arg);
        return 0;
    }
    if (argc > 1 && std::string(argv[1]) == "tune_k1_base_stock")
    {
        double mu = argc > 2 ? std::stod(argv[2]) : 4.0;
        double sigma = argc > 3 ? std::stod(argv[3]) : 2.0;
        double h = argc > 4 ? std::stod(argv[4]) : 1.0;
        double b = argc > 5 ? std::stod(argv[5]) : 9.0;
        int64_t l = argc > 6 ? std::stoll(argv[6]) : 2;
        DebugTuneK1BaseStock(system, mu, sigma, h, b, l);
        return 0;
    }
    if (argc > 1 && std::string(argv[1]) == "eval_adaptive_bs")
    {
        double mu = argc > 2 ? std::stod(argv[2]) : 4.0;
        double sigma = argc > 3 ? std::stod(argv[3]) : 2.0;
        double h = argc > 4 ? std::stod(argv[4]) : 1.0;
        double b = argc > 5 ? std::stod(argv[5]) : 9.0;
        int64_t l = argc > 6 ? std::stoll(argv[6]) : 2;
        int64_t tuned_S = argc > 7 ? std::stoll(argv[7]) : 17;
        DebugEvalAdaptiveBaseStock(system, mu, sigma, h, b, l, tuned_S);
        return 0;
    }
    if (argc > 1 && std::string(argv[1]) == "tune_k2_cdi")
    {
        double mu = argc > 2 ? std::stod(argv[2]) : 4.0;
        double sigma = argc > 3 ? std::stod(argv[3]) : 2.0;
        double h = argc > 4 ? std::stod(argv[4]) : 1.0;
        double b = argc > 5 ? std::stod(argv[5]) : 9.0;
        int64_t l_e = argc > 6 ? std::stoll(argv[6]) : 1;
        int64_t l_r = argc > 7 ? std::stoll(argv[7]) : 2;
        double c_e = argc > 8 ? std::stod(argv[8]) : 1.0;
        int64_t max_order_size = argc > 9 ? std::stoll(argv[9]) : 40;
        double inventory_cap_multiplier = argc > 10 ? std::stod(argv[10]) : 0.0;
        double c_r = argc > 11 ? std::stod(argv[11]) : 0.0;
        DebugTuneK2CDI(system, mu, sigma, h, b, l_e, l_r, c_e, max_order_size, inventory_cap_multiplier, c_r);
        return 0;
    }
    if (argc > 1 && std::string(argv[1]) == "eval_k2_fixed")
    {
        std::string run_path = argc > 2 ? std::string(argv[2]) : std::string();
        double mu = argc > 3 ? std::stod(argv[3]) : 4.0;
        double sigma = argc > 4 ? std::stod(argv[4]) : 2.0;
        double h = argc > 5 ? std::stod(argv[5]) : 1.0;
        double b = argc > 6 ? std::stod(argv[6]) : 9.0;
        int64_t l_e = argc > 7 ? std::stoll(argv[7]) : 1;
        int64_t l_r = argc > 8 ? std::stoll(argv[8]) : 2;
        double c_e = argc > 9 ? std::stod(argv[9]) : 1.0;
        int64_t max_order_size = argc > 10 ? std::stoll(argv[10]) : 40;
        int64_t S_r = argc > 11 ? std::stoll(argv[11]) : 17;
        int64_t S_e = argc > 12 ? std::stoll(argv[12]) : 8;
        double inventory_cap_multiplier = argc > 13 ? std::stod(argv[13]) : 0.0;
        int64_t struct_l_min = argc > 14 ? std::stoll(argv[14]) : 0;
        int64_t struct_l_max = argc > 15 ? std::stoll(argv[15]) : 0;
        double c_r = argc > 16 ? std::stod(argv[16]) : 0.0;
        std::string action_representation = argc > 17 ? std::string(argv[17]) : std::string("sequential");
        int64_t number_of_trajectories = argc > 18 ? std::stoll(argv[18]) : 1000;
        int64_t periods_per_trajectory = argc > 19 ? std::stoll(argv[19]) : 5000;
        DebugEvalK2FixedVsCDI(system, run_path, mu, sigma, h, b, l_e, l_r, c_e, max_order_size, S_r, S_e, inventory_cap_multiplier, struct_l_min, struct_l_max, c_r, action_representation, number_of_trajectories, periods_per_trajectory);
        return 0;
    }
    if (argc > 1 && std::string(argv[1]) == "trace_k2")
    {
        std::string run_path = argc > 2 ? std::string(argv[2]) : std::string();
        double mu = argc > 3 ? std::stod(argv[3]) : 4.0;
        double sigma = argc > 4 ? std::stod(argv[4]) : 2.0;
        double h = argc > 5 ? std::stod(argv[5]) : 1.0;
        double b = argc > 6 ? std::stod(argv[6]) : 9.0;
        int64_t l_e = argc > 7 ? std::stoll(argv[7]) : 1;
        int64_t l_r = argc > 8 ? std::stoll(argv[8]) : 2;
        double c_e = argc > 9 ? std::stod(argv[9]) : 1.0;
        int64_t max_order_size = argc > 10 ? std::stoll(argv[10]) : 40;
        int64_t S_r = argc > 11 ? std::stoll(argv[11]) : 17;
        int64_t S_e = argc > 12 ? std::stoll(argv[12]) : 8;
        DebugTraceK2Actions(system, run_path, mu, sigma, h, b, l_e, l_r, c_e, max_order_size, S_r, S_e);
        return 0;
    }
    if (argc > 1 && std::string(argv[1]) == "eval_k2_gen")
    {
        std::string run_path = argc > 2 ? std::string(argv[2]) : std::string();
        int64_t generation = argc > 3 ? std::stoll(argv[3]) : 1;
        double mu = argc > 4 ? std::stod(argv[4]) : 4.0;
        double sigma = argc > 5 ? std::stod(argv[5]) : 2.0;
        double h = argc > 6 ? std::stod(argv[6]) : 1.0;
        double b = argc > 7 ? std::stod(argv[7]) : 9.0;
        int64_t l_e = argc > 8 ? std::stoll(argv[8]) : 1;
        int64_t l_r = argc > 9 ? std::stoll(argv[9]) : 2;
        double c_e = argc > 10 ? std::stod(argv[10]) : 1.0;
        int64_t max_order_size = argc > 11 ? std::stoll(argv[11]) : 40;
        int64_t S_r = argc > 12 ? std::stoll(argv[12]) : 17;
        int64_t S_e = argc > 13 ? std::stoll(argv[13]) : 8;
        double inventory_cap_multiplier = argc > 14 ? std::stod(argv[14]) : 0.0;
        int64_t struct_l_min = argc > 15 ? std::stoll(argv[15]) : 0;
        int64_t struct_l_max = argc > 16 ? std::stoll(argv[16]) : 0;
        double c_r = argc > 17 ? std::stod(argv[17]) : 0.0;
        std::string action_representation = argc > 18 ? std::string(argv[18]) : std::string("sequential");
        int64_t number_of_trajectories = argc > 19 ? std::stoll(argv[19]) : 300;
        int64_t periods_per_trajectory = argc > 20 ? std::stoll(argv[20]) : 2000;
        DebugEvalK2GenVsCDI(system, run_path, generation, mu, sigma, h, b, l_e, l_r, c_e, max_order_size, S_r, S_e, inventory_cap_multiplier, struct_l_min, struct_l_max, c_r, action_representation, number_of_trajectories, periods_per_trajectory);
        return 0;
    }
    if (argc > 1 && std::string(argv[1]) == "eval_k1_fixed")
    {
        std::string run_path = argc > 2 ? std::string(argv[2]) : std::string();
        double mu = argc > 3 ? std::stod(argv[3]) : 4.0;
        double sigma = argc > 4 ? std::stod(argv[4]) : 2.0;
        double h = argc > 5 ? std::stod(argv[5]) : 1.0;
        double b = argc > 6 ? std::stod(argv[6]) : 9.0;
        int64_t l = argc > 7 ? std::stoll(argv[7]) : 2;
        int64_t tuned_S = argc > 8 ? std::stoll(argv[8]) : 17;
        int64_t max_order_size = argc > 9 ? std::stoll(argv[9]) : 270;
        double inventory_cap_multiplier = argc > 10 ? std::stod(argv[10]) : 0.0;
        DebugEvalK1FixedVsBaseStock(system, run_path, mu, sigma, h, b, l, tuned_S, max_order_size, inventory_cap_multiplier);
        return 0;
    }
    if (argc > 1 && std::string(argv[1]) == "trace_k1")
    {
        std::string run_path = argc > 2 ? std::string(argv[2]) : std::string();
        double mu = argc > 3 ? std::stod(argv[3]) : 4.0;
        double sigma = argc > 4 ? std::stod(argv[4]) : 2.0;
        double h = argc > 5 ? std::stod(argv[5]) : 1.0;
        double b = argc > 6 ? std::stod(argv[6]) : 9.0;
        int64_t l = argc > 7 ? std::stoll(argv[7]) : 2;
        int64_t tuned_S = argc > 8 ? std::stoll(argv[8]) : 17;
        int64_t max_order_size = argc > 9 ? std::stoll(argv[9]) : 270;
        DebugTraceK1Actions(system, run_path, mu, sigma, h, b, l, tuned_S, max_order_size);
        return 0;
    }
    if (argc > 1 && std::string(argv[1]) == "compare_actions")
    {
        std::string run_path_arg = argc > 2 ? std::string(argv[2]) : std::string();
        DebugCompareActionsAcrossPolicies(system, run_path_arg);
        return 0;
    }
    if (argc > 1 && std::string(argv[1]) == "sweep_actions")
    {
        std::string run_path_arg = argc > 2 ? std::string(argv[2]) : std::string();
        std::string sweep_type_arg = argc > 3 ? std::string(argv[3]) : std::string("b");
        DebugSweepNetworkActions(system, run_path_arg, sweep_type_arg);
        return 0;
    }

    VarGroup instance;
    instance.Add("name", std::string("validation_instance"));
    instance.Add("l_e", int64_t(1));
    instance.Add("l_r", int64_t(2));
    instance.Add("mu", 4.0);
    instance.Add("sigma", 2.0);
    instance.Add("h", 1.0);
    instance.Add("b", 9.0);
    instance.Add("c_r", 1.0);
    instance.Add("c_e", 3.0);

    VarGroup sim_config;
    sim_config.Add("warmup_periods", int64_t(200));
    sim_config.Add("periods_per_trajectory", int64_t(5000));
    sim_config.Add("number_of_trajectories", int64_t(1000));

    VarGroup tuning_config;
    tuning_config.Add("warmup_periods", int64_t(200));
    tuning_config.Add("periods_per_trajectory", int64_t(5000));
    tuning_config.Add("number_of_trajectories", int64_t(1000));

    system << "=== CDI Tuner Validation ===" << std::endl;
    system << "Instance: l_e=1, l_r=2, mu=4, h=1, b=9, c_r=1, c_e=3"
           << std::endl;

    VarGroup mdp_config = BuildInstanceConfig(instance);
    DynaPlex::MDP mdp = dp.GetMDP(mdp_config);

    double mu = 4.0, sigma = 2.0, h = 1.0, b = 9.0;
    int64_t l_r = 2;
    auto dist = DiscreteDist::GetAdanEenigeResingDist(mu, sigma);
    auto demand_bound = DiscreteDist::GetZeroDist();
    for (int64_t i = 0; i <= l_r; i++)
        demand_bound = demand_bound.Add(dist);
    int64_t max_val = demand_bound.Fractile(b / (b + h));

    system << "Search bounds: S_r, S_e in [0," << max_val << "]"
           << std::endl;

    system << "\n--- Step 1: Exhaustive Grid Search ---" << std::endl;
    VarGroup grid_result = GridSearchCDI(mdp, sim_config, max_val, system);

    int64_t grid_S_r, grid_S_e;
    double grid_cost;
    grid_result.Get("S_r", grid_S_r);
    grid_result.Get("S_e", grid_S_e);
    grid_result.Get("cost", grid_cost);

    system << "Grid search result:" << std::endl;
    system << "  S_r = " << grid_S_r << std::endl;
    system << "  S_e = " << grid_S_e << std::endl;
    system << "  cost = " << grid_cost << std::endl;

    system << "\n--- Step 2: Coordinate Descent (TuneCDI) ---" << std::endl;

    double c_r = 1.0, c_e = 3.0;
    int64_t l_e = 1;

    VarGroup tune_result = TuneCDI(mdp, tuning_config,
        mu, sigma, b, h, l_r, l_e, max_val);

    int64_t tune_S_r, tune_S_e;
    double tune_cost;
    tune_result.Get("S_r", tune_S_r);
    tune_result.Get("S_e", tune_S_e);
    tune_result.Get("cost", tune_cost);

    system << "Coordinate descent result:" << std::endl;
    system << "  S_r = " << tune_S_r << std::endl;
    system << "  S_e = " << tune_S_e << std::endl;
    system << "  cost = " << tune_cost << std::endl;

    system << "\n--- Step 3: Comparison ---" << std::endl;

    bool params_match = (grid_S_r == tune_S_r && grid_S_e == tune_S_e);
    double cost_diff = std::abs(tune_cost - grid_cost) / grid_cost * 100.0;

    system << "Parameters match: " << (params_match ? "YES" : "NO")
           << std::endl;
    system << "Cost difference: " << cost_diff << "%" << std::endl;

    if (cost_diff < 1.0)
        system << "VALIDATION PASSED: costs within 1%" << std::endl;
    else
        system << "VALIDATION FAILED: cost difference > 1%" << std::endl;

    system << "\n--- Step 4: Newsvendor Sanity Check ---" << std::endl;

    auto demand_lr = DiscreteDist::GetZeroDist();
    for (int64_t i = 0; i < l_r; i++)
        demand_lr = demand_lr.Add(dist);
    int64_t nv_S_r = demand_lr.Fractile(b / (b + h));

    system << "Newsvendor S_r at b/(b+h)=0.9 over l_r=2 periods: "
           << nv_S_r << std::endl;
    system << "Grid search S_r: " << grid_S_r << std::endl;
    system << "Expected: grid S_r should be close to " << nv_S_r
           << " (may differ due to dual sourcing trade-off)" << std::endl;

    return 0;
}
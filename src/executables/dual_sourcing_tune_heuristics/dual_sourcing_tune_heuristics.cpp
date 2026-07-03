#include <iostream>
#include <limits>
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

// Helper functions

double EvaluatePolicy(DynaPlex::MDP& mdp, DynaPlex::Policy& policy, const VarGroup& tuning_config) {
    auto& dp = DynaPlexProvider::Get();
    auto comparer = dp.GetPolicyComparer(mdp, tuning_config);
    auto result = comparer.Assess(policy);
    double cost;
    result.Get("mean", cost);
    int64_t periods;
    tuning_config.Get("periods_per_trajectory", periods);
    return cost / static_cast<double>(periods);
}

int64_t NewsvendorFractile(double mu, double sigma, double fractile, int64_t lead_time) {
    auto dist = DiscreteDist::GetAdanEenigeResingDist(mu, sigma);
    auto demand_over_lt = DiscreteDist::GetZeroDist();
    for (int64_t i = 0; i < lead_time; i++) {
        demand_over_lt = demand_over_lt.Add(dist);
    }
    return demand_over_lt.Fractile(fractile);
}

int64_t LineSearch(DynaPlex::MDP& mdp, const VarGroup& tuning_config, const std::string& policy_id, VarGroup policy_config, const std::string& param_name, int64_t start_val, int64_t max_val) {
    auto& dp = DynaPlexProvider::Get();
    policy_config.Set(param_name, start_val);
    auto policy = mdp->GetPolicy(policy_config);
    double best_cost = EvaluatePolicy(mdp, policy, tuning_config);
    int64_t best_val = start_val;

    // Search upward
    int64_t no_improve_count = 0;
    for (int64_t val = start_val + 1; val <= max_val; val++)
    {
        policy_config.Set(param_name, val);
        policy = mdp->GetPolicy(policy_config);
        double cost = EvaluatePolicy(mdp, policy, tuning_config);
        if (cost < best_cost)
        {
            best_cost = cost;
            best_val = val;
            no_improve_count = 0;
        }
        else
        {
            no_improve_count++;
            if (no_improve_count >= 30) break;
        }
    }

    // Search downward from start
    no_improve_count = 0;
    for (int64_t val = start_val - 1; val >= 0; val--)
    {
        policy_config.Set(param_name, val);
        policy = mdp->GetPolicy(policy_config);
        double cost = EvaluatePolicy(mdp, policy, tuning_config);
        if (cost < best_cost)
        {
            best_cost = cost;
            best_val = val;
            no_improve_count = 0;
        }
        else
        {
            no_improve_count++;
            if (no_improve_count >= 30) break;
        }
    }

    return best_val;

}

// Heuristic tuning

VarGroup TuneCDI(DynaPlex::MDP& mdp, const VarGroup& tuning_config, double mu, double sigma, double b, double h, int64_t l_r, int64_t l_e, int64_t max_val) {
    double fractile = b / (b + h);

    int64_t S_r = NewsvendorFractile(mu, sigma, fractile, l_r);
    int64_t S_e = NewsvendorFractile(mu, sigma, fractile, l_e);

    VarGroup policy_config;
    policy_config.Add("id", std::string("cdi"));
    policy_config.Add("S_r", S_r);
    policy_config.Add("S_e", S_e);

    auto& dp = DynaPlexProvider::Get();
    double prev_cost = std::numeric_limits<double>::infinity();
    
    int64_t prev_S_r = -1;
    int64_t prev_S_e = -1;

    while(true) {
        S_r = LineSearch(mdp, tuning_config, "cdi", policy_config, "S_r", S_r, max_val);
        policy_config.Set("S_r", S_r);

        S_e = LineSearch(mdp, tuning_config, "cdi", policy_config, "S_e", S_e, S_r);
        policy_config.Set("S_e", S_e);


        if (S_r == prev_S_r && S_e == prev_S_e) break;
        prev_S_r = S_r;
        prev_S_e = S_e;
    }

    auto policy = mdp->GetPolicy(policy_config);
    double cost = EvaluatePolicy(mdp, policy, tuning_config);

    VarGroup result;
    result.Add("S_r", S_r);
    result.Add("S_e", S_e);
    result.Add("cost", cost);

    return result;    
}

VarGroup TuneDI(DynaPlex::MDP& mdp, const VarGroup& tuning_config, double mu, double sigma, double b, double h, int64_t l_r, int64_t max_val) {
    double fractile = b / (b + h);
    int64_t S = NewsvendorFractile(mu, sigma, fractile, l_r);

    VarGroup policy_config;
    policy_config.Add("id", std::string("di"));
    policy_config.Add("S", S);

    S = LineSearch(mdp, tuning_config, "di", policy_config, "S", S, max_val);
    policy_config.Set("S", S);

    auto& dp = DynaPlexProvider::Get();
    auto policy = mdp->GetPolicy(policy_config);
    double cost = EvaluatePolicy(mdp, policy, tuning_config);

    VarGroup result;
    result.Add("S", S);
    result.Add("cost", cost);
    return result;
}

VarGroup TuneSI(DynaPlex::MDP& mdp, const VarGroup& tuning_config, double mu, double sigma, double b, double h, int64_t l_r, int64_t max_val) {
    double fractile = b / (b + h);
    int64_t S = NewsvendorFractile(mu, sigma, fractile, l_r);

    VarGroup policy_config;
    policy_config.Add("id", std::string("si"));
    policy_config.Add("S", S);

    S = LineSearch(mdp, tuning_config, "si", policy_config, "S", S, max_val);
    policy_config.Set("S", S);

    auto& dp = DynaPlexProvider::Get();
    auto policy = mdp->GetPolicy(policy_config);
    double cost = EvaluatePolicy(mdp, policy, tuning_config);

    VarGroup result;
    result.Add("S", S);
    result.Add("cost", cost);
    return result;
}

VarGroup TuneTBS(DynaPlex::MDP& mdp, const VarGroup& tuning_config, double mu, double sigma, double b, double h, int64_t l_e, int64_t max_val) {
    double fractile = b / (b + h);
    int64_t S_e = NewsvendorFractile(mu, sigma, fractile, l_e);
    int64_t c = static_cast<int64_t>(std::round(mu));

    VarGroup policy_config;
    policy_config.Add("id", std::string("tbs"));
    policy_config.Add("S_e", S_e);
    policy_config.Add("c", c);

    auto& dp = DynaPlexProvider::Get();
    double prev_cost = std::numeric_limits<double>::infinity();

    int64_t prev_c = -1;
    int64_t prev_S_e = -1;

    while(true) {
        S_e = LineSearch(mdp, tuning_config, "tbs", policy_config, "S_e", S_e, max_val);
        policy_config.Set("S_e", S_e);

        c = LineSearch(mdp, tuning_config, "tbs", policy_config, "c", c, max_val);
        policy_config.Set("c", c);

        if (S_e == prev_S_e && prev_c == c) break;
        prev_S_e = S_e;
        prev_c = c;
    }

    auto policy = mdp->GetPolicy(policy_config);
    double cost = EvaluatePolicy(mdp, policy, tuning_config);

    VarGroup result;
    result.Add("S_e", S_e);
    result.Add("c", c);
    result.Add("cost", cost);
    return result;
}

int main(int argc, char* argv[])
{
    auto& dp = DynaPlexProvider::Get();
    auto& system = dp.System();

    std::string instances_config_name = "instances_config.json";
    if (argc > 1) instances_config_name = argv[1];

    VarGroup instances_config = VarGroup::LoadFromFile(system.filepath("mdp_config_examples", "dual_sourcing_backlog", instances_config_name));

    std::vector<VarGroup> instances;
    instances_config.Get("instances", instances);

    VarGroup tuning_config;
    instances_config.Get("tuning", tuning_config);

    system << "Loaded " << instances.size() << " instances. Starting tuning..." << std::endl;

    std::vector<VarGroup> results;

    for (auto& instance : instances) {
        std::string name;
        instance.Get("name", name);
        system << "Tuning instance: " << name << std::endl;

        VarGroup mdp_config = BuildInstanceConfig(instance);
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
        for (int64_t i = 0; i <= l_r; i++) demand_over_lr = demand_over_lr.Add(dist);
        int64_t max_val = demand_over_lr.Fractile(b / (b + h));

        system << "  Tuning CDI..." << std::endl;
        VarGroup cdi = TuneCDI(mdp, tuning_config, mu, sigma, b, h, l_r, l_e, max_val);

        system << "  Tuning DI..." << std::endl;
        VarGroup di = TuneDI(mdp, tuning_config, mu, sigma, b, h, l_r, max_val);

        system << "  Tuning SI..." << std::endl;
        VarGroup si = TuneSI(mdp, tuning_config, mu, sigma, b, h, l_r, max_val);

        system << "  Tuning TBS..." << std::endl;
        VarGroup tbs = TuneTBS(mdp, tuning_config, mu, sigma, b, h, l_e, max_val);

        VarGroup instance_result = instance;
        instance_result.Add("CDI", cdi);
        instance_result.Add("DI", di);
        instance_result.Add("SI", si);
        instance_result.Add("TBS", tbs);
        results.push_back(instance_result);

        system << "Done.";
    }

    // save results
    VarGroup output;
    output.Add("tuned_policies", results);

    auto output_path = system.filepath("dual_sourcing_backlog", "tuned_heuristic_params.json");
    output.SaveToFile(output_path); // IntelliSense kan hem niet vinden, maar compileert wel

    system << "Tuning complete. Results saved to: " << output_path << std::endl;

    return 0;
}
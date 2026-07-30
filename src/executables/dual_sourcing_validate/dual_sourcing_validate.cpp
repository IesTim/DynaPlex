#include <iostream>
#include <limits>
#include <string>
#include "dynaplex/dynaplexprovider.h"
#include "dynaplex/modelling/discretedist.h"
#include "../../lib/models/models/dual_sourcing_backlog/tuning_utils.h"

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

int main(int argc, char* argv[])
{
    auto& dp = DynaPlexProvider::Get();
    auto& system = dp.System();

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
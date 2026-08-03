#pragma once
#include "dynaplex/dynaplexprovider.h"
#include "dynaplex/modelling/discretedist.h"
#include <cmath>
#include <limits>

using namespace DynaPlex;

inline double EvaluatePolicyTuning(DynaPlex::MDP& mdp,  DynaPlex::Policy& policy, const VarGroup& tuning_config)
{
    auto& dp = DynaPlexProvider::Get();
    auto comparer = dp.GetPolicyComparer(mdp, tuning_config);
    auto result = comparer.Assess(policy);
    double cost;
    result.Get("mean", cost);
    return cost;
}

inline int64_t NewsvendorFractile(double mu, double sigma, double fractile, int64_t lead_time)
{
    auto dist = DiscreteDist::GetAdanEenigeResingDist(mu, sigma);
    auto demand_over_lt = DiscreteDist::GetZeroDist();
    for (int64_t i = 0; i < lead_time; i++)
        demand_over_lt = demand_over_lt.Add(dist);
    return demand_over_lt.Fractile(fractile);
}

inline int64_t LineSearch(DynaPlex::MDP& mdp, const VarGroup& tuning_config, VarGroup policy_config, const std::string& param_name, int64_t start_val, int64_t max_val, int64_t patience = 5)
{
    policy_config.Set(param_name, start_val);
    auto policy = mdp->GetPolicy(policy_config);
    double best_cost = EvaluatePolicyTuning(mdp, policy, tuning_config);
    int64_t best_val = start_val;

    // Search up
    int64_t no_improve = 0;
    for (int64_t val = start_val + 1; val <= max_val; val++)
    {
        policy_config.Set(param_name, val);
        policy = mdp->GetPolicy(policy_config);
        double cost = EvaluatePolicyTuning(mdp, policy, tuning_config);
        if (cost < best_cost)
        {
            best_cost = cost;
            best_val = val;
            no_improve = 0;
        }
        else
        {
            no_improve++;
            if (no_improve >= patience) break;
        }
    }

    // Search down
    no_improve = 0;
    for (int64_t val = start_val - 1; val >= 0; val--)
    {
        policy_config.Set(param_name, val);
        policy = mdp->GetPolicy(policy_config);
        double cost = EvaluatePolicyTuning(mdp, policy, tuning_config);
        if (cost < best_cost)
        {
            best_cost = cost;
            best_val = val;
            no_improve = 0;
        }
        else
        {
            no_improve++;
            if (no_improve >= patience) break;
        }
    }

    return best_val;
}

inline VarGroup TuneCDI(DynaPlex::MDP& mdp, const VarGroup& tuning_config, double mu, double sigma, double b, double h, int64_t l_r, int64_t l_e, int64_t max_val)
{
    double fractile = b / (b + h);
    int64_t S_r = NewsvendorFractile(mu, sigma, fractile, l_r);
    int64_t S_e = std::max(int64_t(1), NewsvendorFractile(mu, sigma, fractile, l_e) / (l_r - l_e + 1));

    VarGroup policy_config;
    policy_config.Add("id", std::string("cdi"));
    policy_config.Add("S_r", S_r);
    policy_config.Add("S_e", S_e);

    int64_t prev_S_r = -1, prev_S_e = -1;
    while (true)
    {
        S_r = LineSearch(mdp, tuning_config, policy_config, "S_r", S_r, max_val);
        policy_config.Set("S_r", S_r);

        S_e = LineSearch(mdp, tuning_config, policy_config, "S_e", S_e, S_r);
        if (S_e < 1) S_e = 1;
        policy_config.Set("S_e", S_e);

        if (S_r == prev_S_r && S_e == prev_S_e) break;
        prev_S_r = S_r;
        prev_S_e = S_e;
    }

    auto policy = mdp->GetPolicy(policy_config);
    double cost = EvaluatePolicyTuning(mdp, policy, tuning_config);

    VarGroup result;
    result.Add("S_r", S_r);
    result.Add("S_e", S_e);
    result.Add("cost", cost);
    return result;
}

inline VarGroup TuneDI(DynaPlex::MDP& mdp, const VarGroup& tuning_config, double mu, double sigma, double b, double h, int64_t l_r, int64_t max_val)
{
    double fractile = b / (b + h);
    int64_t S = NewsvendorFractile(mu, sigma, fractile, l_r);

    VarGroup policy_config;
    policy_config.Add("id", std::string("di"));
    policy_config.Add("S", S);

    S = LineSearch(mdp, tuning_config, policy_config, "S", S, max_val);
    policy_config.Set("S", S);

    auto policy = mdp->GetPolicy(policy_config);
    double cost = EvaluatePolicyTuning(mdp, policy, tuning_config);

    VarGroup result;
    result.Add("S", S);
    result.Add("cost", cost);
    return result;
}

inline VarGroup TuneSI(DynaPlex::MDP& mdp, const VarGroup& tuning_config, double mu, double sigma, double b, double h, int64_t l_r, int64_t max_val)
{
    double fractile = b / (b + h);
    int64_t S = NewsvendorFractile(mu, sigma, fractile, l_r);

    VarGroup policy_config;
    policy_config.Add("id", std::string("si"));
    policy_config.Add("S", S);

    S = LineSearch(mdp, tuning_config, policy_config, "S", S, max_val);
    policy_config.Set("S", S);

    auto policy = mdp->GetPolicy(policy_config);
    double cost = EvaluatePolicyTuning(mdp, policy, tuning_config);

    VarGroup result;
    result.Add("S", S);
    result.Add("cost", cost);
    return result;
}

inline VarGroup TuneTBS(DynaPlex::MDP& mdp,
    const VarGroup& tuning_config,
    double mu, double sigma, double b, double h,
    int64_t l_e, int64_t max_val)
{
    double fractile = b / (b + h);
    int64_t S_e = std::max(int64_t(1), NewsvendorFractile(mu, sigma, fractile, l_e));
    int64_t c = static_cast<int64_t>(std::round(mu));

    VarGroup policy_config;
    policy_config.Add("id", std::string("tbs"));
    policy_config.Add("S_e", S_e);
    policy_config.Add("c", c);

    int64_t prev_S_e = -1, prev_c = -1;
    while (true)
    {
        S_e = LineSearch(mdp, tuning_config, policy_config,
            "S_e", S_e, max_val);
        if (S_e < 1) S_e = 1;
        policy_config.Set("S_e", S_e);

        c = LineSearch(mdp, tuning_config, policy_config,
            "c", c, max_val);
        policy_config.Set("c", c);

        if (S_e == prev_S_e && c == prev_c) break;
        prev_S_e = S_e;
        prev_c = c;
    }

    auto policy = mdp->GetPolicy(policy_config);
    double cost = EvaluatePolicyTuning(mdp, policy, tuning_config);

    VarGroup result;
    result.Add("S_e", S_e);
    result.Add("c", c);
    result.Add("cost", cost);
    return result;
}
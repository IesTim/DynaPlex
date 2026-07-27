#include <iostream>
#include <limits>
#include "dynaplex/dynaplexprovider.h"
#include "dynaplex/modelling/discretedist.h"
#include <cmath>
#include "../../lib/models/models/dual_sourcing_backlog/tuning_utils.h"

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

    auto output_path = system.filepath("dual_sourcing", "tuning", "tuned_heuristic_params.json");
    output.SaveToFile(output_path);

    system << "Tuning complete. Results saved to: " << output_path << std::endl;

    return 0;
}
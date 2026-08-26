#include <iostream>
#include <string>
#include <random>
#include <cmath>
#include <algorithm>
#include "dynaplex/dynaplexprovider.h"
#include "dynaplex/modelling/discretedist.h"
#include "../../lib/models/models/dual_sourcing_backlog/tuning_utils.h"

using namespace DynaPlex;

namespace {

    // Builds an mdp_config for a single fixed instance. action_representation, oracle_mu_init,
    // and rollout_M are all overridable per policy-spec entry so the same instance can be
    // evaluated fairly under "sequential", "flat_joint", with/without oracle mu, etc.
    VarGroup BuildMdpConfig(const VarGroup& instance, int64_t train_l_max, int64_t max_order_size,
        double inventory_cap_multiplier, const std::string& action_representation, bool oracle_mu_init)
    {
        VarGroup config;
        config.Add("id", std::string("dual_sourcing_backlog"));
        config.Add("K", int64_t(2));

        int64_t l_e, l_r;
        instance.Get("l_e", l_e);
        instance.Get("l_r", l_r);
        config.Add("l_min", l_e);
        config.Add("l_max", train_l_max > 0 ? train_l_max : l_r);

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
        config.Add("max_order_size", max_order_size);
        config.Add("inventory_cap_multiplier", inventory_cap_multiplier);
        config.Add("oracle_mu_init", oracle_mu_init);

        VarGroup fixed_instance;
        fixed_instance.Add("h", h);
        fixed_instance.Add("b", b);
        fixed_instance.Add("mu", mu);
        fixed_instance.Add("sigma", sigma);
        fixed_instance.Add("l_e", l_e);
        fixed_instance.Add("l_r", l_r);
        std::vector<double> costs = { c_e, c_r };
        fixed_instance.Add("costs", costs);
        config.Add("fixed_instance", fixed_instance);

        return config;
    }

    // Same fractile/max_val bound used throughout this project's tuning scripts to cap CDI's
    // regular-source search range: the (1-alpha)-fractile of demand over l_r periods.
    int64_t CdiSearchBound(double mu, double sigma, double b, double h, int64_t l_r)
    {
        double fractile = b / (b + h);
        return NewsvendorFractile(mu, sigma, fractile, l_r) * 2 + 10;
    }

    double RandUniform(std::mt19937_64& rng, double lo, double hi)
    {
        std::uniform_real_distribution<double> dist(lo, hi);
        return dist(rng);
    }

    // Mirrors scripts/large_scale_cdi_comparison.py's sample_instance(), now native so the whole
    // pipeline (sampling, tuning, evaluation, logging) runs inside one reproducible executable.
    VarGroup SampleRandomInstance(std::mt19937_64& rng, double mu_lo, double mu_hi, double b_lo, double b_hi,
        double c_lo, double c_hi, double h_lo, double h_hi, int64_t l_min, int64_t l_max)
    {
        double mu = RandUniform(rng, mu_lo, mu_hi);
        double sigma = mu / 2.0;
        double b = RandUniform(rng, b_lo, b_hi);
        double c_e = RandUniform(rng, c_lo, c_hi);
        double c_r = RandUniform(rng, c_lo, c_e);
        double h = RandUniform(rng, h_lo, h_hi);

        std::vector<std::pair<int64_t, int64_t>> valid_pairs;
        for (int64_t a = l_min; a < l_max; a++)
            for (int64_t bb = a + 1; bb <= l_max; bb++)
                valid_pairs.push_back({ a, bb });
        std::uniform_int_distribution<size_t> idx_dist(0, valid_pairs.size() - 1);
        auto [l_e, l_r] = valid_pairs[idx_dist(rng)];

        VarGroup instance;
        instance.Add("mu", mu);
        instance.Add("sigma", sigma);
        instance.Add("h", h);
        instance.Add("b", b);
        instance.Add("c_e", c_e);
        instance.Add("c_r", c_r);
        instance.Add("l_e", l_e);
        instance.Add("l_r", l_r);
        return instance;
    }

    // Reads the "instances" block of an experiment spec: either an explicit list, or a random
    // draw from ranges (mode: "random") using a documented seed for reproducibility.
    std::vector<VarGroup> LoadInstances(const VarGroup& spec)
    {
        VarGroup instances_spec;
        spec.Get("instances", instances_spec);
        std::string mode;
        instances_spec.Get("mode", mode);

        std::vector<VarGroup> instances;
        if (mode == "explicit")
        {
            instances_spec.Get("list", instances);
            return instances;
        }
        if (mode != "random")
            throw DynaPlex::Error("dual_sourcing_eval: instances.mode must be 'explicit' or 'random', got: " + mode);

        int64_t n_instances, seed, l_min, l_max;
        instances_spec.Get("n_instances", n_instances);
        instances_spec.Get("seed", seed);
        instances_spec.Get("l_min", l_min);
        instances_spec.Get("l_max", l_max);
        std::vector<double> mu_range, b_range, c_range, h_range;
        instances_spec.Get("mu_range", mu_range);
        instances_spec.Get("b_range", b_range);
        instances_spec.Get("c_range", c_range);
        instances_spec.Get("h_range", h_range);

        std::mt19937_64 rng(static_cast<uint64_t>(seed));
        instances.reserve(n_instances);
        for (int64_t i = 0; i < n_instances; i++)
            instances.push_back(SampleRandomInstance(rng, mu_range[0], mu_range[1], b_range[0], b_range[1],
                c_range[0], c_range[1], h_range[0], h_range[1], l_min, l_max));
        return instances;
    }

    // Builds+tunes (for heuristics) or loads (for a trained GCA policy) the policy described by
    // one entry of the spec's "policies" list, for one specific instance. Returns the policy plus
    // whatever tuning metadata is relevant (S_r/S_e/etc, empty for GCA).
    struct ResolvedPolicy {
        DynaPlex::Policy policy;
        DynaPlex::MDP mdp;
        VarGroup tuning_info;
        bool has_tuning_info = false;
    };

    ResolvedPolicy ResolvePolicy(const VarGroup& policy_spec, const VarGroup& instance,
        int64_t train_l_max, int64_t max_order_size, double inventory_cap_multiplier,
        const VarGroup& tuning_sim_config)
    {
        auto& dp = DynaPlexProvider::Get();

        std::string type;
        policy_spec.Get("type", type);
        std::string action_representation = "flat_joint";
        if (policy_spec.HasKey("action_representation"))
            policy_spec.Get("action_representation", action_representation);
        bool oracle_mu_init = false;
        if (policy_spec.HasKey("oracle_mu_init"))
            policy_spec.Get("oracle_mu_init", oracle_mu_init);

        VarGroup mdp_config = BuildMdpConfig(instance, train_l_max, max_order_size,
            inventory_cap_multiplier, action_representation, oracle_mu_init);
        DynaPlex::MDP mdp = dp.GetMDP(mdp_config);

        double mu, sigma, h, b;
        int64_t l_e, l_r;
        instance.Get("mu", mu); instance.Get("sigma", sigma);
        instance.Get("h", h); instance.Get("b", b);
        instance.Get("l_e", l_e); instance.Get("l_r", l_r);
        int64_t max_val = CdiSearchBound(mu, sigma, b, h, l_r);

        ResolvedPolicy result;
        result.mdp = mdp;

        if (type == "gca")
        {
            std::string run_path;
            policy_spec.Get("run_path", run_path);
            auto& system = dp.System();
            if (policy_spec.HasKey("generation"))
            {
                // Load a specific mid-training checkpoint instead of policy_final - needed for
                // runs stopped early (e.g. the flat_joint infeasibility demonstration, killed
                // after gen 3 rather than running all 5 generations to make an already-clear
                // point). Mirrors dual_sourcing_validate.cpp's eval_k2_gen mechanism.
                int64_t generation;
                policy_spec.Get("generation", generation);
                VarGroup run_info = VarGroup::LoadFromFile(system.filepath("dual_sourcing", "runs", run_path, "run_info.json"));
                std::string mdp_identifier;
                run_info.Get("mdp_identifier", mdp_identifier);
                auto gen_path = system.filepath(mdp_identifier, "dcl_policy_gen" + std::to_string(generation));
                result.policy = dp.LoadPolicy(mdp, gen_path);
            }
            else
            {
                auto full_path = system.filepath("dual_sourcing", "runs", run_path, "policy_final");
                result.policy = dp.LoadPolicy(mdp, full_path);
            }
        }
        else if (type == "cdi")
        {
            VarGroup tuned = TuneCDI(mdp, tuning_sim_config, mu, sigma, b, h, l_r, l_e, max_val);
            result.tuning_info = tuned; result.has_tuning_info = true;
            int64_t S_r, S_e;
            tuned.Get("S_r", S_r); tuned.Get("S_e", S_e);
            VarGroup pc; pc.Add("id", std::string("cdi")); pc.Add("S_r", S_r); pc.Add("S_e", S_e);
            result.policy = mdp->GetPolicy(pc);
        }
        else if (type == "di")
        {
            VarGroup tuned = TuneDI(mdp, tuning_sim_config, mu, sigma, b, h, l_r, max_val);
            result.tuning_info = tuned; result.has_tuning_info = true;
            int64_t S; tuned.Get("S", S);
            VarGroup pc; pc.Add("id", std::string("di")); pc.Add("S", S);
            result.policy = mdp->GetPolicy(pc);
        }
        else if (type == "si")
        {
            VarGroup tuned = TuneSI(mdp, tuning_sim_config, mu, sigma, b, h, l_r, max_val);
            result.tuning_info = tuned; result.has_tuning_info = true;
            int64_t S; tuned.Get("S", S);
            VarGroup pc; pc.Add("id", std::string("si")); pc.Add("S", S);
            result.policy = mdp->GetPolicy(pc);
        }
        else if (type == "tbs")
        {
            VarGroup tuned = TuneTBS(mdp, tuning_sim_config, mu, sigma, b, h, l_e, max_val);
            result.tuning_info = tuned; result.has_tuning_info = true;
            int64_t S_e, c; tuned.Get("S_e", S_e); tuned.Get("c", c);
            VarGroup pc; pc.Add("id", std::string("tbs")); pc.Add("S_e", S_e); pc.Add("c", c);
            result.policy = mdp->GetPolicy(pc);
        }
        else
            throw DynaPlex::Error("dual_sourcing_eval: unknown policy type: " + type);

        return result;
    }

    // Core workhorse: evaluate every policy in spec.policies against every instance in
    // spec.instances, with raw per-trajectory costs logged for every (policy, instance) pair.
    // Covers the in-range benchmark table, the OOD sweep, zero-shot comparison, and the
    // mu-oracle-vs-estimated test - they differ only in which instances/policies the spec lists.
    void RunCompare(const std::string& spec_name)
    {
        auto& dp = DynaPlexProvider::Get();
        auto& system = dp.System();

        VarGroup spec = VarGroup::LoadFromFile(system.filepath("mdp_config_examples", "dual_sourcing_backlog", "configs", spec_name));

        int64_t train_l_max = 0, max_order_size;
        double inventory_cap_multiplier = 3.0;
        if (spec.HasKey("train_l_max")) spec.Get("train_l_max", train_l_max);
        spec.Get("max_order_size", max_order_size);
        if (spec.HasKey("inventory_cap_multiplier")) spec.Get("inventory_cap_multiplier", inventory_cap_multiplier);

        VarGroup sim_config, tuning_sim_config;
        spec.Get("simulation", sim_config);
        sim_config.Set("return_raw_trajectories", true);
        if (spec.HasKey("tuning_simulation"))
            spec.Get("tuning_simulation", tuning_sim_config);
        else
            tuning_sim_config = sim_config;

        std::vector<VarGroup> policy_specs;
        spec.Get("policies", policy_specs);

        std::vector<VarGroup> instances = LoadInstances(spec);

        system << "dual_sourcing_eval compare: " << instances.size() << " instances x "
               << policy_specs.size() << " policies" << std::endl;

        std::vector<VarGroup> per_instance_results;
        per_instance_results.reserve(instances.size());

        for (size_t i = 0; i < instances.size(); i++)
        {
            auto& instance = instances[i];
            system << "instance " << (i + 1) << "/" << instances.size() << std::endl;

            VarGroup instance_result;
            instance_result.Add("instance_idx", static_cast<int64_t>(i));
            instance_result.Add("instance", instance);

            std::vector<VarGroup> policy_results;
            for (auto& pspec : policy_specs)
            {
                std::string label;
                pspec.Get("label", label);
                try
                {
                    ResolvedPolicy resolved = ResolvePolicy(pspec, instance, train_l_max, max_order_size,
                        inventory_cap_multiplier, tuning_sim_config);
                    auto comparer = dp.GetPolicyComparer(resolved.mdp, sim_config);
                    auto assessment = comparer.Assess(resolved.policy);

                    VarGroup pres;
                    pres.Add("label", label);
                    pres.Add("mean", assessment);
                    if (resolved.has_tuning_info)
                        pres.Add("tuning_info", resolved.tuning_info);
                    policy_results.push_back(pres);
                    double mean_cost;
                    assessment.Get("mean", mean_cost);
                    system << "  " << label << ": " << mean_cost << std::endl;
                }
                catch (const DynaPlex::Error& e)
                {
                    system << "  " << label << " FAILED: " << e.what() << std::endl;
                    VarGroup pres;
                    pres.Add("label", label);
                    pres.Add("error", std::string(e.what()));
                    policy_results.push_back(pres);
                }
            }
            instance_result.Add("policies", policy_results);
            per_instance_results.push_back(instance_result);
        }

        VarGroup output;
        output.Add("spec", spec);
        output.Add("results", per_instance_results);

        std::string output_name;
        spec.Get("output", output_name);
        auto out_path = system.filepath("dual_sourcing", "evaluation", output_name);
        output.SaveToFile(out_path, 2);
        system << "Done. Results saved to " << out_path << std::endl;
    }

    // General-K mdp_config builder for the K-scaling stress test. Parallel to BuildMdpConfig
    // rather than a generalization of it, to avoid any risk of regressing the K=2 "compare" mode
    // used by every other result in this thesis. Instance schema: {K, mu, sigma, h, b,
    // l: [l_0..l_{K-1}], c: [c_0..c_{K-1}]}.
    VarGroup BuildMdpConfigK(const VarGroup& instance, int64_t K, int64_t train_l_max, int64_t max_order_size,
        double inventory_cap_multiplier, const std::string& action_representation, double rollout_M)
    {
        VarGroup config;
        config.Add("id", std::string("dual_sourcing_backlog"));
        config.Add("K", K);

        std::vector<int64_t> l;
        std::vector<double> c;
        instance.Get("l", l);
        instance.Get("c", c);
        config.Add("l_min", l.front());
        // l_max must match the structural bound the network was trained with (train_l_max), not
        // this instance's own l_r, otherwise the state's pipeline-vector length - and hence
        // NumFlatFeatures() - won't match what the trained network expects.
        config.Add("l_max", train_l_max);

        double mu, sigma, h, b;
        instance.Get("mu", mu); instance.Get("sigma", sigma);
        instance.Get("h", h); instance.Get("b", b);

        config.Add("min_h", h);   config.Add("max_h", h);
        config.Add("min_b", b);   config.Add("max_b", b);
        config.Add("min_c", *std::min_element(c.begin(), c.end()));
        config.Add("max_c", *std::max_element(c.begin(), c.end()));
        config.Add("min_mu", mu); config.Add("max_mu", mu);
        config.Add("action_representation", action_representation);
        config.Add("discount_factor", 1.0);
        config.Add("max_order_size", max_order_size);
        config.Add("inventory_cap_multiplier", inventory_cap_multiplier);
        config.Add("rollout_M", rollout_M);

        VarGroup fixed_instance;
        fixed_instance.Add("h", h);
        fixed_instance.Add("b", b);
        fixed_instance.Add("mu", mu);
        fixed_instance.Add("sigma", sigma);
        fixed_instance.Add("l", l);
        fixed_instance.Add("costs", c);
        config.Add("fixed_instance", fixed_instance);

        return config;
    }

    void RunCompareK(const std::string& spec_name)
    {
        auto& dp = DynaPlexProvider::Get();
        auto& system = dp.System();

        VarGroup spec = VarGroup::LoadFromFile(system.filepath("mdp_config_examples", "dual_sourcing_backlog", "configs", spec_name));

        int64_t K, max_order_size, train_l_max;
        double inventory_cap_multiplier = 3.0, rollout_M = 100;
        spec.Get("K", K);
        spec.Get("max_order_size", max_order_size);
        spec.Get("train_l_max", train_l_max);
        if (spec.HasKey("inventory_cap_multiplier")) spec.Get("inventory_cap_multiplier", inventory_cap_multiplier);
        if (spec.HasKey("rollout_M")) spec.Get("rollout_M", rollout_M);

        VarGroup sim_config;
        spec.Get("simulation", sim_config);
        sim_config.Set("return_raw_trajectories", true);

        std::vector<VarGroup> policy_specs;
        spec.Get("policies", policy_specs);
        std::vector<VarGroup> instances;
        spec.Get("instances", instances);

        system << "dual_sourcing_eval compare_k: K=" << K << ", " << instances.size()
               << " instances x " << policy_specs.size() << " policies" << std::endl;

        std::vector<VarGroup> per_instance_results;
        for (size_t i = 0; i < instances.size(); i++)
        {
            auto& instance = instances[i];
            system << "instance " << (i + 1) << "/" << instances.size() << std::endl;
            VarGroup instance_result;
            instance_result.Add("instance_idx", static_cast<int64_t>(i));
            instance_result.Add("instance", instance);

            std::vector<VarGroup> policy_results;
            for (auto& pspec : policy_specs)
            {
                std::string label, type, action_representation = "sequential";
                pspec.Get("label", label);
                pspec.Get("type", type);
                if (pspec.HasKey("action_representation")) pspec.Get("action_representation", action_representation);

                VarGroup mdp_config = BuildMdpConfigK(instance, K, train_l_max, max_order_size, inventory_cap_multiplier, action_representation, rollout_M);
                DynaPlex::MDP mdp = dp.GetMDP(mdp_config);

                try
                {
                    DynaPlex::Policy policy;
                    if (type == "gca")
                    {
                        std::string run_path;
                        pspec.Get("run_path", run_path);
                        auto full_path = system.filepath("dual_sourcing", "runs", run_path, "policy_final");
                        policy = dp.LoadPolicy(mdp, full_path);
                    }
                    else if (type == "adaptive_cdi_k")
                    {
                        policy = mdp->GetPolicy("adaptive_cdi_k");
                    }
                    else
                        throw DynaPlex::Error("compare_k: unknown policy type " + type);

                    auto comparer = dp.GetPolicyComparer(mdp, sim_config);
                    auto assessment = comparer.Assess(policy);
                    VarGroup pres;
                    pres.Add("label", label);
                    pres.Add("mean", assessment);
                    policy_results.push_back(pres);
                    double mean_cost;
                    assessment.Get("mean", mean_cost);
                    system << "  " << label << ": " << mean_cost << std::endl;
                }
                catch (const DynaPlex::Error& e)
                {
                    system << "  " << label << " FAILED: " << e.what() << std::endl;
                    VarGroup pres;
                    pres.Add("label", label);
                    pres.Add("error", std::string(e.what()));
                    policy_results.push_back(pres);
                }
            }
            instance_result.Add("policies", policy_results);
            per_instance_results.push_back(instance_result);
        }

        VarGroup output;
        output.Add("spec", spec);
        output.Add("results", per_instance_results);
        std::string output_name;
        spec.Get("output", output_name);
        auto out_path = system.filepath("dual_sourcing", "evaluation", output_name);
        output.SaveToFile(out_path, 2);
        system << "Done. Results saved to " << out_path << std::endl;
    }

    // Structural analysis (K=2 only): drives a trajectory forward under CDI's own actions (so
    // the system stays in realistic, representative states rather than an untested policy
    // possibly running away), and at every period's two sub-decisions, records what CDI actually
    // chose alongside what GCA-DS *would* have chosen from that exact same state - the standard
    // "trace both policies against the same realistic trajectory" comparison, following the
    // pattern already validated in dual_sourcing_validate.cpp's DebugTraceK2Actions, just looped
    // over every period rather than a handful of snapshots. Uses the type-erased Trajectory/
    // SetAction/IncorporateAction interface throughout (DynaPlex::MDP/Policy are fully
    // type-erased - there is no direct GetAction(State) available on them from application code).
    void RunStructural(const std::string& spec_name)
    {
        auto& dp = DynaPlexProvider::Get();
        auto& system = dp.System();

        VarGroup spec = VarGroup::LoadFromFile(system.filepath("mdp_config_examples", "dual_sourcing_backlog", "configs", spec_name));

        int64_t train_l_max, max_order_size, num_periods, seed;
        double inventory_cap_multiplier = 3.0;
        spec.Get("train_l_max", train_l_max);
        spec.Get("max_order_size", max_order_size);
        spec.Get("num_periods", num_periods);
        spec.Get("seed", seed);
        if (spec.HasKey("inventory_cap_multiplier")) spec.Get("inventory_cap_multiplier", inventory_cap_multiplier);

        VarGroup instance, tuning_sim_config;
        spec.Get("instance", instance);
        spec.Get("tuning_simulation", tuning_sim_config);
        std::string run_path;
        spec.Get("run_path", run_path);

        VarGroup mdp_config = BuildMdpConfig(instance, train_l_max, max_order_size, inventory_cap_multiplier, "sequential", false);
        DynaPlex::MDP mdp = dp.GetMDP(mdp_config);

        auto full_path = system.filepath("dual_sourcing", "runs", run_path, "policy_final");
        DynaPlex::Policy gca_policy = dp.LoadPolicy(mdp, full_path);

        double mu, sigma, h, b;
        int64_t l_e, l_r;
        instance.Get("mu", mu); instance.Get("sigma", sigma);
        instance.Get("h", h); instance.Get("b", b);
        instance.Get("l_e", l_e); instance.Get("l_r", l_r);
        int64_t max_val = CdiSearchBound(mu, sigma, b, h, l_r);
        VarGroup cdi_tuned = TuneCDI(mdp, tuning_sim_config, mu, sigma, b, h, l_r, l_e, max_val);
        int64_t S_r, S_e;
        cdi_tuned.Get("S_r", S_r); cdi_tuned.Get("S_e", S_e);
        VarGroup cdi_pc; cdi_pc.Add("id", std::string("cdi")); cdi_pc.Add("S_r", S_r); cdi_pc.Add("S_e", S_e);
        DynaPlex::Policy cdi_policy = mdp->GetPolicy(cdi_pc);
        system << "Tuned CDI: S_r=" << S_r << " S_e=" << S_e << std::endl;

        DynaPlex::Trajectory traj(0);
        traj.RNGProvider.SeedEventStreams(false, seed, 0);
        mdp->InitiateState({ &traj, 1 });

        std::vector<VarGroup> rows;
        for (int64_t period = 0; period < num_periods; period++)
        {
            mdp->IncorporateUntilAction({ &traj, 1 });

            // Source 0 (expedited) decision.
            cdi_policy->SetAction({ &traj, 1 });
            int64_t cdi_q_e = traj.NextAction;
            gca_policy->SetAction({ &traj, 1 });
            int64_t gca_q_e = traj.NextAction;
            mdp->IncorporateAction({ &traj, 1 }, cdi_policy);

            // Source 1 (regular) decision - state now reflects cdi_q_e already placed.
            cdi_policy->SetAction({ &traj, 1 });
            int64_t cdi_q_r = traj.NextAction;
            gca_policy->SetAction({ &traj, 1 });
            int64_t gca_q_r = traj.NextAction;
            mdp->IncorporateAction({ &traj, 1 }, cdi_policy);

            VarGroup row;
            row.Add("period", period);
            row.Add("gca_q_expedited", gca_q_e);
            row.Add("gca_q_regular", gca_q_r);
            row.Add("cdi_q_expedited", cdi_q_e);
            row.Add("cdi_q_regular", cdi_q_r);
            rows.push_back(row);
        }
        system << "Traced " << num_periods << " periods." << std::endl;

        VarGroup output;
        output.Add("spec", spec);
        output.Add("cdi_tuned", cdi_tuned);
        output.Add("rows", rows);
        std::string output_name;
        spec.Get("output", output_name);
        auto out_path = system.filepath("dual_sourcing", "evaluation", output_name);
        output.SaveToFile(out_path, 2);
        system << "Done. Results saved to " << out_path << std::endl;
    }
}

int main(int argc, char* argv[])
{
    auto& dp = DynaPlexProvider::Get();
    auto& system = dp.System();

    if (argc < 3)
    {
        system << "Usage: dual_sourcing_eval <mode> <spec.json>" << std::endl;
        system << "Modes: compare, compare_k, structural" << std::endl;
        return 1;
    }

    std::string mode = argv[1];
    std::string spec_name = argv[2];

    if (mode == "compare")
        RunCompare(spec_name);
    else if (mode == "compare_k")
        RunCompareK(spec_name);
    else if (mode == "structural")
        RunStructural(spec_name);
    else
    {
        system << "Unknown mode: " << mode << std::endl;
        return 1;
    }

    return 0;
}

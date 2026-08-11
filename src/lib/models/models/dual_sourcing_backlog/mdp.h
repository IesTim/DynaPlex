#pragma once
#include "dynaplex/dynaplex_model_includes.h"
#include "dynaplex/modelling/discretedist.h"
#include "dynaplex/modelling/queue.h"

namespace DynaPlex::Models {
    namespace dual_sourcing_backlog {

        class MDP {
            public:
            int64_t K;
            int64_t l_min, l_max;
            int64_t max_lr;
            double min_h, max_h;
            double min_b, max_b;
            double min_c, max_c;
            double min_mu, max_mu;
            double discount_factor;
            int64_t MaxOrderSize;
            std::vector<std::vector<int64_t>> valid_lead_time_tuples;

            bool use_fixed_instance = false;
            double fixed_h, fixed_b, fixed_mu, fixed_sigma;
            std::vector<double> fixed_c;
            std::vector<int64_t> fixed_l;

            double inventory_cap_multiplier = 0.0;

            // If non-empty, mu is sampled uniformly from this explicit discrete set instead of
            // continuously from [min_mu, max_mu] - for testing generalization across a small,
            // known, finite set of instances rather than a continuous range.
            std::vector<double> mu_values;
            // Same idea, extended to the other instance-defining parameters, so that
            // combinations of them can be varied together across a small discrete set of
            // instances (rather than each independently continuous). All non-empty *_values
            // vectors present in a config must have the same instance count (c_values/l_values
            // flattened with stride K); a single shared index selects the instance each episode,
            // so e.g. mu_values[i] and b_values[i] together describe instance i.
            std::vector<double> b_values;
            std::vector<double> h_values;
            std::vector<double> c_values;   // flattened, stride K
            std::vector<int64_t> l_values;  // flattened, stride K

            struct State {
                DynaPlex::StateCategory cat;
                Queue<int64_t> state_vector;
                int64_t total_inv;
                int64_t K;
                std::vector<int64_t> l;
                std::vector<double> c;
                double mu, sigma;

                double h, b;

                std::vector<double> demand_cdf;
                int64_t demand_min;

                int64_t current_source;
                std::vector<int64_t> pending_orders;

                int64_t MaxOrderSize;

                int64_t backlog_floor;
                int64_t inventory_ceiling;

                DynaPlex::VarGroup ToVarGroup() const;

                int64_t n_obs;
                double sum_demand;
                double sum_sq_demand;
                double mu_hat;
                double sigma_hat;
            };

            using Event = int64_t;
            
            double ModifyStateWithAction(State&, int64_t action) const;
            double ModifyStateWithEvent(State&, const Event&) const;
            Event GetEvent(const State&, DynaPlex::RNG&) const;
            DynaPlex::VarGroup GetStaticInfo() const;
            DynaPlex::StateCategory GetStateCategory(const State&) const;
            bool IsAllowedAction(const State&, int64_t action) const;
            State GetInitialState(DynaPlex::RNG& rng) const;
            State GetState(const DynaPlex::VarGroup&) const;
            void GetFeatures(const State&, DynaPlex::Features&) const;
            // Rollout-control overrides consulted by DCL's SampleGenerator (see mdpadapter.h's
            // HasStateDependendentH/M/L/RestartCounter dispatch). Without these, SampleGenerator
            // falls back to reinitiate_counter's default of ~1e6 periods, i.e. trajectories are
            // effectively never restarted; in a backlog model that lets any policy-driven drift
            // into a large backlog compound for the rest of the generation instead of resetting.
            int64_t GetH(const State&) const;
            int64_t GetM(const State&) const;
            int64_t GetL(const State&) const;
            int64_t GetReinitiateCounter(const State&) const;
            explicit MDP(const DynaPlex::VarGroup&);
            void RegisterPolicies(DynaPlex::Erasure::PolicyRegistry<MDP>&) const;
            std::string action_representation;  // "flat_joint", "sequential"
            bool use_estimation;
        };
    }
}
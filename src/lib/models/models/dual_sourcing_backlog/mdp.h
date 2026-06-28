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

                DynaPlex::VarGroup ToVarGroup() const;
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
            explicit MDP(const DynaPlex::VarGroup&);
            void RegisterPolicies(DynaPlex::Erasure::PolicyRegistry<MDP>&) const;
        };
    }
}
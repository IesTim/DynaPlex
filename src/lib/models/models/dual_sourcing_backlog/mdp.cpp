#include "mdp.h"
#include "dynaplex/erasure/mdpregistrar.h"
#include "policies.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace DynaPlex::Models {
    namespace dual_sourcing_backlog {

        void Register(DynaPlex::Registry& registry) {
            DynaPlex::Erasure::MDPRegistrar<MDP>::RegisterModel(
                "dual_sourcing_backlog",
                "Dual sourcing inventory control with backlogging, K sources, zero-shot generalisation via TED framework.",
                registry);
        }

        void MDP::RegisterPolicies(DynaPlex::Erasure::PolicyRegistry<MDP>& registry) const {
            registry.Register<CDIPolicy>("cdi", "Constant Dual Index policy with parameters S_r and S_e.");
            registry.Register<DIPolicy>("di", "Dual Index policy with single parameter S.");
            registry.Register<CDIPolicy>("si", "Single Index policy with orders only from regular source.");
            registry.Register<TBSPolicy>("tbs", "Tailored Base-Surge policy with parameters S_e and c.");
        }

        MDP::MDP(const DynaPlex::VarGroup& config) {
            config.Get("K", K);
            config.Get("l_min", l_min);
            config.Get("l_max", l_max);
            config.Get("min_h", min_h);
            config.Get("max_h", max_h);
            config.Get("min_b", min_b);
            config.Get("max_b", max_b);
            config.Get("min_c", min_c);
            config.Get("max_c", max_c);
            config.Get("min_mu", min_mu);
            config.Get("max_mu", max_mu);
            if (config.HasKey("action_representation"))
                config.Get("action_representation", action_representation);
            else
                action_representation = "flat_joint";

            if (config.HasKey("use_estimation"))
                config.Get("use_estimation", use_estimation);
            else
                use_estimation = false;

            // checks:
            if (K < 2)
                throw DynaPlex::Error("dual_sourcing_backlog: K must be >= 2.");
            if (l_min < 1)
                throw DynaPlex::Error("dual_sourcing_backlog: l_min must be >= 1.");
            if (l_max <= l_min)
                throw DynaPlex::Error("dual_sourcing_backlog: l_max must be > l_min.");
            if (min_c < 0)
                throw DynaPlex::Error("dual_sourcing_backlog: min_c must be >= 0.");
            if (max_c <= min_c)
                throw DynaPlex::Error("dual_sourcing_backlog: max_c must be > min_c.");
            if (min_mu <= 0.0)
                throw DynaPlex::Error("dual_sourcing_backlog: min_mu must be > 0.");

            max_lr = l_max;
            

            // order quantity upperbound on m
            double worst_sigma = max_mu * 2.0;
            DiscreteDist single_dist = DiscreteDist::GetAdanEenigeResingDist(max_mu, worst_sigma);
            DiscreteDist demand_over_lr = DiscreteDist::GetZeroDist();
            for (int64_t i = 0; i < max_lr; i++)
                demand_over_lr = demand_over_lr.Add(single_dist);
            
            int64_t MaxOrderSize;
            config.Get("max_order_size", MaxOrderSize);

            // Create all K sized subsets
            std::vector<int64_t> current_tuple;
            std::function<void(int64_t, int64_t)> generate = [&](int64_t start, int64_t depth) {
                if (depth == K) {
                    valid_lead_time_tuples.push_back(current_tuple);
                    return;
                }
                for (int64_t l = start; l <= l_max - (K - depth - 1); l++) {
                    current_tuple.push_back(l);
                    generate(l + 1, depth + 1);
                    current_tuple.pop_back();
                }
            };
            generate(l_min, 0);

            if (valid_lead_time_tuples.empty())
                throw DynaPlex::Error(
                    "dual_sourcing_backlog: No valude lead time tupples found. Check that l_max - l_min > = K - 1."
                );
        }

        DynaPlex::VarGroup MDP::GetStaticInfo() const {
            VarGroup vars;

            int64_t valid_actions;
            if (action_representation == "sequential")
                valid_actions = MaxOrderSize + 1;
            else {
                valid_actions = 1;
                for (int64_t k = 0; k < K; k++)
                    valid_actions *= (MaxOrderSize + 1);
            }

            vars.Add("valid_actions", MaxOrderSize + 1);
            vars.Add("discount_factor", 1.0);
            vars.Add("horizon_type", "infinite");
            
            VarGroup diagnostics{};
            diagnostics.Add("action_representation", action_representation);
            diagnostics.Add("MaxOrderSize", MaxOrderSize);
            vars.Add("diagnostics", diagnostics);
            
            return vars;
        }

        DynaPlex::StateCategory MDP::GetStateCategory(const State& state) const {
            return state.cat;
        }

        bool MDP::IsAllowedAction(const State& state, int64_t action) const {
            return true;
        }

        MDP::Event MDP::GetEvent(const State& state, DynaPlex::RNG& rng) const {
            double u = rng.genUniform();
            auto it = std::lower_bound(state.demand_cdf.begin(), state.demand_cdf.end(), u);
        
            return state.demand_min + static_cast<int64_t>(std::distance(state.demand_cdf.begin(), it));
        }

        double MDP::ModifyStateWithAction(State& state, int64_t action) const {
            std::vector<int64_t> q(K, 0);

            if (action_representation == "sequential") {
                int64_t k = state.current_source;
                q[k] = action;
            
                state.current_source++;
                if (state.current_source < K) {
                    state.cat = StateCategory::AwaitAction();
                    double cost = state.c[k] * q[k];
                    state.state_vector.at(state.l[k] - 1) += q[k];
                    state.total_inv += q[k];
                    return cost;
                } else {
                    state.current_source = 0;
                    state.cat = StateCategory::AwaitEvent();
                    double cost = state.c[k] * q[k];
                    state.state_vector.at(state.l[k] - 1) += q[k];
                    state.total_inv += q[k];
                    return cost;
                }
            } else {
                int64_t m_plus_1 = MaxOrderSize + 1;
                int64_t remaining = action;
                for (int64_t k = K - 1; k >= 0; k--) {
                    q[k] = remaining % m_plus_1;
                    remaining /= m_plus_1;
                }

                double cost = 0.0;
                for (int64_t k = 0; k < K; k++) {
                    cost += state.c[k] * q[k];
                    state.state_vector.at(state.l[k] - 1) += q[k];
                    state.total_inv += q[k];
                }

                state.cat = StateCategory::AwaitEvent();
                return cost;
            }


            return 0.0;
        }

        double MDP::ModifyStateWithEvent(State& state, const Event& event) const {
            int64_t inventory = state.state_vector.pop_front();

            inventory -= event;
            state.total_inv -= event;

            state.n_obs++;
            state.sum_demand += static_cast<double>(event);
            state.sum_sq_demand += static_cast<double>(event * event);
            state.mu_hat = state.sum_demand / state.n_obs;

            double mean_sq = state.sum_sq_demand / state.n_obs;
            double var_hat = std::max(mean_sq - state.mu_hat * state.mu_hat, DiscreteDist::LeastVarianceRequiredForAERFit(state.mu_hat));
            state.sigma_hat = std::sqrt(var_hat);

            double cost = state.h * static_cast<double>(std::max(static_cast<int64_t>(0), inventory)) + state.b * static_cast<double>(std::max(static_cast<int64_t>(0), -inventory));

            state.state_vector.push_back(0);
            state.state_vector.front() += inventory;

            state.total_inv = state.state_vector.sum();

            state.current_source = 0;
            state.cat = StateCategory::AwaitAction();
            return cost;
        }

        void MDP::GetFeatures(const State& state, DynaPlex::Features& features) const {
            features.Add(state.state_vector);

            features.Add(state.mu_hat);
            features.Add(state.sigma_hat);

            features.Add(state.h);
            features.Add(state.b);

            for (int64_t k = 0; k < K; k++) {
                features.Add(state.c[k]);
                features.Add(static_cast<double>(state.l[k]));
            }

            features.Add(static_cast<double>(state.current_source));
        }

        MDP::State MDP::GetState(const DynaPlex::VarGroup& vars) const {
            State state{};
            vars.Get("cat", state.cat);
            vars.Get("state_vector", state.state_vector);
            vars.Get("total_inv", state.total_inv);
            vars.Get("K", state.K);
            vars.Get("l", state.l);
            vars.Get("c", state.c);
            vars.Get("mu", state.mu);
            vars.Get("sigma", state.sigma);
            vars.Get("h", state.h);
            vars.Get("b", state.b);
            vars.Get("demand_cdf", state.demand_cdf);
            vars.Get("demand_min", state.demand_min);
            vars.Get("current_source", state.current_source);
            vars.Get("pending_orders", state.pending_orders);
            vars.Get("MaxOrderSize", state.MaxOrderSize);
            vars.Get("n_obs", state.n_obs);
            vars.Get("sum_demand", state.sum_demand);
            vars.Get("sum_sq_demand", state.sum_sq_demand);
            vars.Get("mu_hat", state.mu_hat);
            vars.Get("sigma_hat", state.sigma_hat);
            return state;
        }

        DynaPlex::VarGroup MDP::State::ToVarGroup() const {
            DynaPlex::VarGroup vars;
            vars.Add("cat", cat);
            vars.Add("state_vector", state_vector);
            vars.Add("total_inv", total_inv);
            vars.Add("K", K);
            vars.Add("l", l);
            vars.Add("c", c);
            vars.Add("mu", mu);
            vars.Add("sigma", sigma);
            vars.Add("h", h);
            vars.Add("b", b);
            vars.Add("demand_cdf", demand_cdf);
            vars.Add("demand_min", demand_min);
            vars.Add("current_source", current_source);
            vars.Add("pending_orders", pending_orders);
            vars.Add("MaxOrderSize", MaxOrderSize);
            vars.Add("n_obs", n_obs);
            vars.Add("sum_demand", sum_demand);
            vars.Add("sum_sq_demand", sum_sq_demand);
            vars.Add("mu_hat", mu_hat);
            vars.Add("sigma_hat", sigma_hat);
            return vars;
        }

        MDP::State MDP::GetInitialState(DynaPlex::RNG& rng) const {
            State state{};

            int64_t tuple_idx = static_cast<int64_t>(std::floor(rng.genUniform() * valid_lead_time_tuples.size()));
            state.l = valid_lead_time_tuples[tuple_idx];

            state.h = min_h + rng.genUniform() * (max_h - min_h);
            state.b = min_b + rng.genUniform() * (max_b - min_b);

            state.c.resize(K);
            for (int64_t k = 0; k < K; k++)
                state.c[k] = min_c + rng.genUniform() * (max_c - min_c);
            std::sort(state.c.begin(), state.c.end(), std::greater<double>());
            
            state.mu = min_mu + rng.genUniform() * (max_mu - min_mu);
            double min_sigma = std::sqrt(DiscreteDist::LeastVarianceRequiredForAERFit(state.mu));
            double max_sigma = state.mu * 2.0;
            state.sigma = min_sigma + rng.genUniform() * (max_sigma - min_sigma);
            
            DiscreteDist demand_dist = DiscreteDist::GetAdanEenigeResingDist(state.mu, state.sigma);
            state.demand_min = demand_dist.Min();
            state.demand_cdf.clear();
            double cumulative = 0.0;
            for (const auto& [value, prob] : demand_dist) {
                cumulative += prob;
                state.demand_cdf.push_back(cumulative);
            }

            state.K = K;
            state.MaxOrderSize = MaxOrderSize;

            auto queue = Queue<int64_t>{};
            queue.reserve(max_lr);
            for (int64_t i = 0; i < max_lr; i++) {
                queue.push_back(0);
            }
            state.state_vector = queue;
            state.total_inv = 0;

            state.current_source = 0;
            state.pending_orders.resize(K, 0);

            state.cat = StateCategory::AwaitAction();

            state.mu_hat = (min_mu + max_mu) / 2;
            state.sigma_hat = (min_mu + max_mu) / 2.0;
            state.n_obs = 0;
            state.sum_demand = 0.0;
            state.sum_sq_demand = 0.0;  
            return state;

        }
    }
}
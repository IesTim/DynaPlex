#include "mdp.h"
#include "dynaplex/erasure/mdpregistrar.h"
#include "policies.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace DynaPlex::Models {
    namespace dual_sourcing_backlog {

        namespace {
            int64_t NewsvendorFractile(double mu, double sigma, double fractile, int64_t periods) {
                DiscreteDist dist = DiscreteDist::GetAdanEenigeResingDist(mu, sigma);
                DiscreteDist demand_over_lt = DiscreteDist::GetZeroDist();
                for (int64_t i = 0; i < periods; i++)
                    demand_over_lt = demand_over_lt.Add(dist);
                return demand_over_lt.Fractile(fractile);
            }
        }

        void Register(DynaPlex::Registry& registry) {
            DynaPlex::Erasure::MDPRegistrar<MDP>::RegisterModel(
                "dual_sourcing_backlog",
                "Dual sourcing inventory control with backlogging, K sources, zero-shot generalisation via TED framework.",
                registry);
        }

        int64_t MDP::GetH(const State&) const { return rollout_H; }
        int64_t MDP::GetM(const State&) const { return rollout_M; }
        int64_t MDP::GetL(const State&) const { return rollout_L; }
        int64_t MDP::GetReinitiateCounter(const State&) const { return rollout_reinitiate_counter; }

        void MDP::RegisterPolicies(DynaPlex::Erasure::PolicyRegistry<MDP>& registry) const {
            registry.Register<CDIPolicy>("cdi", "Constant Dual Index policy with parameters S_r and S_e.");
            registry.Register<DIPolicy>("di", "Dual Index policy with single parameter S.");
            registry.Register<SIPolicy>("si", "Single Index policy with orders only from regular source.");
            registry.Register<TBSPolicy>("tbs", "Tailored Base-Surge policy with parameters S_e and c.");
            registry.Register<BaseStockPolicy>("base_stock", "Standard single-channel order-up-to-S policy. The correct benchmark for K=1.");
            registry.Register<AdaptiveCDIPolicy>("adaptive_cdi", "CDI with S_r/S_e recomputed each decision from the state's own instance parameters. K=2 only; used as an instance-agnostic behavior/initial policy for wide-distribution DCL training.");
            registry.Register<AdaptiveBaseStockPolicy>("adaptive_base_stock", "base_stock with S recomputed each decision from the state's own instance parameters. K=1 only; used as an instance-agnostic behavior/initial policy for wide-distribution DCL training.");
            registry.Register<AdaptiveKSourceCDIPolicy>("adaptive_cdi_k", "K-generic greedy generalization of adaptive_cdi for K>=1 (not a claim of optimality, see policies.h doc comment). Used as behavior/initial policy and evaluation baseline for the K-scaling stress test.");
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
            config.Get("max_order_size", MaxOrderSize);
            if (config.HasKey("action_representation"))
                config.Get("action_representation", action_representation);
            else
                action_representation = "flat_joint";

            if (config.HasKey("use_estimation"))
                config.Get("use_estimation", use_estimation);
            else
                use_estimation = false;

            if (config.HasKey("inventory_cap_multiplier"))
                config.Get("inventory_cap_multiplier", inventory_cap_multiplier);
            else
                inventory_cap_multiplier = 0.0;

            if (config.HasKey("mu_values"))
                config.Get("mu_values", mu_values);
            if (config.HasKey("b_values"))
                config.Get("b_values", b_values);
            if (config.HasKey("h_values"))
                config.Get("h_values", h_values);
            if (config.HasKey("c_values"))
                config.Get("c_values", c_values);
            if (config.HasKey("l_values"))
                config.Get("l_values", l_values);

            config.GetOrDefault("oracle_mu_init", oracle_mu_init, oracle_mu_init);

            config.GetOrDefault("rollout_H", rollout_H, rollout_H);
            config.GetOrDefault("rollout_M", rollout_M, rollout_M);
            config.GetOrDefault("rollout_L", rollout_L, rollout_L);
            config.GetOrDefault("rollout_reinitiate_counter", rollout_reinitiate_counter, rollout_reinitiate_counter);

            use_fixed_instance = false;
            if (config.HasKey("fixed_instance"))
            {
                use_fixed_instance = true;
                VarGroup fi;
                config.Get("fixed_instance", fi);
                fi.Get("h", fixed_h);
                fi.Get("b", fixed_b);
                fi.Get("mu", fixed_mu);
                fi.Get("sigma", fixed_sigma);
                // General-K form (fixed_instance.l = [l_0, ..., l_{K-1}], increasing) if present,
                // otherwise the original K=2-only form (l_e/l_r) for backward compatibility with
                // every existing config that uses it.
                if (fi.HasKey("l"))
                    fi.Get("l", fixed_l);
                else
                {
                    int64_t l_e, l_r;
                    fi.Get("l_e", l_e);
                    fi.Get("l_r", l_r);
                    fixed_l = {l_e, l_r};
                }
                fixed_c.resize(K);
                std::vector<double> costs;
                fi.Get("costs", costs);
                for (int64_t k = 0; k < K; k++)
                    fixed_c[k] = costs[k];
            }

            // checks:
            if (K < 1)
                throw DynaPlex::Error("dual_sourcing_backlog: K must be >= 1.");
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

            vars.Add("valid_actions", valid_actions);
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

            auto clip_to_ceiling = [&](int64_t q_k) {
                int64_t projected_total = state.total_inv + q_k;
                if (projected_total > state.inventory_ceiling)
                    q_k = std::max(int64_t(0), state.inventory_ceiling - state.total_inv);
                return q_k;
            };

            if (action_representation == "sequential") {
                int64_t k = state.current_source;
                q[k] = clip_to_ceiling(action);

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
                    q[k] = clip_to_ceiling(q[k]);
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
            if (inventory < state.backlog_floor)
                inventory = state.backlog_floor;
            state.total_inv -= event;

            // In oracle mode mu_hat/sigma_hat are pinned to the instance's true mu/sigma (set in
            // GetInitialState) and never overwritten by the running estimate below - this gives
            // the "perfect demand-rate knowledge" upper-bound comparison point. n_obs/sum_demand
            // are still tracked for logging/consistency but do not feed back into mu_hat/sigma_hat.
            state.n_obs++;
            state.sum_demand += static_cast<double>(event);
            state.sum_sq_demand += static_cast<double>(event * event);
            if (!oracle_mu_init)
            {
                state.mu_hat = state.sum_demand / state.n_obs;

                double mean_sq = state.sum_sq_demand / state.n_obs;
                double var_hat = std::max(mean_sq - state.mu_hat * state.mu_hat, DiscreteDist::LeastVarianceRequiredForAERFit(state.mu_hat));
                state.sigma_hat = std::sqrt(var_hat);
            }

            double cost = state.h * static_cast<double>(std::max(static_cast<int64_t>(0), inventory)) + state.b * static_cast<double>(std::max(static_cast<int64_t>(0), -inventory));

            state.state_vector.push_back(0);
            state.state_vector.front() += inventory;

            state.total_inv = state.state_vector.sum();

            state.current_source = 0;
            state.cat = StateCategory::AwaitAction();
            return cost;
        }

        void MDP::GetFeatures(const State& state, DynaPlex::Features& features) const {
            double scale = std::max(state.mu_hat, 0.1);

            for (const auto& v : state.state_vector)
                features.Add(static_cast<double>(v) / scale);
            features.Add(static_cast<double>(state.total_inv) / scale);
            features.Add(state.mu_hat);
            features.Add(state.sigma_hat);

            features.Add(state.h);
            features.Add(state.b / (state.b + state.h));

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
            vars.Get("backlog_floor", state.backlog_floor);
            vars.Get("inventory_ceiling", state.inventory_ceiling);
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
            vars.Add("backlog_floor", backlog_floor);
            vars.Add("inventory_ceiling", inventory_ceiling);
            vars.Add("n_obs", n_obs);
            vars.Add("sum_demand", sum_demand);
            vars.Add("sum_sq_demand", sum_sq_demand);
            vars.Add("mu_hat", mu_hat);
            vars.Add("sigma_hat", sigma_hat);
            return vars;
        }

        MDP::State MDP::GetInitialState(DynaPlex::RNG& rng) const {
            State state{};

            if (use_fixed_instance)
            {
                state.l = fixed_l;
                state.h = fixed_h;
                state.b = fixed_b;
                state.c = fixed_c;
                state.mu = fixed_mu;
                state.sigma = fixed_sigma;
            }
            else
            {
                // A single shared index picks the instance for whichever *_values vectors are
                // present in this config, so e.g. mu_values[i] and b_values[i] jointly describe
                // instance i; parameters without a *_values vector keep sampling continuously.
                int64_t instance_count = -1;
                if (!mu_values.empty()) instance_count = static_cast<int64_t>(mu_values.size());
                if (!b_values.empty()) instance_count = static_cast<int64_t>(b_values.size());
                if (!h_values.empty()) instance_count = static_cast<int64_t>(h_values.size());
                if (!c_values.empty()) instance_count = static_cast<int64_t>(c_values.size() / K);
                if (!l_values.empty()) instance_count = static_cast<int64_t>(l_values.size() / K);

                int64_t idx = 0;
                if (instance_count > 0)
                    idx = std::min(static_cast<int64_t>(std::floor(rng.genUniform() * instance_count)), instance_count - 1);

                if (!l_values.empty())
                    state.l.assign(l_values.begin() + idx * K, l_values.begin() + (idx + 1) * K);
                else
                {
                    int64_t tuple_idx = static_cast<int64_t>(std::floor(rng.genUniform() * valid_lead_time_tuples.size()));
                    state.l = valid_lead_time_tuples[tuple_idx];
                }

                state.h = !h_values.empty() ? h_values[idx] : (min_h + rng.genUniform() * (max_h - min_h));
                state.b = !b_values.empty() ? b_values[idx] : (min_b + rng.genUniform() * (max_b - min_b));

                state.c.resize(K);
                if (!c_values.empty())
                {
                    for (int64_t k = 0; k < K; k++)
                        state.c[k] = c_values[idx * K + k];
                }
                else
                {
                    for (int64_t k = 0; k < K; k++)
                        state.c[k] = min_c + rng.genUniform() * (max_c - min_c);
                }
                std::sort(state.c.begin(), state.c.end(), std::greater<double>());

                if (!mu_values.empty())
                    state.mu = mu_values[idx];
                else
                    state.mu = min_mu + rng.genUniform() * (max_mu - min_mu);
            }


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

            if (inventory_cap_multiplier > 0.0)
            {
                int64_t max_l = *std::max_element(state.l.begin(), state.l.end());
                double fractile = state.b / (state.b + state.h);
                int64_t newsvendor_s = NewsvendorFractile(state.mu, state.sigma, fractile, max_l);
                state.backlog_floor = -static_cast<int64_t>(inventory_cap_multiplier * static_cast<double>(newsvendor_s));
                state.inventory_ceiling = static_cast<int64_t>(inventory_cap_multiplier * static_cast<double>(newsvendor_s));
            }
            else
            {
                state.backlog_floor = std::numeric_limits<int64_t>::min();
                state.inventory_ceiling = std::numeric_limits<int64_t>::max();
            }

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

            if (oracle_mu_init)
            {
                state.mu_hat = state.mu;
                state.sigma_hat = state.sigma;
            }
            else
            {
                state.mu_hat = (min_mu + max_mu) / 2;
                state.sigma_hat = (min_mu + max_mu) / 2.0;
            }
            state.n_obs = 0;
            state.sum_demand = 0.0;
            state.sum_sq_demand = 0.0;  
            return state;

        }
    }
}
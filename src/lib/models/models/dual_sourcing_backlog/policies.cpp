#include "policies.h"
#include "mdp.h"
#include "dynaplex/error.h"
#include <algorithm>
#include <cmath>
#include <map>
#include <tuple>

namespace DynaPlex::Models {
    namespace dual_sourcing_backlog {

        namespace {
            // Normal-approximation fallback newsvendor bound, used when GetAdanEenigeResingDist
            // throws. AdaptiveCDIPolicy/AdaptiveBaseStockPolicy call the AER-based fractile with
            // live, arbitrary (mu_hat, sigma_hat) pairs; GetAdanEenigeResingDist's underdispersed
            // (a<0) branch has a latent numerical edge case (some parameter combinations produce
            // a binomial p slightly outside [0,1], throwing "Invalid parameters for binomial
            // distribution") that no other caller hits because they use fixed, pre-validated
            // instance parameters rather than live running estimates. Rather than chase that
            // numerical edge case in shared distribution code, fall back to a simple approximation
            // here - this is only ever used to drive DCL's initial/behavior-policy rollouts, not
            // the final learned policy, so approximate robustness matters far more than precision.
            double NormalQuantileApprox(double p) {
                // Beasley-Springer-Moro-style rational approximation of the inverse normal CDF.
                static const double a[] = { -3.969683028665376e+01, 2.209460984245205e+02,
                    -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01,
                    2.506628277459239e+00 };
                static const double b[] = { -5.447609879822406e+01, 1.615858368580409e+02,
                    -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01 };
                static const double c[] = { -7.784894002430293e-03, -3.223964580411365e-01,
                    -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00,
                    2.938163982698783e+00 };
                static const double d[] = { 7.784695709041462e-03, 3.224671290700398e-01,
                    2.445134137142996e+00, 3.754408661907416e+00 };
                p = std::clamp(p, 1e-9, 1.0 - 1e-9);
                double q, r;
                if (p < 0.02425) {
                    q = std::sqrt(-2.0 * std::log(p));
                    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
                        ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
                }
                else if (p <= 0.97575) {
                    q = p - 0.5;
                    r = q * q;
                    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
                        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
                }
                else {
                    q = std::sqrt(-2.0 * std::log(1.0 - p));
                    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
                        ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
                }
            }

            int64_t NormalApproxFractile(double mu, double sigma, double fractile, int64_t periods) {
                double mean = mu * static_cast<double>(periods);
                double sd = sigma * std::sqrt(static_cast<double>(periods));
                double z = NormalQuantileApprox(fractile);
                return std::max(int64_t(0), static_cast<int64_t>(std::llround(mean + z * sd)));
            }

            int64_t AdaptiveNewsvendorFractile(double mu, double sigma, double fractile, int64_t periods) {
                try {
                    DiscreteDist dist = DiscreteDist::GetAdanEenigeResingDist(mu, sigma);
                    DiscreteDist demand_over_lt = DiscreteDist::GetZeroDist();
                    for (int64_t i = 0; i < periods; i++)
                        demand_over_lt = demand_over_lt.Add(dist);
                    return demand_over_lt.Fractile(fractile);
                }
                catch (const DynaPlex::Error&) {
                    return NormalApproxFractile(mu, sigma, fractile, periods);
                }
            }

            // AdaptiveNewsvendorFractile does an O(periods) chain of O(n^2) distribution
            // convolutions. AdaptiveCDIPolicy/AdaptiveBaseStockPolicy call it on every single
            // decision, including every rollout-continuation step during DCL sample generation
            // (H*M*candidates*N calls) - uncached, this makes wide-instance-distribution DCL
            // training computationally infeasible (billions of uncached convolutions). Cache the
            // result per-thread, keyed on (mu, sigma, fractile) rounded to a coarse grid: the
            // newsvendor bound is itself an approximate heuristic, so this quantization doesn't
            // meaningfully change the policy while making repeat calls for similar (mu_hat,
            // sigma_hat) O(1).
            int64_t CachedAdaptiveNewsvendorFractile(double mu, double sigma, double fractile, int64_t periods) {
                // Rounding to 1 decimal on mu/sigma plus 3 decimals on fractile gives a key space
                // of ~1e8 - across a wide instance distribution (mu/sigma/b all varying per
                // sample) that's far larger than the number of calls made, so the cache almost
                // never hits. Round to whole units on mu/sigma and 2 decimals on fractile instead
                // (~1e5 keys): still a fine enough grid for an approximate newsvendor heuristic
                // used only to drive initial/behavior-policy rollouts, but small enough to
                // actually get reused.
                static thread_local std::map<std::tuple<int64_t, int64_t, int64_t, int64_t>, int64_t> cache;
                auto key = std::make_tuple(
                    static_cast<int64_t>(std::llround(mu)),
                    static_cast<int64_t>(std::llround(sigma)),
                    static_cast<int64_t>(std::llround(fractile * 100.0)),
                    periods);
                auto it = cache.find(key);
                if (it != cache.end())
                    return it->second;
                int64_t result = AdaptiveNewsvendorFractile(mu, sigma, fractile, periods);
                cache.emplace(key, result);
                return result;
            }
        }

        CDIPolicy::CDIPolicy(std::shared_ptr<const MDP> mdp, const VarGroup& config): mdp{ mdp } {
            config.GetOrDefault("S_r", S_r, mdp->MaxOrderSize);
            config.GetOrDefault("S_e", S_e, mdp->MaxOrderSize / 2);
        }

        int64_t CDIPolicy::GetAction(const MDP::State& state) const {
            int64_t IP_e = state.state_vector.front();
            int64_t IP_r = state.total_inv;

            int64_t q_e = std::max(int64_t(0), S_e - IP_e);
            q_e = std::min(q_e, mdp->MaxOrderSize);

            int64_t q_r = std::max(int64_t(0), S_r - (IP_r + q_e));
            q_r = std::min(q_r, mdp->MaxOrderSize);

            if (mdp->action_representation == "sequential")
                return state.current_source == 0 ? q_e : q_r;
            return q_e * (mdp->MaxOrderSize + 1) + q_r;
        }

        SIPolicy::SIPolicy(std::shared_ptr<const MDP> mdp, const VarGroup& config): mdp{ mdp } {
            config.GetOrDefault("S", S, mdp->MaxOrderSize);
        }

        int64_t SIPolicy::GetAction(const MDP::State& state) const {
            int64_t IP = state.total_inv;

            int64_t q_r = std::max(int64_t(0), S - IP);
            q_r = std::min(q_r, mdp->MaxOrderSize);

            if (mdp->action_representation == "sequential")
                return state.current_source == 0 ? 0 : q_r;
            return q_r;
        }

        DIPolicy::DIPolicy(std::shared_ptr<const MDP> mdp, const VarGroup& config): mdp{ mdp } {
            config.GetOrDefault("S", S, mdp->MaxOrderSize);
        }

        int64_t DIPolicy::GetAction(const MDP::State& state) const {
            int64_t IP_r = state.total_inv;
            int64_t IP_e = state.state_vector.front();

            int64_t q_e = std::max(int64_t(0), S - IP_e);
            q_e = std::min(q_e, mdp->MaxOrderSize);

            int64_t q_r = std::max(int64_t(0), S - (IP_r + q_e));
            q_r = std::min(q_r, mdp->MaxOrderSize);

            if (mdp->action_representation == "sequential")
                return state.current_source == 0 ? q_e : q_r;
            return q_e * (mdp->MaxOrderSize + 1) + q_r;
        }

        TBSPolicy::TBSPolicy(std::shared_ptr<const MDP> mdp, const VarGroup& config) : mdp{ mdp } {
            config.GetOrDefault("S_e", S_e, mdp->MaxOrderSize / 2);
            config.GetOrDefault("c", c, mdp->MaxOrderSize / 4);
        }

        int64_t TBSPolicy::GetAction(const MDP::State& state) const {
            int64_t IP = state.total_inv;

            int64_t q_r = std::min(c, mdp->MaxOrderSize);
            int64_t q_e = std::max(int64_t(0), S_e - IP);
            q_e = std::min(q_e, mdp->MaxOrderSize);

            if (mdp->action_representation == "sequential")
                return state.current_source == 0 ? q_e : q_r;
            return q_e * (mdp->MaxOrderSize + 1) + q_r;
        }

        BaseStockPolicy::BaseStockPolicy(std::shared_ptr<const MDP> mdp, const VarGroup& config): mdp{ mdp } {
            config.GetOrDefault("S", S, mdp->MaxOrderSize);
        }

        int64_t BaseStockPolicy::GetAction(const MDP::State& state) const {
            int64_t IP = state.total_inv;
            int64_t q = std::max(int64_t(0), S - IP);
            q = std::min(q, mdp->MaxOrderSize);
            return q;
        }

        AdaptiveCDIPolicy::AdaptiveCDIPolicy(std::shared_ptr<const MDP> mdp, const VarGroup& config): mdp{ mdp } {
            if (mdp->K != 2)
                throw DynaPlex::Error("AdaptiveCDIPolicy: only implemented for K=2 (its q_e/q_r cascade does not generalize to other K).");
        }

        int64_t AdaptiveCDIPolicy::GetAction(const MDP::State& state) const {
            // state.l = {l_e, l_r} (see MDP::RegisterPolicies / fixed_instance parsing convention).
            int64_t l_e = state.l[0];
            int64_t l_r = state.l[1];
            double fractile = state.b / (state.b + state.h);

            int64_t S_r = CachedAdaptiveNewsvendorFractile(state.mu_hat, state.sigma_hat, fractile, l_r);
            int64_t S_e = std::max(int64_t(1),
                CachedAdaptiveNewsvendorFractile(state.mu_hat, state.sigma_hat, fractile, l_e) / (l_r - l_e + 1));

            int64_t IP_e = state.state_vector.front();
            int64_t IP_r = state.total_inv;

            int64_t q_e = std::max(int64_t(0), S_e - IP_e);
            q_e = std::min(q_e, mdp->MaxOrderSize);

            int64_t q_r = std::max(int64_t(0), S_r - (IP_r + q_e));
            q_r = std::min(q_r, mdp->MaxOrderSize);

            if (mdp->action_representation == "sequential")
                return state.current_source == 0 ? q_e : q_r;
            return q_e * (mdp->MaxOrderSize + 1) + q_r;
        }

        AdaptiveBaseStockPolicy::AdaptiveBaseStockPolicy(std::shared_ptr<const MDP> mdp, const VarGroup& config): mdp{ mdp } {
            if (mdp->K != 1)
                throw DynaPlex::Error("AdaptiveBaseStockPolicy: only meaningful for K=1 (see BaseStockPolicy's own doc comment).");
        }

        int64_t AdaptiveBaseStockPolicy::GetAction(const MDP::State& state) const {
            double fractile = state.b / (state.b + state.h);
            int64_t S = CachedAdaptiveNewsvendorFractile(state.mu_hat, state.sigma_hat, fractile, state.l[0]);

            int64_t IP = state.total_inv;
            int64_t q = std::max(int64_t(0), S - IP);
            q = std::min(q, mdp->MaxOrderSize);
            return q;
        }

        AdaptiveKSourceCDIPolicy::AdaptiveKSourceCDIPolicy(std::shared_ptr<const MDP> mdp, const VarGroup& config): mdp{ mdp } {
            if (mdp->K < 1)
                throw DynaPlex::Error("AdaptiveKSourceCDIPolicy: K must be >= 1.");
        }

        int64_t AdaptiveKSourceCDIPolicy::GetAction(const MDP::State& state) const {
            int64_t K = mdp->K;
            double fractile = state.b / (state.b + state.h);

            // state.l/state.c are ordered fastest/priciest (k=0) to slowest/cheapest (k=K-1) by
            // construction (see MDP::GetInitialState). Processing sources in that order lets each
            // one target NewsvendorFractile(l_k)/K while only ordering the gap left after faster
            // sources' quantities (decided earlier in this same cascade) are already accounted for.
            std::vector<int64_t> q(K, 0);
            int64_t cumulative = state.total_inv;
            for (int64_t k = 0; k < K; k++) {
                int64_t target = std::max(int64_t(1),
                    CachedAdaptiveNewsvendorFractile(state.mu_hat, state.sigma_hat, fractile, state.l[k]) / K);
                int64_t qk = std::max(int64_t(0), target - cumulative);
                qk = std::min(qk, mdp->MaxOrderSize);
                q[k] = qk;
                cumulative += qk;
            }

            if (mdp->action_representation == "sequential")
                return q[state.current_source];

            int64_t action = 0;
            for (int64_t k = 0; k < K; k++)
                action = action * (mdp->MaxOrderSize + 1) + q[k];
            return action;
        }
    }
}
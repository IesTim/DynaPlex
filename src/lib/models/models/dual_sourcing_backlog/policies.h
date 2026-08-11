#pragma once
#include "mdp.h"
#include "dynaplex/vargroup.h"
#include <memory>

namespace DynaPlex::Models {
    namespace dual_sourcing_backlog {

        class MDP;

        struct CDIPolicy {
            std::shared_ptr<const MDP> mdp;
            int64_t S_r;
            int64_t S_e;

            CDIPolicy(std::shared_ptr<const MDP> mdp, const VarGroup& config);
            int64_t GetAction(const MDP::State& state) const;
        };

        struct DIPolicy {
            std::shared_ptr<const MDP> mdp;
            int64_t S;

            DIPolicy(std::shared_ptr<const MDP> mdp, const VarGroup& config);
            int64_t GetAction(const MDP::State& state) const;
        };

        struct SIPolicy {
            std::shared_ptr<const MDP> mdp;
            int64_t S;

            SIPolicy(std::shared_ptr<const MDP> mdp, const VarGroup& config);
            int64_t GetAction(const MDP::State& state) const;
        };

        struct TBSPolicy {
            std::shared_ptr<const MDP> mdp;
            int64_t S_e;
            int64_t c;

            TBSPolicy(std::shared_ptr<const MDP> mdp, const VarGroup& config);
            int64_t GetAction(const MDP::State& state) const;
        };

        // Standard single-channel order-up-to-S policy. The K=2-specific policies above
        // (CDI/DI/SI/TBS) all hardcode an expedited/regular split and are not meaningful
        // benchmarks for K=1. This is the correct, simple reference for that case.
        struct BaseStockPolicy {
            std::shared_ptr<const MDP> mdp;
            int64_t S;

            BaseStockPolicy(std::shared_ptr<const MDP> mdp, const VarGroup& config);
            int64_t GetAction(const MDP::State& state) const;
        };

        // CDI with S_r/S_e recomputed each decision from the state's own (mu_hat, sigma_hat, h, b, l)
        // via the same newsvendor-fractile formula tuning_utils.h's TuneCDI uses as its starting guess.
        // Unlike CDIPolicy (fixed S_r/S_e), this adapts to whatever instance the state belongs to, so
        // it is a reasonable behavior/initial policy for DCL training across a distribution of
        // instances (K=2 only; the cascading q_e/q_r formula does not generalize past two sources).
        struct AdaptiveCDIPolicy {
            std::shared_ptr<const MDP> mdp;

            AdaptiveCDIPolicy(std::shared_ptr<const MDP> mdp, const VarGroup& config);
            int64_t GetAction(const MDP::State& state) const;
        };

        // BaseStockPolicy's instance-adaptive counterpart: S recomputed each decision from the
        // state's own (mu_hat, sigma_hat, h, b, l). Behavior/initial policy for K=1 wide-distribution
        // DCL training, where a single fixed S is not well-suited to varying instances.
        struct AdaptiveBaseStockPolicy {
            std::shared_ptr<const MDP> mdp;

            AdaptiveBaseStockPolicy(std::shared_ptr<const MDP> mdp, const VarGroup& config);
            int64_t GetAction(const MDP::State& state) const;
        };
    }
}
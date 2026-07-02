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
    }
}
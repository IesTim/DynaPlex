#include "policies.h"
#include "mdp.h"
#include "dynaplex/error.h"
#include <algorithm>

namespace DynaPlex::Models {
    namespace dual_sourcing_backlog {
        CDIPolicy::CDIPolicy(std::shared_ptr<const MDP> mdp, const VarGroup& config): mdp{ mdp } {
            config.GetOrDefault("S_r", S_r, mdp->MaxOrderSize);
            config.GetOrDefault("S_e", S_e, mdp->MaxOrderSize / 2);
        }

        int64_t CDIPolicy::GetAction(const MDP::State& state) const {
            int64_t IP = state.total_inv;

            int64_t q_e = std::max(int64_t(0), S_e - IP);
            q_e = std::min(q_e, mdp->MaxOrderSize);

            int64_t q_r = std::max(int64_t(0), S_r - (IP + q_e));
            q_r = std::min(q_r, mdp->MaxOrderSize);

            return q_r * (mdp->MaxOrderSize + 1) + q_e;
        }

        SIPolicy::SIPolicy(std::shared_ptr<const MDP> mdp, const VarGroup& config): mdp{ mdp } {
            config.GetOrDefault("S", S, mdp->MaxOrderSize);
        }

        int64_t SIPolicy::GetAction(const MDP::State& state) const {
            int64_t IP = state.total_inv;

            int64_t q_r = std::max(int64_t(0), S - IP);
            q_r = std::min(q_r, mdp->MaxOrderSize);

            return q_r * (mdp->MaxOrderSize + 1);
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

            return q_r * (mdp->MaxOrderSize + 1) + q_e;
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

            return q_r * (mdp->MaxOrderSize + 1) + q_e;
        }
    }
}
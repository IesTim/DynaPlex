#pragma once
#include <algorithm>
#include "dynaplex/policy.h"
#include "dynaplex/trajectory.h"
#include "dynaplex/vargroup.h"
#include "dynaplex/error.h"
#include "dynaplex/statecategory.h"
#include "stateadapter.h"
#include "erasure_concepts.h"
#include "actionrangeprovider.h"

namespace DynaPlex::Erasure
{
	template<typename t_MDP, typename t_Policy>
	class PolicyAdapter final : public PolicyInterface
	{
		static_assert(HasState<t_MDP>, "MDP must publicly define a nested type or using declaration for State");
		static_assert(HasGetStateCategory<t_MDP>, "MDP must publicly define a function GetStateCategory(const MDP::State) const that returns a StateCategory");
		using t_State = t_MDP::State;

		static_assert(
			(HasGetAction<t_Policy, t_State> && !HasGetActionRNG<t_Policy, t_State, DynaPlex::RNG> && !HasGetActionState<t_Policy, t_State>) ||
			(!HasGetAction<t_Policy, t_State> && HasGetActionRNG<t_Policy, t_State, DynaPlex::RNG> && !HasGetActionState<t_Policy, t_State>) ||
			(!HasGetAction<t_Policy, t_State> && !HasGetActionRNG<t_Policy, t_State, DynaPlex::RNG> && HasGetActionState<t_Policy, t_State>),
			" t_Policy should implement exactly one of the three methods: GetAction(const State), GetAction(const State, RNG), or GetAction(State) method!"
			);

		t_Policy policy;
		std::shared_ptr<const t_MDP> mdp;
		std::string identifier;
		int64_t mdp_int_hash;
		const DynaPlex::VarGroup vars;

	public:
		const DynaPlex::VarGroup& GetConfig() const override
		{
			return vars;
		}

		std::string TypeIdentifier() const override
		{
			return identifier;
		}

		PolicyAdapter(std::shared_ptr<const t_MDP> mdp, const DynaPlex::VarGroup& policy_vars, const int64_t mdp_int_hash)
			:mdp{ mdp },
			policy{ mdp,policy_vars },
			identifier{ policy_vars.Identifier() },
			mdp_int_hash{ mdp_int_hash },
			vars{ policy_vars }
		{

		}

		void SetAction(std::span<Trajectory> trajectories) const override
		{	
			for (Trajectory& traj: trajectories)
			{
				// Check that the states belong to this MDP
				if (traj.GetState()->mdp_int_hash != mdp_int_hash)
				{
					throw DynaPlex::Error("Error in Policy->SetAction: It seems you tried to call with states not"
						"associated with the MDP that this policy was obtained from. Please note that policies, even "
						"generic ones, can only act on states from the same mdp instance that the policy was obtained from.");
				}
				
				const DynaPlex::StateCategory& cat = traj.Category; 
				
				if (cat.IsAwaitAction())
				{
					//convert type-erased state to underlying type. 
					StateAdapter<t_State>* adapter = static_cast<StateAdapter<t_State>*>(traj.GetState().get());
					t_State& state = adapter->state;
					//dispatch to policy, depending on the signature of the GetAction implemented
					//on the policy
					if constexpr (HasGetAction<t_Policy, t_State>)
					{
						traj.NextAction = policy.GetAction(state);
					}
					else if constexpr (HasGetActionRNG<t_Policy, t_State, DynaPlex::RNG>)
					{
						RNG& rng = traj.RNGProvider.GetPolicyRNG();
						traj.NextAction = policy.GetAction(state, rng);
					}
					else
					{
						traj.NextAction = policy.GetActionState(state);
					}
				}
				else
				{
					throw DynaPlex::Error("Error in Policy->SetAction: Cannot set action when Trajectory.Category is not IsAwaitAction, i.e. when state is not IsAwaitAction.");
				}
			}

			
		}

		std::vector<int64_t> GetPromisingActions(const DynaPlex::dp_State& dp_state, int64_t num_actions) const override
		{
			// Check that the states belong to this MDP
			if (dp_state->mdp_int_hash != mdp_int_hash)
			{
				throw DynaPlex::Error("Error in Policy->GetPromisingActions: It seems you tried to call with states not"
					"associated with the MDP that this policy was obtained from. Please note that policies, even "
					"generic ones, can only act on states from the same mdp instance that the policy was obtained from.");
			}

			StateAdapter<t_State>* adapter = static_cast<StateAdapter<t_State>*>(dp_state.get());
			t_State& state = adapter->state;

			const DynaPlex::StateCategory& cat = mdp->GetStateCategory(state);
			if (cat.IsAwaitAction())
			{
				//dispatch to policy, depending on the signature of the GetPromisingActions implemented
				//on the policy
				if constexpr (HasGetPromisingActions<t_Policy, t_State>)
				{
					return policy.GetPromisingActions(state, num_actions);
				}
				//or return (a subset of) the allowed actions if GetPromisingActions not implemented.
				//Note: without this cap, SimulateOnlyPromisingActions is silently ignored for any
				//policy lacking its own GetPromisingActions (e.g. the default "random" policy used
				//to generate the first DCL generation), which is harmless for small action spaces but
				//becomes prohibitively expensive as the action space grows (cost scales with the
				//number of returned actions in downstream rollout comparisons).
				else
				{
					ActionRangeProvider<t_MDP> provider(mdp);
					auto actions = provider(state);
					int64_t count = actions.Count();
					std::vector<int64_t> vec;
					vec.reserve(actions.Count());
					for (int64_t action : actions)
					{
						vec.push_back(action);
					}
					if (num_actions > 0 && count > num_actions)
					{
						// If the policy can offer a concrete reference action (e.g. a heuristic
						// like CDI that has GetAction(state) but no GetPromisingActions), center
						// the candidate window on that action instead of spreading uniformly
						// across the whole range. A uniform grid ignores what the policy would
						// actually do - for a heuristic that (correctly) prescribes small order
						// quantities, an evenly-spaced grid over e.g. [0,270] with 12 points steps
						// in ~25-unit increments, so every rollout-selected "best" action gets
						// discretized to the nearest multiple of ~25 regardless of the heuristic's
						// real (often much smaller) recommendation - corrupting the training
						// labels generated whenever this policy drives sampling. Policies needing
						// an RNG (e.g. "random") have no meaningful single reference action, so
						// they keep the uniform-grid behavior below.
						if constexpr (HasGetAction<t_Policy, t_State>)
						{
							int64_t reference_action = policy.GetAction(state);
							auto it = std::lower_bound(vec.begin(), vec.end(), reference_action);
							int64_t ref_idx = std::min(static_cast<int64_t>(it - vec.begin()), count - 1);
							int64_t half = num_actions / 2;
							int64_t start_idx = ref_idx - half;
							if (start_idx < 0) start_idx = 0;
							if (start_idx + num_actions > count) start_idx = count - num_actions;
							std::vector<int64_t> windowed;
							windowed.reserve(num_actions);
							for (int64_t i = 0; i < num_actions; i++)
								windowed.push_back(vec[start_idx + i]);
							return windowed;
						}
						std::vector<int64_t> subsampled;
						subsampled.reserve(num_actions);
						for (int64_t i = 0; i < num_actions; i++)
						{
							int64_t idx = num_actions == 1 ? 0 : (i * (count - 1)) / (num_actions - 1);
							subsampled.push_back(vec[idx]);
						}
						return subsampled;
					}
					return vec;
				}
			}
			else
			{
				throw DynaPlex::Error("Error in Policy->GetPromisingActions: Cannot GetPromisingActions when state is not IsAwaitAction.");
			}
		}

	};
}

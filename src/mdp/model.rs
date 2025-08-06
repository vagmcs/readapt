use crate::mdp::policy::Policy;
use std::fmt::Debug;
use std::hash::Hash;
use thiserror::Error;

#[derive(Debug, PartialEq, Eq, Error)]
pub enum MDPError<'a, S: State> {
    #[error("The MDP cannot be empty")]
    Empty,
    #[error("No action available for state {}", state.id())]
    NoAction { state: &'a S },
    #[error("No transition is available for state {}", state.id())]
    NoTransition { state: &'a S },
    #[error("The transition matrix is invalid. Either the dimensions are incorrect or the probabilities do not sum to 1")]
    InvalidTransitionMatrix,
    #[error("The reward matrix has invalid dimensions")]
    InvalidRewardMatrix,
}

/// Represents an episode of an MDP. Each such episode has a starting state, a trajectory
/// of states that the agent navigated in the MDP horizon and a total reward.
#[derive(Debug)]
pub struct Episode<'a, S: State> {
    pub starting_state: &'a S,
    pub trajectory: Vec<&'a S>,
    pub total_reward: f64,
}

/// Represents a state in the MDP. Each state should have a unique index or ID,
/// always starting from 0, up to the number of states. However, the user of the trait
/// is responsible to ensure that state indices are unique across the MDP states.
///
/// # Examples
///
/// The simplest state implementation only stores the index:
///
/// ```
/// use readapt::mdp::model::State;
///
/// #[derive(Debug, Hash, PartialEq, Eq)]
/// struct S { id: usize }
///
/// impl State for S {
///     fn id(&self) -> usize { self.id }
/// }
///
/// let states: Vec<S> = (0..5).map(|id| S { id }).collect();
///
/// assert_eq!(states.len(), 5);
/// assert_eq!(states.iter().map(|state| state.id()).collect::<Vec<_>>(), vec![0, 1, 2, 3, 4]);
/// ```
pub trait State: Debug + Hash + Eq {
    fn id(&self) -> usize;
}

/// Represents an action in the MDP. Each action should have a unique index or ID,
/// always starting from 0, up to the number of actions. However, the user of the trait
/// is responsible to ensure that action indices are unique across the MDP states.
pub trait Action: Debug + Eq {
    fn id(&self) -> usize;
}

/// Represents a Markov Decision Process (MDP) model for decision-making. The idea is
/// that an agent situated in a stochastic environment which changes in discrete "timesteps".
/// The agent can influence the way the environment changes via "actions". For each action the
/// agent can perform, the environment will transition from a state "s" to a state "s'" following
/// a certain transition function. The transition function specifies, for each triplet in SxAxS'
/// the probability that such a transition will happen.
pub trait MDP<S: State, A: Action> {
    /// Returns the number of states.
    fn n_states(&self) -> usize;

    /// Returns all states.
    fn states(&self) -> &[S];

    /// Returns the number of actions.
    fn n_actions(&self) -> usize;

    /// Returns all actions.
    fn actions(&self) -> &[A];

    /// Returns true if the given state is a terminal.
    fn is_terminal(&self, state: &S) -> bool;

    /// Discount factor determines the value of future rewards. By default
    /// this function always returns 1, which accounts for no discount.
    #[inline(always)]
    fn discount_factor(&self) -> f64 {
        1.0
    }

    /// Returns the transition probability of the triplet (state, action, state).
    fn transition_probability(&self, state: &S, action: &A, next_state: &S) -> f64;

    /// Returns the reward for the triplet (state, action, state).
    fn reward(&self, state: &S, action: &A, next_state: &S) -> f64;

    /// Acts on the given state using the given action and returns the next state.
    fn act(&self, state: &S, action: &A) -> &S;

    /// Executes a given policy on the MDP and returns an episode.
    ///
    /// # Arguments
    ///
    /// - `policy` - the policy to be executed.
    /// - `starting_state` - the init state of the MDP, that is, the state that the agent starts.
    /// - `maximum_steps` - the maximum iterations for the execution. If no terminal state is achieved the execution terminates.
    fn run_policy<'a>(
        &'a self,
        policy: &'a Policy<S, A>,
        starting_state: &'a S,
        maximum_steps: usize,
    ) -> Result<Episode<'a, S>, MDPError<'a, S>> {
        let mut total_reward = 0f64;
        let mut trajectory = vec![starting_state];
        let mut state = starting_state;

        for _ in 0..maximum_steps {
            // select policy action and place the agent to the next state
            match policy.select_action(state) {
                Some(action) => {
                    let next_state = self.act(state, action);
                    trajectory.push(next_state);
                    total_reward += self.reward(state, action, next_state);
                    state = next_state;
                }
                None => {
                    return Err(MDPError::NoAction { state });
                }
            };

            // if current state is terminal (after the action) then break
            if self.is_terminal(state) {
                break;
            }
        }

        Ok(Episode {
            starting_state,
            trajectory,
            total_reward,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::mdp::model::{Action, State, MDP};
    use crate::mdp::policy::Policy;
    use rand::Rng;

    #[derive(Debug, Hash, PartialEq, Eq)]
    struct S {
        id: usize,
    }

    impl State for S {
        fn id(&self) -> usize {
            self.id
        }
    }

    #[derive(Debug, PartialEq, Eq)]
    enum A {
        Forward,
        Backward,
    }

    impl Action for A {
        fn id(&self) -> usize {
            0
        }
    }

    struct Line {
        states: Vec<S>,
        actions: Vec<A>,
    }

    impl MDP<S, A> for Line {
        fn n_states(&self) -> usize {
            self.states.len()
        }

        fn n_actions(&self) -> usize {
            self.actions.len()
        }

        fn states(&self) -> &[S] {
            &self.states
        }

        fn actions(&self) -> &[A] {
            &self.actions
        }

        fn is_terminal(&self, state: &S) -> bool {
            state.id() == self.n_states() - 1
        }

        fn act(&self, state: &S, _: &A) -> &S {
            if rand::thread_rng().gen_bool(0.5) {
                // forward
                if state.id() != self.n_states() - 1 {
                    &self.states[state.id() + 1]
                } else {
                    &self.states[state.id()]
                }
            } else {
                // backward
                if state.id() != 0 {
                    &self.states[state.id() - 1]
                } else {
                    &self.states[0]
                }
            }
        }

        fn transition_probability(&self, state: &S, action: &A, next_state: &S) -> f64 {
            match action {
                A::Forward if state.id() == next_state.id() - 1 => 0.5,
                A::Backward if state.id() == next_state.id() + 1 => 0.5,
                _ => 0.0,
            }
        }

        #[rustfmt::skip]
        fn reward(&self, _: &S, _: &A, next_state: &S) -> f64 {
            if next_state.id() != self.n_states() - 1 { -1.0 } else { 0.0 }
        }
    }

    #[test]
    fn run_incomplete_policy() {
        let env = Line {
            states: (0..2).map(|id| S { id }).collect(),
            actions: vec![A::Forward, A::Backward],
        };

        // creates a policy having no action for the starting state
        let incomplete_policy = Policy::new(HashMap::from([(&env.states[1], &A::Forward)]));
        let episode = env.run_policy(&incomplete_policy, &env.states[0], 10);

        assert!(episode.is_err());
        assert!(episode
            .unwrap_err()
            .to_string()
            .contains("No action available for state 0."));
    }

    #[test]
    fn run_random_policy() {
        let env = Line {
            states: (0..10).map(|id| S { id }).collect(),
            actions: vec![A::Forward, A::Backward],
        };

        // starting state should always exist
        let starting_state = env.states.iter().find(|state| state.id() == 0);
        assert!(starting_state.is_some());

        // create a random policy (random assignment of states to actions) and run the policy on the environment
        let policy = Policy::random(&env.states, &env.actions);
        let episode = env
            .run_policy(&policy, starting_state.unwrap(), 100)
            .unwrap();

        // starting state is always zero
        assert_eq!(episode.starting_state.id(), 0);

        // consecutive states in the trajectory should have contiguous IDs
        for i in 0..episode.trajectory.len() - 1 {
            assert!(
                episode.trajectory[i]
                    .id()
                    .abs_diff(episode.trajectory[i + 1].id())
                    <= 1
            );
        }

        // the total reward should be the number of states traversed except the final state
        let actual_reward: f64 = episode.trajectory[1..] // skip the starting state
            .iter()
            .filter(|&&state| !env.is_terminal(state))
            .map(|_| -1f64)
            .sum();

        assert_eq!(episode.total_reward, actual_reward);
    }
}

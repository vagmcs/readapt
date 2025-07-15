use crate::mdp::policy::Policy;
use rand::Rng;
use std::fmt;
use std::hash::Hash;

#[derive(Debug)]
pub enum MDPError<'a, S: State> {
    NoAction { state: &'a S },
    NoTransition { state: &'a S },
}

impl<'a, S: State> fmt::Display for MDPError<'a, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MDPError::NoAction { state } => {
                write!(f, "No action available for state {}.", state.id())
            }
            MDPError::NoTransition { state } => {
                write!(f, "No transition is available for state {}.", state.id())
            }
        }
    }
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
/// #[derive(Hash, PartialEq, Eq)]
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
pub trait State: Hash + Eq {
    fn id(&self) -> usize;
}

/// Represents an action in the MDP. Each action should have a unique index or ID,
/// always starting from 0, up to the number of actions. However, the user of the trait
/// is responsible to ensure that action indices are unique across the MDP states.
pub trait Action {
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
    /// Returns the number of actions.
    fn n_actions(&self) -> usize;
    /// Returns true if the given state is a terminal.
    fn is_terminal(&self, state: &S) -> bool;
    /// Discount factor determines the value of future rewards. By default
    /// this function always returns 1, which accounts for no discount.
    #[rustfmt::skip]
    #[inline(always)]
    fn discount_factor(&self) -> f64 { 1.0 }
    /// Returns a vector of possible states
    fn next_states<'a>(&'a self, state: &'a S, action: &'a A) -> Vec<&'a S>;
    /// Returns the transition probability of the triplet (state, action, state).
    fn transition_probability(&self, state: &S, action: &A, next_state: &S) -> f64;
    /// Returns the reward for the triplet (state, action, state).
    fn reward(&self, state: &S, action: &A, next_state: &S) -> f64;
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
        let mut trajectory = Vec::new();
        let mut current_state = starting_state;

        for _ in 0..maximum_steps {
            // select policy action and place the agent to the next state
            match policy.select_action(current_state) {
                Some(action) => {
                    let possible_states = self.next_states(current_state, action);
                    if possible_states.is_empty() {
                        return Err(MDPError::NoTransition {
                            state: current_state,
                        });
                    } else {
                        for next_state in possible_states {
                            let prob =
                                self.transition_probability(current_state, action, next_state);
                            if prob > rand::thread_rng().gen() {
                                trajectory.push(next_state);
                                current_state = next_state;
                                total_reward += self.reward(current_state, action, next_state);
                            } else {
                                trajectory.push(current_state);
                                total_reward += self.reward(current_state, action, current_state);
                            }
                        }
                    }
                }
                None => {
                    // there is no action, therefore remain in the current state
                    trajectory.push(current_state);
                }
            };

            // if current state is terminal (after the action) then break
            if self.is_terminal(current_state) {
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
    use crate::mdp::model::{Action, State, MDP};
    use crate::mdp::policy::Policy;

    #[derive(Debug, Hash, PartialEq, Eq)]
    struct S {
        id: usize,
    }

    impl State for S {
        fn id(&self) -> usize {
            self.id
        }
    }

    enum A {
        Forward,
        Backward,
    }

    impl Action for A {
        fn id(&self) -> usize {
            0
        }
    }

    struct Grid {
        states: Vec<S>,
        actions: Vec<A>,
    }

    impl MDP<S, A> for Grid {
        fn n_states(&self) -> usize {
            self.states.len()
        }

        fn n_actions(&self) -> usize {
            self.actions.len()
        }

        fn is_terminal(&self, state: &S) -> bool {
            state.id() == self.n_states() - 1
        }

        fn next_states(&self, state: &S, _action: &A) -> Vec<&S> {
            self.states
                .iter()
                .filter(|s| {
                    (state.id() == 0 && s.id() == 1)
                        || (state.id() > 0 && s.id() == state.id() - 1)
                        || (state.id() == self.n_states() - 1 && s.id() == state.id() - 1)
                        || (state.id() < self.n_states() - 1 && s.id() == state.id() + 1)
                })
                .collect()
        }

        fn transition_probability(&self, _state: &S, _action: &A, _next_state: &S) -> f64 {
            1f64 / self.n_actions() as f64 // every action has equal probability to succeed
        }

        fn reward(&self, _state: &S, _action: &A, next_state: &S) -> f64 {
            -((next_state.id() != self.n_states() - 1) as i8) as f64
        }
    }

    #[test]
    fn run_random_policy() {
        let env = Grid {
            states: (0..10).map(|id| S { id }).collect(),
            actions: vec![A::Forward, A::Backward],
        };

        // starting state should always exist
        let starting_state = env.states.iter().find(|state| state.id() == 0);
        assert!(starting_state.is_some());

        // create a random policy (random assignment of states to actions) and run the policy on the environment
        let policy = Policy::random(&env.states, &env.actions);
        let episode = env
            .run_policy(&policy, starting_state.unwrap(), 1000)
            .unwrap();

        // starting state is always zero
        assert_eq!(episode.starting_state.id(), 0);

        // the total negative reward should be the number of states traversed except the final state
        let actual_reward = episode
            .trajectory
            .iter()
            .filter(|&&state| !env.is_terminal(state))
            .map(|_| -1f64)
            .sum();

        assert_eq!(episode.total_reward, actual_reward);
    }
}

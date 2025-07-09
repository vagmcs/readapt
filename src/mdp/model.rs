use std::fmt;

use rand::Rng;

use crate::mdp::policy::Policy;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MDPError {
    NoAction { state: usize },
    NoTransition { state: usize },
}

impl fmt::Display for MDPError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MDPError::NoAction { state } => write!(f, "No action available for state {state}."),
            MDPError::NoTransition { state } => {
                write!(f, "No transition is available for state {state}.")
            }
        }
    }
}

/// Episode
#[derive(Debug)]
pub struct Episode {
    pub starting_state: usize,
    pub trajectory: Vec<usize>,
    pub total_reward: f64,
}

/// Represents a Markov Decision Process (MDP) model for decision-making. The idea is
/// that an agent situated in a stochastic environment which changes in discrete "timesteps".
/// The agent can influence the way the environment changes via "actions". For each action the
/// agent can perform, the environment will transition from a state "s" to a state "s'" following
/// a certain transition function. The transition function specifies, for each triplet in SxAxS'
/// the probability that such a transition will happen.
pub trait MDP {
    /// Returns the number of states.
    fn n_states(&self) -> usize;
    /// Returns the number of actions.
    fn n_actions(&self) -> usize;
    /// Returns true if the given state is a terminal.
    fn is_terminal(&self, state: usize) -> bool;
    /// Discount factor determines the value of future rewards.
    fn discount_factor(&self) -> f64;
    /// Returns a vector of possible states
    fn next_states(&self, state: usize, action: usize) -> Vec<usize>;
    /// Returns the transition probability of the triplet (state, action, state).
    fn transition_probability(&self, state: usize, action: usize, next_state: usize) -> f64;
    /// Returns the reward for the triplet (state, action, state).
    fn reward(&self, state: usize, action: usize, next_state: usize) -> f64;
    /// Executes a given policy on the MDP and returns an episode.
    ///
    /// # Arguments
    ///
    /// - `policy` - The policy to be executed.
    /// - `starting_state` - the init state of the MDP, that is, the state the agent starts.
    /// - `maximum_steps` - the maximum iterations for the execution. If no terminal state is achieved the execution terminates.
    fn run_policy(
        &self,
        policy: &Policy,
        starting_state: usize,
        maximum_steps: usize,
    ) -> Result<Episode, MDPError> {
        let mut total_reward = 0f64;
        let mut trajectory = Vec::new();
        let mut current_state = starting_state;

        for _ in 0..maximum_steps {
            match policy.select_action(current_state) {
                Some(action) => {
                    let possible_states = self.next_states(current_state, *action);
                    if possible_states.is_empty() {
                        return Err(MDPError::NoTransition {
                            state: current_state,
                        });
                    } else {
                        for next_state in possible_states {
                            let prob =
                                self.transition_probability(current_state, *action, next_state);
                            if prob > rand::thread_rng().gen() {
                                trajectory.push(next_state);
                                current_state = next_state;
                                total_reward += self.reward(current_state, *action, next_state);
                            } else {
                                trajectory.push(current_state);
                                total_reward += self.reward(current_state, *action, current_state);
                            }
                        }
                    }
                }
                None => {
                    // there is no action, therefore remain in the current state
                    trajectory.push(current_state);
                }
            };

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
    use crate::mdp::{model::MDP, policy::Policy};

    struct Env;

    impl MDP for Env {
        fn n_states(&self) -> usize {
            4
        }

        fn n_actions(&self) -> usize {
            4
        }

        fn is_terminal(&self, state: usize) -> bool {
            state == self.n_states() - 1
        }

        fn discount_factor(&self) -> f64 {
            1.0
        }

        fn next_states(&self, _state: usize, _action: usize) -> Vec<usize> {
            (0..self.n_states()).collect()
        }

        fn transition_probability(&self, _state: usize, _action: usize, _next_state: usize) -> f64 {
            1f64 / self.n_actions() as f64 // every action has equal probability
        }

        fn reward(&self, _state: usize, _action: usize, next_state: usize) -> f64 {
            -((next_state != self.n_states() - 1) as i8) as f64
        }
    }

    #[test]
    fn run_random_policy() {
        let env = Env;
        let policy = Policy::random(env.n_states(), env.n_actions());
        let episode = env.run_policy(&policy, 0, 1000).unwrap();

        // starting state is always zero
        assert_eq!(episode.starting_state, 0);

        // the total negative reward should be the number of states traversed except the final state
        let actual_reward = -(episode
            .trajectory
            .iter()
            .filter(|&&state| !env.is_terminal(state))
            .count() as f64);

        assert_eq!(episode.total_reward, actual_reward);
    }
}

use crate::mdp::model::{Action, MDPError, State, MDP};
use crate::mdp::policy::Policy;
use rand::Rng;
use std::collections::HashMap;

/// Represents any algorithm that searches for an optimal policy given a Markov Decision Process.
pub trait Optimizer<'a, S: State, A: Action, M: MDP<S, A>> {
    /// Returns an optimal policy for the provided MDP.
    ///
    /// # Arguments
    ///
    /// - `mdp` - Markov Decision Process.
    fn find_optimal_policy(&self, mdp: &'a M) -> Result<Policy<'a, S, A>, MDPError<'a, S>>;
}

pub struct PolicyIteration {
    /// Small positive number determining the accuracy of estimation.
    pub theta: f64,
    // Maximum iterations for policy evaluation.
    pub max_iterations: usize,
}

impl<'a, S: State, A: Action, M: MDP<S, A>> Optimizer<'a, S, A, M> for PolicyIteration {
    fn find_optimal_policy(&self, mdp: &'a M) -> Result<Policy<'a, S, A>, MDPError<'a, S>> {
        let mut delta;
        let mut values = vec![0.0; mdp.n_states()];

        // start from a random policy
        let mut rng = rand::thread_rng();
        let mut mapping: HashMap<&S, &A> = mdp
            .states()
            .iter()
            .map(|state| (state, &mdp.actions()[rng.gen_range(0..mdp.n_actions())]))
            .collect();

        loop {
            // policy evaluation
            for _ in 0..self.max_iterations {
                delta = 0f64;
                for state in mdp.states() {
                    let value = values[state.id()];
                    match mapping.get(state) {
                        Some(&action) => {
                            let new_value = mdp.states().iter().fold(0.0, |v, next_state| {
                                let r = mdp.reward(state, action, next_state);
                                let p = mdp.transition_probability(state, action, next_state);
                                v + p * (r + mdp.discount_factor() * values[next_state.id()])
                            });

                            delta = delta.max((value - new_value).abs());
                            values[state.id()] = new_value;
                        }
                        None => return Err(MDPError::NoAction { state }),
                    }
                }
                if delta < self.theta {
                    break;
                }
            }

            // policy improvement
            let mut stable = true;
            for state in mdp.states() {
                match mapping.get(state) {
                    Some(&prev_action) => {
                        let mut best_action = prev_action;
                        let mut best_value = f64::NEG_INFINITY;

                        for action in mdp.actions() {
                            let v = mdp.states().iter().fold(0.0, |v, s| {
                                let r = mdp.reward(state, action, s);
                                let p = mdp.transition_probability(state, action, s);
                                v + p * (r + mdp.discount_factor() * values[s.id()])
                            });

                            if v > best_value {
                                best_value = v;
                                best_action = action;
                            }
                        }

                        stable &= best_action == prev_action;
                        mapping.insert(state, best_action);
                    }
                    None => return Err(MDPError::NoAction { state }),
                }
            }

            if stable {
                return Ok(Policy::new(mapping));
            }
        }
    }
}

pub struct ValueIteration {
    /// Small positive number determining the accuracy of estimation.
    pub theta: f64,
    // Maximum iterations for policy evaluation.
    pub max_iterations: usize,
}

impl<'a, S: State, A: Action, M: MDP<S, A>> Optimizer<'a, S, A, M> for ValueIteration {
    fn find_optimal_policy(&self, mdp: &'a M) -> Result<Policy<'a, S, A>, MDPError<'a, S>> {
        let mut delta;
        let mut values = vec![0.0; mdp.n_states()];

        // policy evaluation
        for _ in 0..self.max_iterations {
            delta = 0f64;
            for state in mdp.states() {
                let value = values[state.id()];

                values[state.id()] =
                    mdp.actions()
                        .iter()
                        .fold(f64::NEG_INFINITY, |max_v, action| {
                            let x = mdp.states().iter().fold(0.0, |v, next_state| {
                                let r = mdp.reward(state, action, next_state);
                                let p = mdp.transition_probability(state, action, next_state);
                                v + p * (r + mdp.discount_factor() * values[next_state.id()])
                            });

                            max_v.max(x)
                        });

                delta = delta.max((value - values[state.id()]).abs());
            }
            if delta < self.theta {
                break;
            }
        }

        // output a policy
        let mut mapping = HashMap::with_capacity(mdp.n_states());
        for state in mdp.states() {
            let mut best_action = &mdp.actions()[0];
            let mut best_value = f64::NEG_INFINITY;

            // find best action
            for action in mdp.actions() {
                let v = mdp.states().iter().fold(0.0, |v, s| {
                    let r = mdp.reward(state, action, s);
                    let p = mdp.transition_probability(state, action, s);
                    v + p * (r + mdp.discount_factor() * values[s.id()])
                });

                if v > best_value {
                    best_value = v;
                    best_action = action;
                }
            }

            mapping.insert(state, best_action);
        }

        Ok(Policy::new(mapping))
    }
}

#[cfg(test)]
mod tests {

    use crate::mdp::environment::{GridWorld, Move};
    use crate::mdp::model::{State, MDP};
    use crate::mdp::optimizer::{Optimizer, PolicyIteration, ValueIteration};

    #[test]
    fn test_policy_iteration() {
        let grid = GridWorld::from(
            3,
            4,
            |s| s.id() == 5, // wall
            |a| match a {
                Move::North => |d| match d {
                    Move::North => 0.8,
                    Move::South => 0.0,
                    Move::East => 0.1,
                    Move::West => 0.1,
                },
                Move::South => |d| match d {
                    Move::North => 0.0,
                    Move::South => 0.8,
                    Move::East => 0.1,
                    Move::West => 0.1,
                },
                Move::East => |d| match d {
                    Move::North => 0.1,
                    Move::South => 0.1,
                    Move::East => 0.8,
                    Move::West => 0.0,
                },
                Move::West => |d| match d {
                    Move::North => 0.1,
                    Move::South => 0.1,
                    Move::East => 0.0,
                    Move::West => 0.8,
                },
            },
            |s| {
                // states 3 and 7 are terminal, while every other state has a small negative reward
                if s.id() == 3 {
                    1.0
                } else if s.id() == 7 {
                    -1.0
                } else {
                    -0.5
                }
            },
            |s| s.id() == 3 || s.id() == 7, // terminal states
        )
        .unwrap();

        let optimal_policy = PolicyIteration {
            theta: 1e-6,
            max_iterations: 100000,
        }
        .find_optimal_policy(&grid)
        .unwrap();

        assert_eq!(
            optimal_policy.select_action(&grid.states()[0]),
            Some(&Move::East)
        );
        assert_eq!(
            optimal_policy.select_action(&grid.states()[1]),
            Some(&Move::East)
        );
        assert_eq!(
            optimal_policy.select_action(&grid.states()[2]),
            Some(&Move::East)
        );
        assert_eq!(
            optimal_policy.select_action(&grid.states()[4]),
            Some(&Move::North)
        );
        assert_eq!(
            optimal_policy.select_action(&grid.states()[6]),
            Some(&Move::North)
        );
        assert_eq!(
            optimal_policy.select_action(&grid.states()[8]),
            Some(&Move::North)
        );
        assert_eq!(
            optimal_policy.select_action(&grid.states()[9]),
            Some(&Move::East)
        );
        assert_eq!(
            optimal_policy.select_action(&grid.states()[10]),
            Some(&Move::North)
        );
        assert_eq!(
            optimal_policy.select_action(&grid.states()[11]),
            Some(&Move::North)
        );
    }

    #[test]
    fn test_value_iteration() {
        let grid = GridWorld::from(
            3,
            4,
            |s| s.id() == 5, // wall
            |a| match a {
                Move::North => |d| match d {
                    Move::North => 0.8,
                    Move::South => 0.0,
                    Move::East => 0.1,
                    Move::West => 0.1,
                },
                Move::South => |d| match d {
                    Move::North => 0.0,
                    Move::South => 0.8,
                    Move::East => 0.1,
                    Move::West => 0.1,
                },
                Move::East => |d| match d {
                    Move::North => 0.1,
                    Move::South => 0.1,
                    Move::East => 0.8,
                    Move::West => 0.0,
                },
                Move::West => |d| match d {
                    Move::North => 0.1,
                    Move::South => 0.1,
                    Move::East => 0.0,
                    Move::West => 0.8,
                },
            },
            |s| {
                // states 3 and 7 are terminal, while every other state has a small negative reward
                if s.id() == 3 {
                    1.0
                } else if s.id() == 7 {
                    -1.0
                } else {
                    -0.5
                }
            },
            |s| s.id() == 3 || s.id() == 7, // terminal states
        )
        .unwrap();

        let optimal_policy = ValueIteration {
            theta: 1e-6,
            max_iterations: 100000,
        }
        .find_optimal_policy(&grid)
        .unwrap();

        assert_eq!(
            optimal_policy.select_action(&grid.states()[0]),
            Some(&Move::East)
        );
        assert_eq!(
            optimal_policy.select_action(&grid.states()[1]),
            Some(&Move::East)
        );
        assert_eq!(
            optimal_policy.select_action(&grid.states()[2]),
            Some(&Move::East)
        );
        assert_eq!(
            optimal_policy.select_action(&grid.states()[4]),
            Some(&Move::North)
        );
        assert_eq!(
            optimal_policy.select_action(&grid.states()[6]),
            Some(&Move::North)
        );
        assert_eq!(
            optimal_policy.select_action(&grid.states()[8]),
            Some(&Move::North)
        );
        assert_eq!(
            optimal_policy.select_action(&grid.states()[9]),
            Some(&Move::East)
        );
        assert_eq!(
            optimal_policy.select_action(&grid.states()[10]),
            Some(&Move::North)
        );
        assert_eq!(
            optimal_policy.select_action(&grid.states()[11]),
            Some(&Move::North)
        );
    }
}

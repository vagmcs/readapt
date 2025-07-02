use rand::distributions::{Distribution, Uniform};
use std::collections::HashMap;

/// Represents a policy in a Markov Decision Process (MDP), which defines a mapping
/// from each state to an action. The `Policy` encapsulates a strategy for decision-making
/// by associating states to actions. The policy is considered optimal if for every state,
/// the chosen action is the best possible according to some criterion.
///
/// # Examples
///
/// Create a custom policy that maps state 1 to action 10.
///
/// ```
/// use std::collections::HashMap;
/// use readapt::mdp::policy::Policy;
///
/// let policy = Policy::new(HashMap::from([(1, 10)]));
///
/// if let Some(action) = policy.select_action(1) {
///     assert_eq!(*action, 10);
/// }
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Policy {
    mapping: HashMap<usize, usize>,
}

impl Policy {
    /// Creates a custom policy.
    ///
    /// # Arguments
    ///
    /// - `mapping` - a hash map from states to actions
    pub fn new(mapping: HashMap<usize, usize>) -> Self {
        Self { mapping }
    }

    /// Creates a uniform at random policy.
    ///
    /// # Arguments
    ///
    /// - `n_states` - number of states
    /// - `n_actions` - number of actions
    pub fn random(n_states: usize, n_actions: usize) -> Self {
        Self {
            mapping: HashMap::from_iter(
                Uniform::new(0, n_actions) // sample a random action
                    .sample_iter(&mut rand::thread_rng())
                    .take(n_states)
                    .enumerate(),
            ),
        }
    }

    /// Returns the corresponding policy action for the given state, or None if
    /// there is no action assigned to the given state.
    ///
    /// # Arguments
    ///
    /// - `state` - the state of interest
    pub fn select_action(&self, state: usize) -> Option<&usize> {
        self.mapping.get(&state)
    }
}

#[cfg(test)]
mod tests {
    use crate::mdp::policy::Policy;

    #[test]
    fn random_policy() {
        let random_policy = Policy::random(5, 4);

        // there should be 5 states in the policy
        assert_eq!(random_policy.mapping.len(), 5);
        // there should be an action for each state
        assert!((0..4).all(|state| random_policy.select_action(state).is_some()));
    }
}

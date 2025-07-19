use crate::mdp::model::{Action, State};
use rand::seq::SliceRandom;
use std::collections::HashMap;

/// Represents a policy in a Markov Decision Process (MDP), which defines a mapping
/// from each state to an action. The `Policy` encapsulates a strategy for decision-making
/// by associating states to actions. The policy is considered optimal if for every state,
/// the chosen action is the best possible according to some criterion.
///
/// # Examples
///
/// Create a custom policy that maps a state to an action.
///
/// ```
/// use std::collections::HashMap;
/// use readapt::mdp::policy::Policy;
/// use readapt::mdp::model::{State, Action};
///
/// #[derive(Debug, Hash, PartialEq, Eq)]
/// struct S { id: usize }
///
/// #[derive(Debug, PartialEq)]
/// struct A { id: usize }
///
/// impl State for S {
///     fn id(&self) -> usize { self.id }
/// }
///
/// impl Action for A {
///     fn id(&self) -> usize { self.id }
/// }
///
/// let state = S { id: 1 };
/// let action = A { id: 10 };
/// let policy = Policy::new(HashMap::from([(&state, &action)]));
///
/// if let Some(selected_action) = policy.select_action(&state) {
///     assert_eq!(*selected_action, action);
/// }
/// ```
#[derive(Debug, PartialEq, Eq)]
pub struct Policy<'a, S: State, A: Action> {
    mapping: HashMap<&'a S, &'a A>,
}

impl<'a, S: State, A: Action> Policy<'a, S, A> {
    /// Creates a custom policy.
    ///
    /// # Arguments
    ///
    /// - `mapping` - a hash map from states to actions
    pub fn new(mapping: HashMap<&'a S, &'a A>) -> Self {
        Self { mapping }
    }

    /// Creates a uniform at random policy.
    ///
    /// # Arguments
    ///
    /// - `states` - an iterator over states
    /// - `actions` - an iterator over actions
    pub fn random(states: &'a [S], actions: &'a [A]) -> Self {
        let mut rng = rand::thread_rng();
        let mapping = states
            .iter()
            .map(|state| {
                let action = actions.choose(&mut rng).expect("Actions must not be empty");
                (state, action)
            })
            .collect();
        Self { mapping }
    }

    /// Returns the corresponding policy action for the given state, or None if
    /// there is no action assigned to the given state.
    ///
    /// # Arguments
    ///
    /// - `state` - the state of interest
    pub fn select_action(&self, state: &S) -> Option<&A> {
        self.mapping.get(state).copied()
    }
}

#[cfg(test)]
mod tests {
    use crate::mdp::{
        model::{Action, State},
        policy::Policy,
    };

    #[derive(Debug, Hash, PartialEq, Eq)]
    struct S {
        id: usize,
    }

    impl State for S {
        fn id(&self) -> usize {
            self.id
        }
    }

    struct A {
        id: usize,
    }

    impl Action for A {
        fn id(&self) -> usize {
            self.id
        }
    }

    #[test]
    fn random_policy() {
        let states: Vec<S> = (0..5).map(|id| S { id }).collect();
        let actions: Vec<A> = (0..4).map(|id| A { id }).collect();
        let random_policy = Policy::random(&states, &actions);

        // there should be 5 states in the policy
        assert_eq!(random_policy.mapping.len(), 5);

        // there should be an action for each state
        assert!(states
            .iter()
            .all(|state| random_policy.select_action(state).is_some()));

        // there should be no action for state 10
        assert!(random_policy.select_action(&S { id: 10 }).is_none());
    }
}

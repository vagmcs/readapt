use crate::mdp::model::{Action, MDPError, State, MDP};
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use std::collections::HashSet;
use std::fmt;
use std::hash::Hash;

/// Represents a movement action on the grid world environment.
/// There are four possible actions, moving north or up, south or down,
/// east or left, and west or right on the 2-dimensional grid.
#[derive(Debug, PartialEq, Eq)]
pub enum Move {
    North,
    South,
    East,
    West,
}

impl Move {
    /// An array of all possible movement actions.
    pub const ACTIONS: [Move; 4] = [Move::North, Move::South, Move::East, Move::West];

    #[inline(always)]
    fn len() -> usize {
        Move::ACTIONS.len()
    }
}

impl Action for Move {
    fn id(&self) -> usize {
        match self {
            Move::North => 0,
            Move::South => 1,
            Move::East => 2,
            Move::West => 3,
        }
    }
}

/// Represents a tile on the grid.
#[derive(Debug, Eq)]
pub struct Tile {
    id: usize,
    pub x: usize,
    pub y: usize,
}

impl State for Tile {
    fn id(&self) -> usize {
        self.id
    }
}

impl Hash for Tile {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl PartialEq for Tile {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

/// Represents a grid-based Markov Decision Process (MDP).
///
/// `GridWorld` is commonly used in reinforcement learning to model an agent navigating
/// a 2-dimensional grid of stochastic transitions, collecting rewards.
pub struct GridWorld {
    rows: usize,
    columns: usize,
    states: Vec<Tile>,
    transition_probabilities: Vec<Vec<Vec<f64>>>,
    rewards: Vec<Vec<Vec<f64>>>,
    terminal_states: HashSet<usize>,
}

impl fmt::Display for GridWorld {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for row in 0..self.rows {
            // Top horizontal border
            for _ in 0..self.columns {
                write!(f, "+-------------")?;
            }
            writeln!(f, "+")?;

            // First line per cell: N and S rewards
            for col in 0..self.columns {
                let idx = row * self.columns + col;
                write!(f, "|")?;

                if self.terminal_states.contains(&idx) {
                    write!(f, "     =T=     ")?;
                } else {
                    let n = if row > 0 {
                        let n_idx = (row - 1) * self.columns + col;
                        format!("{:.1}", self.rewards[idx][Move::North.id()][n_idx])
                    } else {
                        format!("{:.1}", self.rewards[idx][Move::North.id()][idx])
                    };

                    let s = if row + 1 < self.rows {
                        let s_idx = (row + 1) * self.columns + col;
                        format!("{:.1}", self.rewards[idx][Move::South.id()][s_idx])
                    } else {
                        format!("{:.1}", self.rewards[idx][Move::South.id()][idx])
                    };

                    // Create the content with compact format to fit in 12 characters
                    let content = format!("N:{n} S:{s}");
                    write!(f, "{content:12}")?;
                }
            }
            writeln!(f, "|")?;

            // Second line per cell: E and W rewards
            for col in 0..self.columns {
                let idx = row * self.columns + col;
                write!(f, "|")?;

                if self.terminal_states.contains(&idx) {
                    write!(f, "             ")?; // Terminal content: empty second line
                } else {
                    let e = if col + 1 < self.columns {
                        let e_idx = row * self.columns + (col + 1);
                        format!("{:.1}", self.rewards[idx][Move::East.id()][e_idx])
                    } else {
                        format!("{:.1}", self.rewards[idx][Move::East.id()][idx])
                    };

                    let w = if col > 0 {
                        let w_idx = row * self.columns + (col - 1);
                        format!("{:.1}", self.rewards[idx][Move::West.id()][w_idx])
                    } else {
                        format!("{:.1}", self.rewards[idx][Move::West.id()][idx])
                    };

                    // Create the content with compact format to fit in 12 characters
                    let content = format!("E:{e} W:{w}");
                    write!(f, "{content:12}")?;
                }
            }
            writeln!(f, "|")?;
        }

        // Bottom border
        for _ in 0..self.columns {
            write!(f, "+-------------")?;
        }
        writeln!(f, "+")
    }
}

impl GridWorld {
    /// Creates a custom Grid World.
    ///
    /// # Notes
    ///
    /// 1. The grid cannot be empty.
    /// 2. The transition and reward matrices must have dimensions SxAxS, where S is the number of states and A the number of actions.
    /// 3. The transition probabilities for each action must sum to 1.
    ///
    /// # Arguments
    ///
    /// - `rows` - number of rows
    /// - `columns` - number of columns
    /// - `transition_probabilities` - a matrix of dimension SxAxS holding the movement probabilities for each triplet (s, a, s')
    /// - `rewards` - a matrix of dimension SxAxS holding the rewards for each triplet (s, a, s')
    /// - `is_terminal_state` - a function returning true when any given tile state is terminal
    pub fn new<'a>(
        rows: usize,
        columns: usize,
        transition_probabilities: Vec<Vec<Vec<f64>>>,
        rewards: Vec<Vec<Vec<f64>>>,
        is_terminal_state: fn(&Tile) -> bool,
    ) -> Result<Self, MDPError<'a, Tile>> {
        let n_states = rows * columns;
        let mut states = Vec::with_capacity(n_states);
        let mut terminal_states = HashSet::new();

        if rows == 0 || columns == 0 {
            return Err(MDPError::Empty);
        }
        if transition_probabilities.len() != n_states
            || transition_probabilities[0].len() != Move::len()
            || transition_probabilities[0][0].len() != n_states
        {
            return Err(MDPError::InvalidTransitionMatrix);
        }
        for t in transition_probabilities.iter() {
            for a in t.iter() {
                if a.iter().sum::<f64>() != 1f64 {
                    return Err(MDPError::InvalidTransitionMatrix);
                }
            }
        }
        if rewards.len() != n_states
            || rewards[0].len() != Move::len()
            || rewards[0][0].len() != n_states
        {
            return Err(MDPError::InvalidRewardMatrix);
        }

        for r in 0..rows {
            for c in 0..columns {
                let state = Tile {
                    id: r * columns + c,
                    x: r,
                    y: c,
                };

                if is_terminal_state(&state) {
                    terminal_states.insert(state.id);
                }

                states.push(state);
            }
        }

        Ok(Self {
            rows,
            columns,
            states,
            transition_probabilities,
            rewards,
            terminal_states,
        })
    }

    /// Creates a Grid World where each movement action has a state-independent transition model,
    /// and a fixed reward, that is, they are independent of state transition triplets.
    ///
    /// The transition model is a currying function that takes a movement action and returns
    /// a directional function. The directional function specifies the probability of moving'
    /// in a specific direction given the action. For instace, if the action is `Move::North`,
    /// then the probability of actually moving north could be 80%, while the probability of
    /// moving in another direction due to uncertainty could be 10% for east and 10% for west.
    ///
    /// # Arguments
    ///
    /// - `rows` - number of rows
    /// - `columns` - number of columns
    /// - `is_wall` - a function checking if any given tile is a wall
    /// - `transition_model` - a currying function that takes a movement action and returns a directional function
    /// - `reward` - a function assigning a reward to any given tile state
    /// - `is_terminal_state` - a function checking if any given tile state is terminal
    pub fn from<'a>(
        rows: usize,
        columns: usize,
        is_wall: fn(&Tile) -> bool,
        transition_model: fn(&Move) -> fn(&Move) -> f64,
        reward: fn(&Tile) -> f64,
        is_terminal_state: fn(&Tile) -> bool,
    ) -> Result<Self, MDPError<'a, Tile>> {
        // Check if the grid is empty
        if rows == 0 || columns == 0 {
            return Err(MDPError::Empty);
        }

        let n_states = rows * columns;
        let mut states = Vec::with_capacity(n_states);
        let mut terminal_states = HashSet::new();

        // Create the states
        for r in 0..rows {
            for c in 0..columns {
                let state = Tile {
                    id: r * columns + c,
                    x: r,
                    y: c,
                };

                if is_terminal_state(&state) {
                    terminal_states.insert(state.id());
                }

                states.push(state);
            }
        }

        let mut transition_probabilities = vec![vec![vec![0.0; n_states]; Move::len()]; n_states];
        let mut rewards = vec![vec![vec![0.0; n_states]; Move::len()]; n_states];

        for state in states.iter() {
            // Terminal states are dead ends
            if is_terminal_state(state) {
                for action in Move::ACTIONS.iter() {
                    transition_probabilities[state.id()][action.id()][state.id()] = 1.0;
                }
                continue;
            }

            for action in Move::ACTIONS.iter() {
                // North state relative to the current state
                let north_state_id = if state.x == 0 {
                    state.id
                } else {
                    state.y + (state.x - 1) * columns
                };

                if is_wall(&states[north_state_id]) {
                    transition_probabilities[state.id][action.id()][state.id] +=
                        transition_model(action)(&Move::North);
                } else {
                    transition_probabilities[state.id][action.id()][north_state_id] +=
                        transition_model(action)(&Move::North);

                    rewards[state.id][action.id()][north_state_id] =
                        reward(&states[north_state_id]);
                }

                // South state relative to the current state
                let south_state_id = if state.x == rows - 1 {
                    state.id
                } else {
                    state.y + (state.x + 1) * columns
                };

                if is_wall(&states[south_state_id]) {
                    transition_probabilities[state.id][action.id()][state.id] +=
                        transition_model(action)(&Move::South);
                } else {
                    transition_probabilities[state.id][action.id()][south_state_id] +=
                        transition_model(action)(&Move::South);

                    rewards[state.id][action.id()][south_state_id] =
                        reward(&states[south_state_id]);
                }

                // West state relative to the current state
                let west_state_id = if state.y == 0 {
                    state.id
                } else {
                    (state.y - 1) + state.x * columns
                };

                if is_wall(&states[west_state_id]) {
                    transition_probabilities[state.id][action.id()][state.id] +=
                        transition_model(action)(&Move::West);
                } else {
                    transition_probabilities[state.id][action.id()][west_state_id] +=
                        transition_model(action)(&Move::West);

                    rewards[state.id][action.id()][west_state_id] = reward(&states[west_state_id]);
                }

                // East state relative to the current state
                let east_state_id = if state.y == columns - 1 {
                    state.id
                } else {
                    (state.y + 1) + state.x * columns
                };

                if is_wall(&states[east_state_id]) {
                    transition_probabilities[state.id][action.id()][state.id] +=
                        transition_model(action)(&Move::East);
                } else {
                    transition_probabilities[state.id][action.id()][east_state_id] +=
                        transition_model(action)(&Move::East);

                    rewards[state.id][action.id()][east_state_id] = reward(&states[east_state_id]);
                }
            }
        }

        // Check if the transition probabilities sum to 1 for each action
        for t in transition_probabilities.iter() {
            for a in t.iter() {
                if a.iter().sum::<f64>() != 1f64 {
                    return Err(MDPError::InvalidTransitionMatrix);
                }
            }
        }

        Ok(Self {
            rows,
            columns,
            states,
            transition_probabilities,
            rewards,
            terminal_states,
        })
    }

    /// In the corner problem the upper-left corner and the bottom-right corner
    /// are self-absorbing terminal states. Each transition that is not terminal
    /// results in a reward penalty of -1. Agent movement success is user-defined.
    ///
    /// # Arguments
    ///
    /// - `rows` - number of rows
    /// - `columns` - number of columns
    /// - `uncertainty` - the probability of an action to fail, thus remaining in the same state
    pub fn corner<'a>(
        rows: usize,
        columns: usize,
        uncertainty: f64,
    ) -> Result<Self, MDPError<'a, Tile>> {
        // Check if the grid is empty
        if rows == 0 || columns == 0 {
            return Err(MDPError::Empty);
        }

        let n_states = rows * columns;
        let mut states = Vec::with_capacity(n_states);
        let mut transition_probabilities = vec![vec![vec![0.0; n_states]; Move::len()]; n_states];
        let mut rewards = vec![vec![vec![0.0; n_states]; Move::len()]; n_states];
        let mut terminal_states = HashSet::new();

        for r in 0..rows {
            for c in 0..columns {
                let state = Tile {
                    id: r * columns + c,
                    x: r,
                    y: c,
                };

                if state.id == 0 || state.id == n_states - 1 {
                    terminal_states.insert(state.id);
                    // Self absorbing states (all actions result to the same state)
                    for action in Move::ACTIONS.iter() {
                        transition_probabilities[state.id][action.id()][state.id] = 1f64;
                    }
                } else {
                    for action in Move::ACTIONS.iter() {
                        let next_state_id = match action {
                            Move::North => {
                                if r == 0 {
                                    c
                                } else {
                                    c + (r - 1) * columns
                                }
                            }
                            Move::South => {
                                if r == rows - 1 {
                                    c + (rows - 1) * columns
                                } else {
                                    c + (r + 1) * columns
                                }
                            }
                            Move::East => {
                                if c == columns - 1 {
                                    (columns - 1) + r * columns
                                } else {
                                    (c + 1) + r * columns
                                }
                            }
                            Move::West => {
                                if c == 0 {
                                    r * columns
                                } else {
                                    (c - 1) + r * columns
                                }
                            }
                        };

                        // find adjacent
                        if state.id == next_state_id {
                            transition_probabilities[state.id][action.id()][next_state_id] = 1f64;
                        } else {
                            transition_probabilities[state.id][action.id()][next_state_id] =
                                uncertainty;
                            transition_probabilities[state.id][action.id()][state.id] =
                                1f64 - uncertainty;
                        }
                        // all transitions from a non-terminal state should have a negative reward in order to
                        // force the optimal policy to account for the shorter amount of transitions.
                        rewards[state.id][action.id()][next_state_id] = -1f64;
                    }
                }

                // push the tile in the state vector
                states.push(state);
            }
        }

        // Check if the transition probabilities sum to 1 for each action
        for t in transition_probabilities.iter() {
            for a in t.iter() {
                if a.iter().sum::<f64>() != 1f64 {
                    return Err(MDPError::InvalidTransitionMatrix);
                }
            }
        }

        Ok(Self {
            rows,
            columns,
            states,
            transition_probabilities,
            rewards,
            terminal_states,
        })
    }
}

impl MDP<Tile, Move> for GridWorld {
    fn n_states(&self) -> usize {
        self.rows * self.columns
    }

    fn n_actions(&self) -> usize {
        Move::len()
    }

    fn states(&self) -> &[Tile] {
        &self.states
    }

    fn actions(&self) -> &[Move] {
        &Move::ACTIONS
    }

    fn is_terminal(&self, state: &Tile) -> bool {
        self.terminal_states.contains(&state.id())
    }

    fn transition_probability(&self, state: &Tile, action: &Move, next_state: &Tile) -> f64 {
        self.transition_probabilities[state.id()][action.id()][next_state.id()]
    }

    fn reward(&self, state: &Tile, action: &Move, next_state: &Tile) -> f64 {
        self.rewards[state.id()][action.id()][next_state.id()]
    }

    fn act(&self, state: &Tile, action: &Move) -> &Tile {
        let probs = &self.transition_probabilities[state.id()][action.id()];
        let next_state_id = WeightedIndex::new(probs)
            .unwrap()
            .sample(&mut rand::thread_rng());

        &self.states()[next_state_id]
    }
}

#[cfg(test)]
mod tests {
    use crate::mdp::{
        environment::{GridWorld, Move},
        model::MDPError,
    };

    #[test]
    fn empty_grid() {
        if let Err(error) = GridWorld::new(0, 0, vec![], vec![], |_| false) {
            assert_eq!(error, MDPError::Empty);
        }
    }

    #[test]
    fn invalid_matrices() {
        // The transition matrix does not have proper dimensions
        let transitions = vec![vec![vec![0f64; 4]; 16]; 16];
        let rewards = vec![vec![vec![0f64; 16]; 4]; 16];

        if let Err(error) = GridWorld::new(4, 4, transitions, rewards, |_| false) {
            assert_eq!(error, MDPError::InvalidTransitionMatrix);
        }

        // The transition matrix does not sum to 1
        let transitions = vec![vec![vec![0f64; 16]; 4]; 16];
        let rewards = vec![vec![vec![0f64; 16]; 4]; 16];

        if let Err(error) = GridWorld::new(4, 4, transitions, rewards, |_| false) {
            assert_eq!(error, MDPError::InvalidTransitionMatrix);
        }

        // The reward matrix does not have proper dimensions
        let transitions = vec![vec![vec![1.0 / 16.0; 16]; 4]; 16];
        let rewards = vec![vec![vec![0f64; 4]; 16]; 16];

        if let Err(error) = GridWorld::new(4, 4, transitions, rewards, |_| false) {
            assert_eq!(error, MDPError::InvalidRewardMatrix);
        }
    }

    #[test]
    fn state_independent_world() {
        let grid = GridWorld::from(
            2,
            2,
            |_| false, // there are no walls
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
            |s| if s.id == 3 { 10f64 } else { -1f64 },
            |s| s.id == 3,
        )
        .unwrap();

        assert_eq!(grid.rows, 2);
        assert_eq!(grid.columns, 2);
        assert_eq!(grid.terminal_states.len(), 1);
    }

    #[test]
    fn corner_problem() {
        let grid = GridWorld::corner(3, 3, 0.8).unwrap();

        assert_eq!(grid.rows, 3);
        assert_eq!(grid.columns, 3);
        assert_eq!(grid.terminal_states.len(), 2);
    }
}

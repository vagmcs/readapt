use rand::{random, Rng};

pub trait Bandit {
    /// Selects an arm to pull.
    fn select_arm(&mut self) -> usize;
    /// Rewards the bandit for the selected arm.
    fn receive_reward(&mut self, reward: f64);
    /// Restarts the bandit by clearing the internal state.
    fn restart(&mut self);
}

#[derive(Debug, Default, Clone, PartialEq)]
struct BanditState {
    steps: usize,
    n_available_arms: usize,
    selected_arm: usize,
    arm_pulls: Vec<usize>,
    initial_value: f64,
    estimated_arm_values: Vec<f64>,
}

impl BanditState {
    /// Creates an internal state for a given number of arms.
    ///
    /// - `n_available_arms` - the number of arms.
    fn new(n_available_arms: usize) -> BanditState {
        BanditState {
            steps: 0,
            n_available_arms,
            selected_arm: 0,
            arm_pulls: vec![0; n_available_arms],
            initial_value: 0_f64,
            estimated_arm_values: vec![0_f64; n_available_arms],
        }
    }

    /// Creates an internal state for a given number of arms. The state assumes an initial
    /// biased belief for the value of each arm.
    ///
    /// - `n_available_arms` - the number of arms.
    /// - `initial_value` - the initial value of each arm.
    fn biased(n_available_arms: usize, initial_value: f64) -> BanditState {
        BanditState {
            steps: 0,
            n_available_arms,
            selected_arm: 0,
            arm_pulls: vec![0; n_available_arms],
            initial_value,
            estimated_arm_values: vec![initial_value; n_available_arms],
        }
    }
}

#[derive(Debug, Clone)]
enum BanditAlgorithm {
    EpsilonGreedy(EpsilonGreedy),
    UCB(UCB),
}

#[derive(Debug, Default, Clone)]
struct EpsilonGreedy {
    epsilon: f64,
}

#[derive(Debug, Default, Clone)]
struct UCB {
    exploration_degree: f64,
}

/// Stochastic bandits support the following algorithms:
///
/// - greedy
/// - ε-greedy
/// - Upper Confidence Bound (UCB)
#[derive(Debug, Clone)]
pub struct StochasticBandit {
    state: BanditState,
    algorithm: BanditAlgorithm,
    learning_rate: Option<f64>,
}

impl StochasticBandit {
    /// Creates a greedy stochastic bandit. The greedy bandit is a sample-average method
    /// for estimating action values because each estimate is an average of the sample of
    /// relevant rewards. Greedy bandits, never explore inferior actions.
    ///
    /// - `arms` - the number of available arms.
    pub fn greedy(arms: usize) -> StochasticBandit {
        StochasticBandit {
            state: BanditState::new(arms),
            algorithm: BanditAlgorithm::EpsilonGreedy(EpsilonGreedy { epsilon: 0_f64 }),
            learning_rate: None,
        }
    }

    /// Creates an epsilon-greedy stochastic bandit. In contrast to the greedy bandit which
    /// always exploits current knowledge to maximize immediate reward, epsilon-greedy, every
    /// once in a while, with a small probability epsilon, selects randomly from among all the
    /// actions, independently of the action-value estimates.
    ///
    /// - `arms` - the number of available arms.
    /// - `epsilon` - exploration probability.
    pub fn epsilon_greedy(arms: usize, epsilon: f64) -> StochasticBandit {
        StochasticBandit {
            state: BanditState::new(arms),
            algorithm: BanditAlgorithm::EpsilonGreedy(EpsilonGreedy { epsilon }),
            learning_rate: None,
        }
    }

    /// Creates an Upper-Confidence-Bound (UCB) stochastic bandit. In contrast to the
    /// epsilon-greedy bandit, which explores actions with no preference for those that
    /// are nearly greedy or particularly uncertain, UCB takes into account both how close
    /// their estimates are to being maximal and their uncertainties.
    ///
    /// - `arms` - the number of available arms.
    /// - `exploration_degree` - the degree of exploration
    pub fn ucb(arms: usize, exploration_degree: f64) -> StochasticBandit {
        StochasticBandit {
            state: BanditState::new(arms),
            algorithm: BanditAlgorithm::UCB(UCB { exploration_degree }),
            learning_rate: None,
        }
    }

    pub fn with_constant_learning_rate(self, learning_rate: f64) -> StochasticBandit {
        if learning_rate <= 0.0 || learning_rate > 1.0 {
            panic!("Invalid alpha value: {learning_rate}");
        }

        StochasticBandit {
            state: self.state,
            algorithm: self.algorithm,
            learning_rate: Some(learning_rate),
        }
    }

    pub fn with_biased_state(self, value: f64) -> StochasticBandit {
        StochasticBandit {
            state: BanditState::biased(self.state.n_available_arms, value),
            algorithm: self.algorithm,
            learning_rate: self.learning_rate,
        }
    }
}

impl Bandit for StochasticBandit {
    fn select_arm(&mut self) -> usize {
        if self.state.steps == 0 {
            self.state.selected_arm = self
                .state
                .estimated_arm_values
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(index, _)| index)
                .unwrap();
        }

        match &self.algorithm {
            BanditAlgorithm::EpsilonGreedy(bandit) => {
                // select the next action either randomly or according to the maximum estimated value
                let exploration_probability: f64 = random();
                if exploration_probability > 1.0 - bandit.epsilon {
                    self.state.selected_arm =
                        rand::thread_rng().gen_range(0..self.state.n_available_arms);
                } else {
                    self.state.selected_arm = self
                        .state
                        .estimated_arm_values
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.total_cmp(b))
                        .map(|(index, _)| index)
                        .unwrap();
                }
            }
            BanditAlgorithm::UCB(bandit) => {
                self.state.selected_arm = self
                    .state
                    .estimated_arm_values
                    .iter()
                    .enumerate()
                    .map(|(i, v)| {
                        v + bandit.exploration_degree
                            * f64::sqrt(
                                f64::ln(self.state.steps as f64) / self.state.arm_pulls[i] as f64,
                            )
                    })
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.total_cmp(b))
                    .map(|(index, _)| index)
                    .unwrap();
            }
        }

        self.state.selected_arm
    }

    fn receive_reward(&mut self, reward: f64) {
        // increment the arm pulls
        self.state.steps += 1;
        self.state.arm_pulls[self.state.selected_arm] += 1;

        // determine the step size (learning rate)
        let alpha = self
            .learning_rate
            .unwrap_or(1.0 / self.state.arm_pulls[self.state.selected_arm] as f64);

        // update the estimated value for the best action
        self.state.estimated_arm_values[self.state.selected_arm] +=
            alpha * (reward - self.state.estimated_arm_values[self.state.selected_arm])
    }

    fn restart(&mut self) {
        self.state.steps = 0;
        self.state.selected_arm = 0;
        self.state.arm_pulls = vec![0; self.state.n_available_arms];
        self.state.estimated_arm_values =
            vec![self.state.initial_value; self.state.n_available_arms];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_bandit() {
        let mut greedy_bandit = StochasticBandit::greedy(5);

        assert_eq!(greedy_bandit.state.n_available_arms, 5);
        // assert_eq!(greedy_bandit.epsilon, 0.0);
        assert_eq!(greedy_bandit.state.estimated_arm_values, vec![0.0; 5]);

        greedy_bandit.receive_reward(1.0);
        assert_eq!(greedy_bandit.select_arm(), 0);
        assert_eq!(
            greedy_bandit.state.estimated_arm_values,
            vec![1.0, 0.0, 0.0, 0.0, 0.0]
        );

        greedy_bandit.receive_reward(-5.0);
        assert_eq!(
            greedy_bandit.state.estimated_arm_values,
            vec![-2.0, 0.0, 0.0, 0.0, 0.0]
        );
        assert_ne!(greedy_bandit.select_arm(), 0);
    }

    #[test]
    fn epsilon_greedy_bandit() {
        let epsilon_greedy_bandit = StochasticBandit::epsilon_greedy(10, 0.05);

        assert_eq!(epsilon_greedy_bandit.state.n_available_arms, 10);
        // assert_eq!(epsilon_greedy_bandit.algorithm.epsilon, 0.05);
        assert_eq!(
            epsilon_greedy_bandit.state.estimated_arm_values,
            vec![0.0; 10]
        );
        assert_eq!(epsilon_greedy_bandit.state.arm_pulls, vec![0; 10]);
        assert_eq!(epsilon_greedy_bandit.state.selected_arm, 0);
    }

    #[test]
    fn constant_learning_rate() {
        let mut bandit = StochasticBandit::greedy(5)
            .with_constant_learning_rate(0.5)
            .with_biased_state(1.5)
            .with_constant_learning_rate(1.0);

        bandit.restart();

        assert_eq!(bandit.learning_rate, Some(1.0));
        assert_eq!(bandit.state.estimated_arm_values, vec![1.5; 5]);
    }

    #[test]
    #[should_panic(expected = "Invalid alpha value: 0")]
    fn zero_learning_rate() {
        StochasticBandit::greedy(5).with_constant_learning_rate(0.0);
    }
}

use crate::bandits::arm::{Arm, MultiArm, RandomArm};
use crate::bandits::bandit::Bandit;
use rand::distributions::Distribution;
use rand_distr::Normal;
use std::cmp::Ordering;

#[derive(Clone, Debug)]
pub struct Result {
    pub average_reward_history: Vec<Vec<f64>>,
    pub optimal_action_percentage_history: Vec<Vec<f64>>,
}

#[derive(Clone, Debug)]
pub struct Benchmark<A: Arm> {
    arm: MultiArm<A>,
    bandits: Vec<Bandit>,
}

impl<A: Arm> Benchmark<A> {
    pub fn run(&self, runs: usize, steps: usize) -> Result {
        // average reward and optimal actions statistics across runs
        let mut average_reward_history = vec![vec![0.0; steps]; self.bandits.len()];
        let mut optimal_action_percentage_history = vec![vec![0.0; steps]; self.bandits.len()];

        // find optimal arm
        let optimal_arm = self.arm.optimal_arm();

        // run the benchmark
        for _ in 0..runs {
            let mut bandits: Vec<_> = self
                .bandits
                .iter()
                .map(|x| x.having_init_values(x.init_value))
                .collect();

            for t in 0..steps {
                for (i, bandit) in bandits.iter_mut().enumerate() {
                    let arm = bandit.select_arm();
                    let reward = self.arm.pull(arm);
                    average_reward_history[i][t] += reward;
                    if optimal_arm.map(|j| j == arm).unwrap_or(false) {
                        optimal_action_percentage_history[i][t] += 1.0;
                    }
                    bandit.receive_reward(reward);
                }
            }
        }

        // average results over the number of runs
        for t in 0..steps {
            for i in 0..self.bandits.len() {
                average_reward_history[i][t] /= runs as f64;
                optimal_action_percentage_history[i][t] /= runs as f64;
            }
        }

        Result {
            average_reward_history,
            optimal_action_percentage_history,
        }
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//
//     #[test]
//     fn test() {
//         let mut bandits = vec![Bandit::greedy(10), Bandit::epsilon_greedy(10, 0.1)];
//
//         let normal = Normal::new(0.0, 1.0).unwrap();
//         let arms = (0..10)
//             .map(|_| RandomArm::normal(normal.sample(&mut rand::thread_rng())))
//             .collect();
//
//         let result = MultiArm::new(arms).benchmark(100, 10, &mut bandits);
//
//         assert_eq!(result.average_reward_history.len(), 2);
//         assert_eq!(result.optimal_action_percentage_history.len(), 2);
//     }
// }

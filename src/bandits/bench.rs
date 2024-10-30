use crate::bandits::arm::{Arm, MultiArm};
use crate::bandits::bandit::Bandit;

#[derive(Clone, Debug)]
pub struct BenchmarkResult {
    /// Average reward history is the average reward for each step across N runs.
    pub average_reward_history: Vec<Vec<f64>>,
    /// Optimal action history is the percentage of steps where each bandit chose the optimal action.
    /// Note that this statistic is measured only if the true value of each arm is provided.
    pub optimal_action_percentage_history: Option<Vec<Vec<f64>>>,
}

#[derive(Clone, Debug)]
pub struct Benchmark<A: Arm> {
    arm: MultiArm<A>,
    bandits: Vec<Bandit>,
}

impl<A: Arm> Benchmark<A> {
    /// Runs a benchmark on the provided bandits, for a specified number of steps, and averages
    /// the results across all runs.
    ///
    /// - `runs` - the number of repeated runs.
    /// - `steps` - the number of steps per run.
    pub fn run(&self, runs: usize, steps: usize) -> BenchmarkResult {
        // find optimal arm
        let optimal_arm = self.arm.optimal_arm();

        // average reward and optimal actions statistics across runs
        let mut average_reward_history = vec![vec![0.0; steps]; self.bandits.len()];
        let mut optimal_action_percentage_history = vec![vec![0.0; steps]; self.bandits.len()];

        // run the benchmark
        for _ in 0..runs {
            // restart all bandits
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

        BenchmarkResult {
            average_reward_history,
            optimal_action_percentage_history: optimal_arm
                .map(|_| optimal_action_percentage_history),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bandits::arm::RandomArm;
    use rand_distr::{Distribution, Normal};

    #[test]
    fn test() {
        let bandits = vec![Bandit::greedy(10), Bandit::epsilon_greedy(10, 0.1)];

        let multi_arm = MultiArm::new(
            Normal::new(0.0, 1.0)
                .unwrap()
                .sample_iter(&mut rand::thread_rng())
                .take(10)
                .map(RandomArm::normal)
                .collect(),
        );

        let result = Benchmark {
            arm: multi_arm,
            bandits,
        }
        .run(100, 100);

        assert_eq!(result.average_reward_history.len(), 2);
        assert!(result.optimal_action_percentage_history.is_some());
        assert_eq!(result.optimal_action_percentage_history.unwrap().len(), 2);
    }
}

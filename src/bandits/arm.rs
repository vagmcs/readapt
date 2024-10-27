use super::bandit::Bandit;
use rand_distr::Distribution;
use rand_distr::Normal;

#[derive(Clone, Debug)]
struct Arm {
    value: f32,
    noisy_reward: Normal<f32>,
}

impl Arm {
    fn noisy_normal(value: f32) -> Arm {
        Arm {
            value,
            noisy_reward: Normal::new(value, 1.0).unwrap(),
        }
    }

    fn pull(&self) -> f32 {
        self.noisy_reward.sample(&mut rand::thread_rng())
    }
}

pub struct MultiArm {
    arms: Vec<Arm>,
}

impl MultiArm {
    pub fn new(n_arms: usize) -> MultiArm {
        let normal = Normal::new(0.0, 1.0).unwrap();

        MultiArm {
            arms: (0..n_arms)
                .map(|_| Arm::noisy_normal(normal.sample(&mut rand::thread_rng())))
                .collect(),
        }
    }

    pub fn benchmark(&self, runs: usize, steps: usize, bandits: &mut Vec<Bandit>) -> Result {
        // average reward and optimal actions statistics across runs
        let mut average_reward_history = vec![vec![0.0; steps]; bandits.len()];
        let mut optimal_action_percentage_history = vec![vec![0.0; steps]; bandits.len()];

        // find optimal arm
        let optimal_arm = self
            .arms
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.value.total_cmp(&b.value))
            .map(|(index, _)| index)
            .unwrap();

        // run the benchmark
        for _ in 0..runs {
            let mut bandits: Vec<_> = bandits.iter().map(|x| x.having_init_values(x.init_value)).collect();

            for t in 0..steps {
                for (i, bandit) in bandits.iter_mut().enumerate() {
                    let arm = bandit.select_arm();
                    let reward = self.arms[arm].pull();
                    average_reward_history[i][t] += reward;
                    if arm == optimal_arm {
                        optimal_action_percentage_history[i][t] += 1.0;
                    }
                    bandit.receive_reward(reward);
                }
            }
        }

        // average results over the number of runs
        for t in 0..steps {
            for i in 0..bandits.len() {
                average_reward_history[i][t] /= runs as f32;
                optimal_action_percentage_history[i][t] /= runs as f32;
            }
        }

        Result {
            average_reward_history,
            optimal_action_percentage_history,
        }
    }
}

#[derive(Debug)]
pub struct Result {
    pub average_reward_history: Vec<Vec<f32>>,
    pub optimal_action_percentage_history: Vec<Vec<f32>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {
        let mut bandits = vec![Bandit::greedy(10), Bandit::epsilon_greedy(10, 0.1)];
        let result = MultiArm::new(10).benchmark(100, 10, &mut bandits);

        assert_eq!(result.average_reward_history.len(), 2);
        assert_eq!(result.optimal_action_percentage_history.len(), 2);
    }
}

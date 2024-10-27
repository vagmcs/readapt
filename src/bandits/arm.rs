use super::bandit::Bandit;
use rand_distr::Distribution;
use rand_distr::Normal;

#[derive(Clone, Debug)]
struct Arm {
    value: f32,
}

impl Arm {
    pub fn standard_normal() -> Arm {
        Arm {
            value: Normal::new(0.0, 1.0)
                .unwrap()
                .sample(&mut rand::thread_rng()),
        }
    }

    pub fn pull(&self) -> f32 {
        Normal::new(self.value, 1.0)
            .unwrap()
            .sample(&mut rand::thread_rng())
    }
}

struct MultiArm {
    arms: Vec<Arm>,
}

impl MultiArm {
    fn new(n_arms: usize) -> MultiArm {
        MultiArm {
            arms: (0..n_arms).map(|_| Arm::standard_normal()).collect(),
        }
    }

    fn benchmark(&self, runs: usize, steps: usize, bandits: &mut Vec<Bandit>) -> Result {
        let mut average_reward_history = vec![vec![0.0; steps]; bandits.len()];
        let mut optimal_action_percentage_history = vec![vec![0.0; steps]; bandits.len()];

        // find optimal arm
        let mut optimal_arm = usize::MIN;
        let mut max_value = f32::NEG_INFINITY;
        for (i, arm) in self.arms.iter().enumerate() {
            if arm.value > max_value {
                max_value = arm.value;
                optimal_arm = i;
            }
        }

        // run the benchmark
        for _ in 0..runs {
            for t in 0..steps {
                for (i, bandit) in bandits.iter_mut().enumerate() {
                    let arm = bandit.select_arm();
                    let r = self.arms[arm].pull();
                    average_reward_history[i][t] += r;
                    if arm == optimal_arm {
                        optimal_action_percentage_history[i][t] += 1.0;
                    }
                    bandit.receive_reward(r);
                }
            }
        }

        // normalize statistics
        for t in 0..steps {
            for (i, _) in bandits.iter_mut().enumerate() {
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
struct Result {
    average_reward_history: Vec<Vec<f32>>,
    optimal_action_percentage_history: Vec<Vec<f32>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {
        let mut bandits = vec![Bandit::greedy(10), Bandit::epsilon_greedy(10, 0.1)];

        let result = MultiArm::new(10).benchmark(1000, 10, &mut bandits);

        assert_eq!(result.average_reward_history.len(), 2);
        assert_eq!(result.optimal_action_percentage_history.len(), 2);

        println!("{:?}", result.average_reward_history);
        println!("{:?}", result.optimal_action_percentage_history);
    }
}

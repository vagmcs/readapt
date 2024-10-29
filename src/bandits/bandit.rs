use rand::{random, Rng};

#[derive(Debug, Default, Clone, PartialEq)]
pub struct Bandit {
    n_arms: usize,
    c: Option<f64>,
    epsilon: f64,
    learning_rate: Option<f64>,
    estimated_arm_values: Vec<f64>,
    arm_pulls: Vec<u32>,
    action: usize,
    pub init_value: f64,
    step: usize,
}

impl Bandit {
    pub fn greedy(arms: usize) -> Bandit {
        Bandit {
            n_arms: arms,
            c: None,
            epsilon: 0_f64,
            learning_rate: None,
            estimated_arm_values: vec![0_f64; arms],
            arm_pulls: vec![0; arms],
            action: 0,
            init_value: 0_f64,
            step: 0,
        }
    }

    pub fn epsilon_greedy(arms: usize, epsilon: f64) -> Bandit {
        Bandit {
            n_arms: arms,
            c: None,
            epsilon,
            learning_rate: None,
            estimated_arm_values: vec![0_f64; arms],
            arm_pulls: vec![0; arms],
            action: 0,
            init_value: 0_f64,
            step: 0,
        }
    }

    pub fn ucb(arms: usize, c: f64) -> Bandit {
        Bandit {
            n_arms: arms,
            c: Some(c),
            epsilon: 0_f64,
            learning_rate: None,
            estimated_arm_values: vec![0_f64; arms],
            arm_pulls: vec![0; arms],
            action: 0,
            init_value: 0_f64,
            step: 0,
        }
    }

    pub fn having_constant_learning_rate(&self, alpha: f64) -> Bandit {
        if alpha <= 0.0 || alpha > 1.0 {
            panic!("Invalid alpha value: {alpha}");
        }

        Bandit {
            n_arms: self.n_arms,
            c: self.c,
            epsilon: self.epsilon,
            learning_rate: Some(alpha),
            estimated_arm_values: self.estimated_arm_values.clone(),
            arm_pulls: vec![0; self.n_arms],
            action: 0,
            init_value: self.init_value,
            step: 0,
        }
    }

    pub fn having_init_values(&self, value: f64) -> Bandit {
        Bandit {
            n_arms: self.n_arms,
            c: self.c,
            epsilon: self.epsilon,
            learning_rate: self.learning_rate,
            estimated_arm_values: vec![value; self.n_arms],
            arm_pulls: vec![0; self.n_arms],
            action: 0,
            init_value: value,
            step: 0,
        }
    }

    pub fn select_arm(&mut self) -> usize {
        if self.step == 0 {
            self.action = self
                .estimated_arm_values
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(index, _)| index)
                .unwrap();
        }
        if self.c.is_some() {
            self.action = self
                .estimated_arm_values
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    v + self.c.unwrap()
                        * f64::sqrt(f64::ln(self.step as f64) / self.arm_pulls[i] as f64)
                })
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(index, _)| index)
                .unwrap();
        } else {
            // select the next action either randomly or according to the maximum estimated value
            let exploration_probability: f64 = random();
            if exploration_probability > 1.0 - self.epsilon {
                self.action = rand::thread_rng().gen_range(0..self.n_arms);
            } else {
                self.action = self
                    .estimated_arm_values
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.total_cmp(b))
                    .map(|(index, _)| index)
                    .unwrap();
            }
        }

        self.action
    }

    pub fn receive_reward(&mut self, reward: f64) {
        // increment the arm pulls
        self.step += 1;
        self.arm_pulls[self.action] += 1;

        // determine the step size (learning rate)
        let alpha = self
            .learning_rate
            .unwrap_or(1.0 / self.arm_pulls[self.action] as f64);

        // update the estimated value for the best action
        self.estimated_arm_values[self.action] +=
            alpha * (reward - self.estimated_arm_values[self.action])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_bandit() {
        let mut greedy_bandit = Bandit::greedy(5);

        assert_eq!(greedy_bandit.n_arms, 5);
        assert_eq!(greedy_bandit.epsilon, 0.0);
        assert_eq!(greedy_bandit.estimated_arm_values, vec![0.0; 5]);

        greedy_bandit.receive_reward(1.0);
        assert_eq!(greedy_bandit.select_arm(), 0);
        assert_eq!(
            greedy_bandit.estimated_arm_values,
            vec![1.0, 0.0, 0.0, 0.0, 0.0]
        );

        greedy_bandit.receive_reward(-5.0);
        assert_eq!(
            greedy_bandit.estimated_arm_values,
            vec![-2.0, 0.0, 0.0, 0.0, 0.0]
        );
        assert_ne!(greedy_bandit.select_arm(), 0);
    }

    #[test]
    fn epsilon_greedy_bandit() {
        let epsilon_greedy_bandit = Bandit::epsilon_greedy(10, 0.05);

        assert_eq!(epsilon_greedy_bandit.n_arms, 10);
        assert_eq!(epsilon_greedy_bandit.epsilon, 0.05);
        assert_eq!(epsilon_greedy_bandit.estimated_arm_values, vec![0.0; 10]);
        assert_eq!(epsilon_greedy_bandit.arm_pulls, vec![0; 10]);
        assert_eq!(epsilon_greedy_bandit.action, 0);
    }

    #[test]
    fn constant_learning_rate() {
        let bandit = Bandit::greedy(5)
            .having_constant_learning_rate(0.5)
            .having_init_values(1.5)
            .having_constant_learning_rate(1.0);

        assert_eq!(bandit.learning_rate, Some(1.0));
        assert_eq!(bandit.estimated_arm_values, vec![1.5; 5]);
    }

    #[test]
    #[should_panic(expected = "Invalid alpha value: 0")]
    fn zero_learning_rate() {
        Bandit::greedy(5).having_constant_learning_rate(0.0);
    }
}

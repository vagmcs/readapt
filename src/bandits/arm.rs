use rand_distr::Distribution;
use rand_distr::Normal;

pub trait Arm {
    /// Return the true value of the arm or none if the value is unknown.
    fn value(&self) -> Option<f64>;

    /// Pulling the arm should yield a reward.
    fn pull(&self) -> f64;
}

/// Random arms sample rewards from an underlying reward distribution. The assumption is that
/// the true value of the arm is not directly observable.
#[derive(Clone, Debug)]
pub struct RandomArm<D: Distribution<f64>> {
    value: Option<f64>,
    reward_distribution: D,
}

impl<D: Distribution<f64>> RandomArm<D> {
    /// Creates a random arm from any distribution. The arm can optionally have an
    /// underlying true value.
    ///
    /// - `value` - the true value of the arm.
    /// - `reward_distribution` - the distribution from which the rewards are sampled.
    ///
    /// Note that the assumption is that the true value is a possible outcome of the provided
    /// reward distribution.
    pub fn from_distribution(value: Option<f64>, reward_distribution: D) -> Self {
        RandomArm {
            value,
            reward_distribution,
        }
    }
}

impl RandomArm<Normal<f64>> {
    /// A standard normal arm follows a normal reward distribution and yields random
    /// rewards around using the provided true value as the mean and unit variance.
    ///
    /// - `value` - mean of the reward distribution and the true value of the arm.
    ///
    /// # Example
    ///```
    /// use deathloop::bandits::arm::{Arm, RandomArm};
    ///
    /// // the following arm yield rewards following a standard normal distribution, i.e.,
    /// // having zero mean and unit variance.
    /// let arm = RandomArm::normal(0f64);
    /// println!("Pulling the arm! Received reward: {}", arm.pull())
    ///```
    pub fn normal(value: f64) -> Self {
        RandomArm {
            value: Some(value),
            reward_distribution: Normal::new(value, 1.0).unwrap(),
        }
    }
}

impl<D: Distribution<f64>> Arm for RandomArm<D> {
    fn value(&self) -> Option<f64> {
        self.value
    }

    fn pull(&self) -> f64 {
        self.reward_distribution.sample(&mut rand::thread_rng())
    }
}

#[derive(Clone, Debug)]
pub struct MultiArm<A: Arm> {
    arms: Vec<A>,
}

impl<A: Arm> MultiArm<A> {
    pub fn new(arms: Vec<A>) -> MultiArm<A> {
        MultiArm { arms }
    }

    pub fn pull(&self, k: usize) -> f64 {
        self.arms[k].pull()
    }

    pub fn optimal_arm(&self) -> Option<usize> {
        if self.arms.iter().any(|arm| arm.value().is_none()) {
            None
        } else {
            self.arms
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.value().unwrap().total_cmp(&b.value().unwrap()))
                .map(|(index, _)| index)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::distributions::Uniform;

    #[test]
    fn standard_normal_arm() {
        let arm = RandomArm::normal(0f64);

        assert_eq!(arm.value, Some(0f64));
        assert_eq!(arm.value(), arm.value);
    }

    #[test]
    fn uniform_arm() {
        let arm = RandomArm::from_distribution(None, Uniform::new(0.0, 1.0));

        assert_eq!(arm.value, None);
        assert_eq!(arm.value(), arm.value);

        let reward = arm.pull();
        assert!((0f64..1f64).contains(&reward));
    }

    #[test]
    fn optimal_arm() {
        let arms = vec![
            RandomArm::from_distribution(None, Uniform::new(0.0, 1.0)),
            RandomArm::from_distribution(Some(5f64), Uniform::new(-10.0, 10.0)),
            RandomArm::from_distribution(Some(0.5), Uniform::new(-1.0, 1.0)),
        ];

        let multi_arm = MultiArm::new(arms);

        assert_eq!(multi_arm.optimal_arm(), None);

        let arms = vec![
            RandomArm::from_distribution(Some(1f64), Uniform::new(0.0, 1.0)),
            RandomArm::from_distribution(Some(5f64), Uniform::new(-10.0, 10.0)),
            RandomArm::from_distribution(Some(0.5), Uniform::new(-1.0, 1.0)),
        ];

        let multi_arm = MultiArm::new(arms);

        assert_eq!(multi_arm.optimal_arm(), Some(1));
    }
}

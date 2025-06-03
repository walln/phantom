use phantom_core::{Result, Tensor};

#[derive(Clone)]
pub struct Linear {
    pub weight: Tensor,
}

impl Linear {
    pub fn new(weight: Tensor) -> Self {
        Self { weight }
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.matmul(&self.weight)
    }
}


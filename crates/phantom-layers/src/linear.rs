use phantom_core::{Tensor, TensorError};

#[derive(Clone)]
pub struct Linear {
    pub weight: Tensor,
}

impl Linear {
    pub fn new(weight: Tensor) -> Self {
        Self { weight }
    }

    pub fn forward(&self, input: &Tensor) -> std::result::Result<Tensor, TensorError> {
        input.matmul(&self.weight)
    }
}

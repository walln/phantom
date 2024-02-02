use crate::tensor::{Tensor, TensorID};
use crate::Result;
use std::collections::HashMap;

pub struct GradientStore(HashMap<TensorID, Tensor>);

impl GradientStore {
    pub fn new() -> Self {
        Self(HashMap::new())
    }

    pub fn get(&self, tensor: &Tensor) -> Option<&Tensor> {
        self.0.get(&tensor.id())
    }

    pub fn get_id(&self, id: TensorID) -> Option<&Tensor> {
        self.0.get(&id)
    }

    pub fn insert(&mut self, tensor: &Tensor, gradient: Tensor) {
        self.0.insert(tensor.id(), gradient);
    }

    pub fn or_insert(&mut self, tensor: &Tensor) -> Result<&mut Tensor> {
        use std::collections::hash_map::Entry;
        let grad = match self.0.entry(tensor.id()) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                let grad = tensor.zeros_like()?;
                entry.insert(grad)
            }
        };
        Ok(grad)
    }

    pub fn remove(&mut self, tensor: &Tensor) -> Option<Tensor> {
        self.0.remove(&tensor.id())
    }

    pub fn remove_id(&mut self, id: TensorID) -> Option<Tensor> {
        self.0.remove(&id)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }
}

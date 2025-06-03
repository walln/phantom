use crate::tensor::{Tensor, TensorID};
use crate::{Operation, Result, DType};
use std::collections::HashMap;

impl Tensor {
    /// Return all the nodes that lead to this node in the graph as a topologically sorted
    /// vector with earlier nodes having dependencies on later nodes.
    fn sorted_nodes(&self) -> Vec<&Tensor> {
        // The vector of sorted nodes is passed as an owned value to the recursive walk
        // as a way to avoid having to use a RefCell or Mutex to mutate the vector.
        fn walk<'a>(
            node: &'a Tensor,
            nodes: Vec<&'a Tensor>,
            seen: &mut HashMap<TensorID, bool>,
        ) -> (bool, Vec<&'a Tensor>) {
            if let Some(&target) = seen.get(&node.id()) {
                return (target, nodes);
            }

            let mut tracked = false;
            let mut nodes = if node.variable() {
                tracked = true;
                nodes
            } else if let Some(op) = node.op() {
                match op {
                    Operation::Add(lhs, rhs)
                    | Operation::Sub(lhs, rhs)
                    | Operation::Mul(lhs, rhs)
                    | Operation::Div(lhs, rhs) => {
                        let (target, nodes) = walk(lhs, nodes, seen);
                        tracked |= target;
                        let (target, nodes) = walk(rhs, nodes, seen);
                        tracked |= target;
                        nodes
                    }
                    Operation::Sqr(node)
                    | Operation::Sqrt(node)
                    | Operation::Neg(node)
                    | Operation::Sum(node) => {
                        let (target, nodes) = walk(node, nodes, seen);
                        tracked |= target;
                        nodes
                    }
                    Operation::Affine { node, mul, .. } => {
                        if *mul == 0. {
                            nodes
                        } else {
                            let (target, nodes) = walk(node, nodes, seen);
                            tracked |= target;
                            nodes
                        }
                    }
                }
            } else {
                nodes
            };

            seen.insert(node.id(), tracked);

            if tracked {
                nodes.push(node)
            }

            (tracked, nodes)
        }

        let (_tg, mut nodes) = walk(self, vec![], &mut HashMap::new());
        nodes.reverse();
        nodes
    }

    /// Compute the gradient of this node with respect to all the nodes in the graph.
    /// The result is a map from node id to gradient.
    /// The gradient of a node is the sum of the gradients of all the nodes that depend on it.
    /// The gradient of a node is computed by applying the chain rule to the node's operation.
    /// ```rust
    /// use phantom::{Tensor, Device};
    ///
    /// let x = Tensor::new(&[[2f32, 2.], [1f32, 2.]], Device::CPU)?;
    /// let y = Tensor::new(&[[2f32, 2.], [5f32, 6.]], Device::CPU)?;
    /// let z = x.add(&y)?;
    /// let gradients = z.backward()?;
    /// assert_eq!(gradients.len(), 1);
    ///
    /// # Ok::<(), phantom::Error>(())
    /// ```
    pub fn backward(&self) -> Result<HashMap<TensorID, Tensor>> {
        let sorted_nodes = self.sorted_nodes();
        let mut gradients = HashMap::new();

        gradients.insert(self.id(), self.ones_like());

        for node in sorted_nodes.iter() {
            if node.variable() {
                continue;
            }

            let gradient = gradients.remove(&node.id()).unwrap();

            if let Some(op) = node.op() {
                match op {
                    Operation::Add(lhs, rhs) => {
                        let lhs_gradient_sum = gradients
                            .entry(lhs.id())
                            .or_insert_with(|| lhs.zeros_like());
                        *lhs_gradient_sum = lhs_gradient_sum.add(&gradient)?;
                        let rhs_gradient_sum = gradients
                            .entry(rhs.id())
                            .or_insert_with(|| rhs.zeros_like());
                        *rhs_gradient_sum = rhs_gradient_sum.add(&gradient)?;
                    }
                    Operation::Sub(lhs, rhs) => {
                        let lhs_gradient_sum = gradients
                            .entry(lhs.id())
                            .or_insert_with(|| lhs.zeros_like());
                        *lhs_gradient_sum = lhs_gradient_sum.add(&gradient)?;
                        let rhs_gradient_sum = gradients
                            .entry(rhs.id())
                            .or_insert_with(|| rhs.zeros_like());
                        *rhs_gradient_sum = rhs_gradient_sum.add(&gradient.neg()?)?;
                    }
                    Operation::Mul(lhs, rhs) => {
                        let lhs_gradient = gradient.mul(rhs)?;
                        let lhs_gradient_sum = gradients
                            .entry(lhs.id())
                            .or_insert_with(|| lhs.zeros_like());
                        *lhs_gradient_sum = lhs_gradient_sum.add(&lhs_gradient)?;
                        let rhs_gradient = gradient.mul(lhs)?;
                        let rhs_gradient_sum = gradients
                            .entry(rhs.id())
                            .or_insert_with(|| rhs.zeros_like());
                        *rhs_gradient_sum = rhs_gradient_sum.add(&rhs_gradient)?;
                    }
                    Operation::Div(lhs, rhs) => {
                        let lhs_gradient = gradient.div(rhs)?;
                        let lhs_gradient_sum = gradients
                            .entry(lhs.id())
                            .or_insert_with(|| lhs.zeros_like());
                        *lhs_gradient_sum = lhs_gradient_sum.add(&lhs_gradient)?;
                        let rhs_gradient = gradient.mul(lhs)?.div(&rhs.sqr()?)?;
                        let rhs_gradient_sum = gradients
                            .entry(rhs.id())
                            .or_insert_with(|| rhs.zeros_like());
                        *rhs_gradient_sum = rhs_gradient_sum.add(&rhs_gradient)?;
                    }
                    Operation::Affine { node, mul, .. } => {
                        let node_gradient = gradient.affine(*mul, 0.)?;
                        let gradient_sum = gradients
                            .entry(node.id())
                            .or_insert_with(|| node.zeros_like());
                        *gradient_sum = gradient_sum.add(&node_gradient)?
                    }
                    Operation::Sqr(node) => {
                        let node_gradient = node.mul(&gradient)?.affine(2., 0.)?;
                        let gradient_sum = gradients
                            .entry(node.id())
                            .or_insert_with(|| node.zeros_like());
                        *gradient_sum = gradient_sum.add(&node_gradient)?
                    }
                    Operation::Sqrt(node) => {
                        let node_gradient = gradient.div(node)?.affine(0.5, 0.)?;
                        let gradient_sum = gradients
                            .entry(node.id())
                            .or_insert_with(|| node.zeros_like());
                        *gradient_sum = gradient_sum.add(&node_gradient)?
                    }
                    Operation::Neg(node) => {
                        let node_gradient = gradient.neg()?;
                        let gradient_sum = gradients
                            .entry(node.id())
                            .or_insert_with(|| node.zeros_like());
                        *gradient_sum = gradient_sum.add(&node_gradient)?
                    }
                    Operation::Sum(node) => {
                        let scalar = match node.dtype() {
                            DType::F32 => gradient.to_scalar::<f32>()? as f64,
                            DType::F64 => gradient.to_scalar::<f64>()?,
                        };
                        let node_gradient = node.zeros_like().affine(0., scalar)?;
                        let gradient_sum = gradients
                            .entry(node.id())
                            .or_insert_with(|| node.zeros_like());
                        *gradient_sum = gradient_sum.add(&node_gradient)?
                    }
                }
            }
        }
        Ok(gradients)
    }
}

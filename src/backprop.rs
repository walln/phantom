use crate::gradient_store::GradientStore;
use crate::operation::{BinaryOperations, Operation, UnaryOperations};
use crate::tensor::{Tensor, TensorID};
use crate::Error;
use crate::Result;
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
            let mut nodes = if node.is_variable() {
                tracked = true;
                nodes
            } else if let Some(op) = node.op() {
                match op {
                    Operation::Broadcast(node)
                    | Operation::ToDType(node)
                    | Operation::Transpose(node, _, _)
                    | Operation::Copy(node)
                    | Operation::Unary(node, _) => {
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

                    Operation::Binary(lhs, rhs, _) | Operation::Matmul(lhs, rhs) => {
                        let (target, nodes) = walk(lhs, nodes, seen);
                        tracked |= target;
                        let (target, nodes) = walk(rhs, nodes, seen);
                        tracked |= target;
                        nodes
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
    /// let x = Tensor::new(&[[2f32, 2.], [1f32, 2.]], &Device::CPU)?;
    /// let y = Tensor::new(&[[2f32, 2.], [5f32, 6.]], &Device::CPU)?;
    /// let z = x.add(&y)?;
    /// let gradients = z.backward()?;
    /// assert_eq!(gradients.len(), 1);
    ///
    /// # Ok::<(), phantom::Error>(())
    /// ```
    pub fn backward(&self) -> Result<GradientStore> {
        let sorted_nodes = self.sorted_nodes();
        let mut gradients = GradientStore::new();

        gradients.insert(self, self.ones_like()?.contiguous()?);

        for node in sorted_nodes.iter() {
            if node.is_variable() {
                continue;
            }

            let gradient = gradients.remove(node).unwrap();

            if let Some(op) = node.op() {
                match op {
                    Operation::Binary(lhs, rhs, BinaryOperations::Add) => {
                        let lhs_gradient_sum = gradients.or_insert(lhs)?;
                        *lhs_gradient_sum = lhs_gradient_sum.add(&gradient)?;
                        let rhs_gradient_sum = gradients.or_insert(rhs)?;
                        *rhs_gradient_sum = rhs_gradient_sum.add(&gradient)?;
                    }
                    Operation::Binary(lhs, rhs, BinaryOperations::Sub) => {
                        let lhs_gradient_sum = gradients.or_insert(lhs)?;
                        *lhs_gradient_sum = lhs_gradient_sum.add(&gradient)?;
                        let rhs_gradient_sum = gradients.or_insert(rhs)?;
                        *rhs_gradient_sum = rhs_gradient_sum.sub(&gradient)?;
                    }
                    Operation::Binary(lhs, rhs, BinaryOperations::Mul) => {
                        let lhs_gradient = gradient.mul(rhs)?;
                        let lhs_gradient_sum = gradients.or_insert(lhs)?;
                        *lhs_gradient_sum = lhs_gradient_sum.add(&lhs_gradient)?;
                        let rhs_gradient = gradient.mul(lhs)?;
                        let rhs_gradient_sum = gradients.or_insert(rhs)?;
                        *rhs_gradient_sum = rhs_gradient_sum.add(&rhs_gradient)?;
                    }
                    Operation::Binary(lhs, rhs, BinaryOperations::Div) => {
                        let lhs_gradient = gradient.div(rhs)?;
                        let lhs_gradient_sum = gradients.or_insert(lhs)?;
                        *lhs_gradient_sum = lhs_gradient_sum.add(&lhs_gradient)?;
                        let rhs_gradient = gradient.mul(lhs)?.div(&rhs.sqr()?)?;
                        let rhs_gradient_sum = gradients.or_insert(rhs)?;
                        *rhs_gradient_sum = rhs_gradient_sum.add(&rhs_gradient)?;
                    }
                    Operation::Affine { node: arg, mul, .. } => {
                        let gradient_arg = gradient.affine(*mul, 0.)?;
                        let gradient_sum = gradients.or_insert(arg)?;
                        *gradient_sum = gradient_sum.add(&gradient_arg)?
                    }
                    Operation::Unary(arg, UnaryOperations::Sqr) => {
                        let gradient_arg = arg.mul(&gradient)?.affine(2., 0.)?;
                        let gradient_sum = gradients.or_insert(node)?;
                        *gradient_sum = gradient_sum.add(&gradient_arg)?
                    }
                    Operation::Unary(arg, UnaryOperations::Sqrt) => {
                        let gradient_arg = gradient.div(arg)?.affine(0.5, 0.)?;
                        let gradient_sum = gradients.or_insert(arg)?;
                        *gradient_sum = gradient_sum.add(&gradient_arg)?
                    }
                    Operation::Unary(arg, UnaryOperations::Neg) => {
                        let gradient_sum = gradients.or_insert(arg)?;
                        *gradient_sum = gradient_sum.sub(&gradient)?
                    }
                    Operation::Broadcast(_) => {
                        return Err(Error::BackwardUnsupported {
                            operation: "broadcast",
                        })
                    }
                    Operation::ToDType(arg) => {
                        let gradient_sum = gradients.or_insert(arg)?;
                        *gradient_sum = gradient_sum.add(&gradient.to_dtype(node.dtype())?)?
                    }
                    Operation::Matmul(lhs, rhs) => {
                        // Skipping checks, the op went ok, we can skip
                        // the matmul size checks for now.

                        let lhs_gradient = gradient.matmul(&rhs.t()?)?;
                        let lhs_gradient_sum = gradients.or_insert(lhs)?;
                        *lhs_gradient_sum = lhs_gradient_sum.add(&lhs_gradient)?;

                        let rhs_gradient = lhs.t()?.matmul(&gradient)?;
                        let rhs_gradient_sum = gradients.or_insert(rhs)?;
                        *rhs_gradient_sum = rhs_gradient_sum.add(&rhs_gradient)?;
                    }
                    Operation::Transpose(arg, dim1, dim2) => {
                        let gradient_arg = gradient.transpose(*dim1, *dim2)?;
                        let gradient_sum = gradients.or_insert(arg)?;
                        *gradient_sum = gradient_sum.add(&gradient_arg)?
                    }
                    Operation::Copy(arg) => {
                        let gradient_sum = gradients.or_insert(arg)?;
                        *gradient_sum = gradient_sum.add(&gradient)?
                    }
                }
            }
        }
        Ok(gradients)
    }
}

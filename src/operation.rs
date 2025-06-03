use crate::Tensor;

pub enum Operation {
    Add(Tensor, Tensor),
    Sub(Tensor, Tensor),
    Mul(Tensor, Tensor),
    Div(Tensor, Tensor),

    Sqr(Tensor),
    Sqrt(Tensor),
    Neg(Tensor),

    Sum(Tensor),

    Affine { node: Tensor, mul: f64, add: f64 },
}

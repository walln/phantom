use crate::Tensor;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOperations {
    Add,
    Mul,
    Sub,
    Div,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOperations {
    Sqr,
    Sqrt,
    Neg,
}

#[derive(Debug, Clone)]
pub enum Operation {
    Unary(Tensor, UnaryOperations),
    Binary(Tensor, Tensor, BinaryOperations),

    // Casting and complex ops
    Affine { node: Tensor, mul: f64, add: f64 },
    Broadcast(Tensor),
    Transpose(Tensor, usize, usize),
    Matmul(Tensor, Tensor),

    Copy(Tensor),
    ToDType(Tensor),
}

pub(crate) trait UnaryOperation {
    const NAME: &'static str;
    const KERNEL: &'static str;
    const V: Self;
    fn f32(v1: f32) -> f32;
    fn f64(v1: f64) -> f64;
    fn u32(v1: u32) -> u32;
}

pub(crate) trait BinaryOperation {
    const NAME: &'static str;
    const KERNEL: &'static str;
    const V: Self;
    fn f32(v1: f32, v2: f32) -> f32;
    fn f64(v1: f64, v2: f64) -> f64;
    fn u32(v1: u32, v2: u32) -> u32;
}

pub(crate) struct Add;
pub(crate) struct Sub;
pub(crate) struct Mul;
pub(crate) struct Div;
pub(crate) struct Sqr;
pub(crate) struct Sqrt;
pub(crate) struct Neg;

macro_rules! bin_op {
    ($op:ident, $name: literal, $e: expr) => {
        impl BinaryOperation for $op {
            const NAME: &'static str = $name;
            const KERNEL: &'static str = concat!("b", $name);
            const V: Self = $op;

            fn f32(v1: f32, v2: f32) -> f32 {
                $e(v1, v2)
            }

            fn f64(v1: f64, v2: f64) -> f64 {
                $e(v1, v2)
            }

            fn u32(v1: u32, v2: u32) -> u32 {
                $e(v1, v2)
            }
        }
    };
}

bin_op!(Add, "add", |v1, v2| v1 + v2);
bin_op!(Sub, "sub", |v1, v2| v1 - v2);
bin_op!(Mul, "mul", |v1, v2| v1 * v2);
bin_op!(Div, "div", |v1, v2| v1 / v2);

macro_rules! unary_op {
    ($op: ident, $name: literal, $a: ident, $e: expr) => {
        impl UnaryOperation for $op {
            const NAME: &'static str = $name;
            const KERNEL: &'static str = concat!("u", $name);
            const V: Self = $op;

            fn f32($a: f32) -> f32 {
                $e
            }

            fn f64($a: f64) -> f64 {
                $e
            }

            fn u32($a: u32) -> u32 {
                todo!("no unary function for u32")
            }
        }
    };
}

unary_op!(Sqr, "sqr", a, a * a);
unary_op!(Sqrt, "sqrt", a, a.sqrt());
unary_op!(Neg, "neg", a, -a);

#[derive(Clone, Debug)]
pub struct BackpropOperation(Option<Operation>);

impl BackpropOperation {
    pub(crate) fn none() -> Self {
        BackpropOperation(None)
    }

    pub(crate) fn new<A: AsRef<Tensor>>(args: &[A], f: impl Fn(Vec<Tensor>) -> Operation) -> Self {
        let operation = if args.iter().any(|arg| arg.as_ref().track_op()) {
            let args: Vec<Tensor> = args.iter().map(|arg| arg.as_ref().clone()).collect();
            Some(f(args))
        } else {
            None
        };
        Self(operation)
    }

    pub(crate) fn new_unary(arg: &Tensor, f: impl Fn(Tensor) -> Operation) -> Self {
        let operation = if arg.track_op() {
            Some(f(arg.clone()))
        } else {
            None
        };
        Self(operation)
    }

    pub(crate) fn new_binary(
        arg1: &Tensor,
        arg2: &Tensor,
        f: impl Fn(Tensor, Tensor) -> Operation,
    ) -> Self {
        let operation = if arg1.track_op() || arg2.track_op() {
            Some(f(arg1.clone(), arg2.clone()))
        } else {
            None
        };
        Self(operation)
    }
}

impl std::ops::Deref for BackpropOperation {
    type Target = Option<Operation>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

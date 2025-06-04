use phantom_core::{Device, Tensor};
use phantom_layers::Linear;

#[test]
fn linear_forward_and_backward() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let weight = Tensor::var(&[[1f32, 2.], [3., 4.], [5., 6.]], Device::CPU)?;
    let layer = Linear::new(weight.clone());
    let input = Tensor::var(&[[1f32, 2., 3.], [4., 5., 6.]], Device::CPU)?;
    let output = layer.forward(&input)?;
    let expected = vec![vec![22f32, 28.], vec![49., 64.]];
    assert_eq!(output.to_vector_rank_two::<f32>()?, expected);

    let grads = output.backward()?;
    let gw = grads.get(&weight.id()).unwrap();
    let gx = grads.get(&input.id()).unwrap();
    assert_eq!(
        gw.to_vector_rank_two::<f32>()?,
        vec![vec![5., 5.], vec![7., 7.], vec![9., 9.],]
    );
    assert_eq!(
        gx.to_vector_rank_two::<f32>()?,
        vec![vec![3., 7., 11.], vec![3., 7., 11.],]
    );
    Ok(())
}

use phantom_core::{DType, Device, Result, Tensor};

#[test]
fn construct() -> Result<()> {
    let tensor = Tensor::zeros(&[2, 3], DType::F32, &Device::CPU)?;

    let rank = tensor.rank();
    assert!(rank == 2);

    let (dim_zero, dim_one) = tensor.shape().rank_two()?;

    assert_eq!(dim_zero, 2);
    assert_eq!(dim_one, 3);

    Ok(())
}

#[test]
fn zeros() -> Result<()> {
    let tensor = Tensor::zeros((5, 2), DType::F32, &Device::CPU)?;
    let (dim_one, dim_two) = tensor.shape().rank_two()?;

    assert_eq!(dim_one, 5);
    assert_eq!(dim_two, 2);

    let data = vec![vec![0f32; 2]; 5];
    let content: Vec<Vec<f32>> = tensor.to_vector_rank_two()?;
    assert_eq!(content, data);

    Ok(())
}

#[test]
fn ones() -> Result<()> {
    let tensor = Tensor::ones((5, 2), DType::F32, &Device::CPU)?;
    let (dim_one, dim_two) = tensor.shape().rank_two()?;

    assert_eq!(dim_one, 5);
    assert_eq!(dim_two, 2);

    let data = vec![vec![1f32; 2]; 5];
    let content: Vec<Vec<f32>> = tensor.to_vector_rank_two()?;
    assert_eq!(content, data);

    Ok(())
}

#[test]
fn rank_one() -> Result<()> {
    let data = &[1f32, 2f32, 3f32, 4f32, 5f32, 6f32];
    let tensor = Tensor::new(data, &Device::CPU)?;
    let dims = tensor.shape().rank_one()?;

    assert_eq!(dims, 6);

    let content: Vec<f32> = tensor.to_vector_rank_one()?;
    assert_eq!(content, data);

    Ok(())
}

#[test]
fn rank_two() -> Result<()> {
    let data = &[
        [1f32, 2f32, 3f32, 4f32, 5f32, 6f32],
        [7f32, 8f32, 9f32, 10f32, 11f32, 12f32],
    ];
    let tensor = Tensor::new(data, &Device::CPU)?;
    let dims = tensor.shape().rank_two()?;

    assert_eq!(dims, (2, 6));

    let content: Vec<Vec<f32>> = tensor.to_vector_rank_two()?;
    assert_eq!(content, data);

    Ok(())
}

#[test]
fn add_rank_one() -> Result<()> {
    let a = Tensor::zeros(&[6], DType::F32, &Device::CPU)?;
    let b = Tensor::ones(&[6], DType::F32, &Device::CPU)?;

    let c = Tensor::add(&a, &b)?;

    let dim_zero = c.shape().rank_one()?;

    assert_eq!(dim_zero, 6);

    let content: Vec<f32> = c.to_vector_rank_one()?;
    assert_eq!(content, vec![1f32; 6]);

    let data = &[1f32, 2f32, 3f32, 4f32, 5f32, 6f32];
    let a = Tensor::ones(&[6], DType::F32, &Device::CPU)?;
    let b = Tensor::new(data, &Device::CPU)?;

    let c = (&a + &b)?;
    let content: Vec<f32> = c.to_vector_rank_one()?;

    let expected = vec![2f32, 3f32, 4f32, 5f32, 6f32, 7f32];

    assert_eq!(content, expected);

    Ok(())
}

#[test]
fn add_rank_two() -> Result<()> {
    let a = Tensor::zeros(&[2, 3], DType::F32, &Device::CPU)?;
    let b = Tensor::ones(&[2, 3], DType::F32, &Device::CPU)?;

    let c = Tensor::add(&a, &b)?;

    let (dim_zero, dim_one) = c.shape().rank_two()?;

    assert_eq!(dim_zero, 2);
    assert_eq!(dim_one, 3);

    let content: Vec<Vec<f32>> = c.to_vector_rank_two()?;
    assert_eq!(content, vec![vec![1f32; 3]; 2]);

    let data = &[
        [1f32, 2f32, 3f32, 4f32, 5f32, 6f32],
        [7f32, 8f32, 9f32, 10f32, 11f32, 12f32],
    ];
    let a = Tensor::ones(&[2, 6], DType::F32, &Device::CPU)?;
    let b = Tensor::new(data, &Device::CPU)?;

    let c = (&a + &b)?;
    let content: Vec<Vec<f32>> = c.to_vector_rank_two()?;

    let expected = vec![
        vec![2f32, 3f32, 4f32, 5f32, 6f32, 7f32],
        vec![8f32, 9f32, 10f32, 11f32, 12f32, 13f32],
    ];

    assert_eq!(content, expected);

    Ok(())
}

#[test]
fn mul_rank_one() -> Result<()> {
    let a = Tensor::zeros(&[6], DType::F32, &Device::CPU)?;
    let b = Tensor::ones(&[6], DType::F32, &Device::CPU)?;

    let c = Tensor::mul(&a, &b)?;

    let dim_zero = c.shape().rank_one()?;

    assert_eq!(dim_zero, 6);

    let content: Vec<f32> = c.to_vector_rank_one()?;
    assert_eq!(content, vec![0f32; 6]);

    let data = &[1f32, 2f32, 3f32, 4f32, 5f32, 6f32];
    let a = Tensor::ones(&[6], DType::F32, &Device::CPU)?;
    let b = Tensor::new(data, &Device::CPU)?;

    let c = (&a * &b)?;
    let content: Vec<f32> = c.to_vector_rank_one()?;

    let expected = vec![1f32, 2f32, 3f32, 4f32, 5f32, 6f32];

    assert_eq!(content, expected);

    Ok(())
}

#[test]
fn mul_rank_two() -> Result<()> {
    let a = Tensor::zeros(&[2, 3], DType::F32, &Device::CPU)?;
    let b = Tensor::ones(&[2, 3], DType::F32, &Device::CPU)?;

    let c = Tensor::mul(&a, &b)?;

    let (dim_zero, dim_one) = c.shape().rank_two()?;

    assert_eq!(dim_zero, 2);
    assert_eq!(dim_one, 3);

    let content: Vec<Vec<f32>> = c.to_vector_rank_two()?;
    assert_eq!(content, vec![vec![0f32; 3]; 2]);

    let data = &[
        [1f32, 2f32, 3f32, 4f32, 5f32, 6f32],
        [7f32, 8f32, 9f32, 10f32, 11f32, 12f32],
    ];
    let a = Tensor::ones(&[2, 6], DType::F32, &Device::CPU)?;
    let b = Tensor::new(data, &Device::CPU)?;

    let c = (&a * &b)?;
    let content: Vec<Vec<f32>> = c.to_vector_rank_two()?;

    let expected = vec![
        vec![1f32, 2f32, 3f32, 4f32, 5f32, 6f32],
        vec![7f32, 8f32, 9f32, 10f32, 11f32, 12f32],
    ];

    assert_eq!(content, expected);

    Ok(())
}

#[test]
fn binary_chaining() -> Result<()> {
    let data_a = &[[3f32, 1., 4., 1., 5.], [2., 1., 7., 8., 2.]];
    let a = Tensor::new(data_a, &Device::CPU)?;

    let data_b = &[[5f32, 5., 5., 5., 5.], [2., 1., 7., 8., 2.]];
    let b = Tensor::new(data_b, &Device::CPU)?;

    let c = (&a + (&a * &a)? / (&a + &b))?;
    let dims = a.shape().rank_two()?;

    assert_eq!(dims, (2, 5));

    let content: Vec<Vec<f32>> = c.to_vector_rank_two()?;
    assert_eq!(content[0], [4.125, 1.1666666, 5.7777777, 1.1666666, 7.5]);
    assert_eq!(content[1], [3.0, 1.5, 10.5, 12.0, 3.0]);

    let d = (&c - &c)?;
    let content: Vec<Vec<f32>> = d.to_vector_rank_two()?;
    assert_eq!(content[0], [0., 0., 0., 0., 0.]);

    Ok(())
}

#[test]
fn matmul() -> Result<()> {
    let device = &Device::CPU;
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let a = Tensor::from_slice(&data, (2, 2), device)?;
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let b = Tensor::from_slice(&data, (2, 2), device)?;

    let c = a.matmul(&b)?;
    assert_eq!(
        c.to_vector_rank_two::<f32>()?,
        &[[7.0f32, 10.0], [15.0, 22.0]]
    );

    let data = vec![1.0f32, 2.0];
    let a = Tensor::from_slice(&data, (2, 1), device)?;
    let data = vec![3.0f32, 4.0];
    let b = Tensor::from_slice(&data, (1, 2), device)?;
    let c = a.matmul(&b)?;
    assert_eq!(c.to_vector_rank_two::<f32>()?, &[&[3.0, 4.0], &[6.0, 8.0]]);

    let data: Vec<_> = (0..6).map(|i| i as f32).collect();
    let a = Tensor::from_slice(&data, (2, 3), device)?;
    let data: Vec<_> = (0..6).map(|i| (i + 2) as f32).collect();
    let b = Tensor::from_slice(&data, (3, 2), device)?;
    let c = a.matmul(&b)?;
    assert_eq!(c.to_vector_rank_two::<f32>()?, &[&[16., 19.], &[52., 64.]]);

    // TODO: test matmul with broadcasting
    // TODO: tests with higher ranks
    // TODO: tests on contigious transposed tensors

    Ok(())
}

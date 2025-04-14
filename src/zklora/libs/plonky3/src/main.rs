use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;

/// Performs matrix multiplication between two matrices of u16 elements.
///
/// # Arguments
/// * `a` - First matrix (m x n)
/// * `b` - Second matrix (n x p)
///
/// # Returns
/// A new matrix (m x p) containing the result of the multiplication
///
/// # Panics
/// Panics if the number of columns in `a` does not match the number of rows in `b`
fn matrix_multiply(a: &RowMajorMatrix<u16>, b: &RowMajorMatrix<u16>) -> RowMajorMatrix<u16> {
    assert_eq!(
        a.width(),
        b.height(),
        "Matrix dimensions must be compatible for multiplication"
    );

    let mut result = vec![0; a.height() * b.width()];

    for i in 0..a.height() {
        for j in 0..b.width() {
            let mut sum = 0;
            for k in 0..a.width() {
                sum += a.get(i, k) * b.get(k, j);
            }
            result[i * b.width() + j] = sum;
        }
    }

    RowMajorMatrix::new(result, b.width())
}

fn main() {
    let matrix_a: RowMajorMatrix<u16> = RowMajorMatrix::new(vec![1, 2, 3, 4, 5, 6], 3);
    let matrix_b: RowMajorMatrix<u16> = RowMajorMatrix::new(vec![1, 2, 3, 4, 5, 6], 2);

    let result = matrix_multiply(&matrix_a, &matrix_b);
    println!("Matrix multiplication result: {:?}", result);
}

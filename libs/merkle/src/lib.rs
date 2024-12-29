use pyo3::prelude::*;

#[pyfunction]
fn add(a: i32, b: i32) -> PyResult<i32> {
    Ok(a + b)
}

#[pyfunction]
fn minus(a: i32, b: i32) -> PyResult<i32> {
    Ok(a - b)
}

#[pymodule]
fn merkle(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_function(wrap_pyfunction!(minus, m)?)?;
    Ok(())
}
#![warn(clippy::all, clippy::style, clippy::correctness)]
#![allow(unsafe_op_in_unsafe_fn, internal_features, incomplete_features)]
#![feature(
    unchecked_shifts,
    exact_div,
    likely_unlikely,
    generic_const_exprs,
    const_trait_impl,
    default_field_values
)]

pub mod avx512;
pub mod peq;
pub mod prelude;
pub mod scalar;

pub(crate) mod macros;

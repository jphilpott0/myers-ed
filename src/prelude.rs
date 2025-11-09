//! Prelude for common functions and types available in this crate.

pub use crate::avx512::single::{
    myers_ed_single_avx512, myers_ed_single_avx512_with_peq, try_myers_ed_single_avx512,
};
pub use crate::peq::SingleWordPeq;
pub use crate::scalar::single::{
    myers_ed_single_scalar, myers_ed_single_scalar_with_peq, try_myers_ed_single_scalar,
};

#![cfg(feature = "avx512")]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, doc(cfg(feature = "avx512")))]

use crate::assert_target_features;

pub mod multi;
pub mod plumbing;
pub mod single;

// Safety: the safe functions in this module using AVX-512 intrinsics are
// safe because we guarantee here that avx512f and avx512vpopcntdq are
// available when the avx512 crate feature compiles.
assert_target_features! { "avx512", "avx512f", "avx512vpopcntdq" }

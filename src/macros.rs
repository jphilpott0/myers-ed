/// Assert that all the target_features for a given crate feature are available.
/// Takes a single crate feature, then as many valid target features as required.
/// Throws a compile-time error if target features are not required. Does not run
/// with rustdoc.
///
/// # Examples:
///
/// ```no_run
/// # use myers_ed::assert_target_features;
/// // Emits a compile-time error if avx512 crate feature is enabled and
/// // either avx512f or avx512vpopcntdq target_features are absent.
/// assert_target_features! { "avx512 ", "avx512f", "avx512vpopcntdq" }
/// ```
#[macro_export]
macro_rules! assert_target_features {
    ($feature:literal, $( $tf:literal ),+ $(,)?) => {
        #[cfg(all(
            not(doc),
            feature = $feature,
            not(all($(target_feature = $tf),+))
        ))]
        #[allow(dead_code)]
        const _: () = {
            // If we don't have all the target_features required for a crate feature, then compile
            // this compile-time error.
            compile_error!(concat!(
                "Crate feature `", stringify!($feature), "` requires enabled target features: ",
                stringify!($( $tf ),+),
            ));
        };
    };
}

use anyhow::{Result, anyhow};

use crate::peq::SingleWordPeq;

/// Perform Myers algorithm to find the edit distance between `a` and `b`. Uses 64-bit words.
/// Input bytes `a` must be `<= 64` bytes. Input bytes `b` can be any length.
///
/// # Examples
///
/// ```
/// # use myers_ed::scalar::single::myers_ed_single_scalar;
/// # fn main() {
/// let d = myers_ed_single_scalar(b"ACCC", b"ACCT");
///
/// assert_eq!(d, 1);
/// # }
/// ```
pub fn myers_ed_single_scalar(a: &[u8], b: &[u8]) -> usize {
    assert!(a.len() <= 64, "Input must be <= 64 bytes");

    let peq = SingleWordPeq::from_bytes(a);

    myers_ed_single_scalar_with_peq(&peq, b)
}

pub fn try_myers_ed_single_scalar(a: &[u8], b: &[u8]) -> Result<usize> {
    if a.len() > 64 {
        return Err(anyhow!("Input must be <= 64 bytes"));
    }

    let peq = SingleWordPeq::from_bytes(a);

    Ok(myers_ed_single_scalar_with_peq(&peq, b))
}

pub fn myers_ed_single_scalar_with_peq(peq: &SingleWordPeq<u64>, b: &[u8]) -> usize {
    // Vertical positive delta bit-vector.
    let mut vp = u64::MAX;

    // Vertical negative delta bit-vector.
    let mut vn = 0_u64;

    // Update loop.
    for &x in b {
        // Get the equality mask for the current character.
        //
        // Infallible: `x as usize` \in [0, 255] and `peq.peq.len() == 256`.
        let eq = peq[x as usize];

        // Calculate diagonal zero delta bit-vector.
        let d0 = (((eq & vp).saturating_add(vp)) ^ vp) | eq;

        // Calculate horizontal delta bit-vectors.
        let mut hp = vn | !(vp | d0);
        let hn = vp & d0;

        // Calculate intermediate mask for next column's vertical delta bits.
        let xh = eq | vn;

        // Move one column right in DP matrix.
        hp = (hp << 1_u64) | 1_u64;

        // Update vertical delta bit-vectors.
        vp = (hn << 1_u64) | !(xh | hp);
        vn = hp & xh;
    }

    // Compute mask to get only real bits.
    let m = 1_u64 << (peq.len()).wrapping_sub(1);

    // Compute final edit distance.
    let vp_popcnt = (vp & m).count_ones();
    let vn_popcnt = (vn & m).count_ones();

    b.len() + vp_popcnt as usize - vn_popcnt as usize
}

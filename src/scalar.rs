use anyhow::{Result, anyhow};

use crate::peq::SingleWordPeq;

/// Perform Myers algorithm to find the edit distance between `a` and `b`. Uses `usize::BITS`-sized words.
/// Input bytes `a` must be `<= usize::BITS` bytes. Input bytes `b` can be any length.
///
/// # Examples
///
/// ```
/// # use myers_ed::scalar::myers_ed_single_scalar;
/// # fn main() {
/// let d = myers_ed_single_scalar(b"ACCC", b"ACCT");
///
/// assert_eq!(d, 1);
/// # }
/// ```
pub fn myers_ed_single_scalar(a: &[u8], b: &[u8]) -> usize {
    assert!(
        a.len() <= usize::BITS as usize,
        "Input must be <= {} bytes",
        usize::BITS
    );

    let peq = SingleWordPeq::from_bytes(a);

    myers_ed_single_scalar_with_peq(&peq, b)
}

pub fn try_myers_ed_single_scalar(a: &[u8], b: &[u8]) -> Result<usize> {
    if a.len() > usize::BITS as usize {
        return Err(anyhow!("Input must be <= {} bytes", usize::BITS));
    }

    let peq = SingleWordPeq::from_bytes(a);

    Ok(myers_ed_single_scalar_with_peq(&peq, b))
}

pub fn myers_ed_single_scalar_with_peq(peq: &SingleWordPeq<usize>, b: &[u8]) -> usize {
    // Vertical positive delta bit-vector.
    let mut vp = usize::MAX;

    // Vertical negative delta bit-vector.
    let mut vn = 0_usize;

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
        hp = (hp << 1_usize) | 1_usize;

        // Update vertical delta bit-vectors.
        vp = (hn << 1_usize) | !(xh | hp);
        vn = hp & xh;
    }

    // Compute mask to get only real bits.
    let m = 1_usize.unbounded_shl(peq.len() as u32).wrapping_sub(1);

    // Compute final edit distance.
    let vp_popcnt = (vp & m).count_ones();
    let vn_popcnt = (vn & m).count_ones();

    b.len() + vp_popcnt as usize - vn_popcnt as usize
}

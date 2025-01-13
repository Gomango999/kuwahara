--------------------------------------------------------------------------------
-- Metadata
--------------------------------------------------------------------------------
Invocation:       /bin/cg_annotate cachegrind.out.37580
Command:          ./target/release/kuwahara anisotropic --input ./examples/squirrel.png --intermediate-results
Events recorded:  Ir
Events shown:     Ir
Event sort order: Ir
Threshold:        0.1%
Annotation:       on

--------------------------------------------------------------------------------
-- Summary
--------------------------------------------------------------------------------
Ir_____________________ 

35,340,605,634 (100.0%)  PROGRAM TOTALS

--------------------------------------------------------------------------------
-- File:function summary
--------------------------------------------------------------------------------
  Ir___________________________  file:function

< 23,198,970,727 (65.6%, 65.6%)  /home/kevin/01_rust/06_kuwahara/src/anisotropic/mod.rs:
  23,174,825,800 (65.6%)           kuwahara::anisotropic::run

<  3,212,330,295  (9.1%, 74.7%)  /home/kevin/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/lib/rustlib/src/rust/library/core/src/cmp.rs:
   3,202,973,352  (9.1%)           kuwahara::anisotropic::run

<  1,989,823,891  (5.6%, 80.4%)  /home/kevin/.cargo/registry/src/index.crates.io-6f17d22bba15001f/ndarray-0.16.1/src/dimension/dimension_trait.rs:
   1,962,553,181  (5.6%)           kuwahara::anisotropic::run

<  1,381,025,110  (3.9%, 84.3%)  /home/kevin/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/lib/rustlib/src/rust/library/core/src/ops/range.rs:
   1,381,024,890  (3.9%)           kuwahara::anisotropic::run

<    988,407,673  (2.8%, 87.1%)  /home/kevin/.cargo/registry/src/index.crates.io-6f17d22bba15001f/ndarray-0.16.1/src/dimension/mod.rs:
     978,568,286  (2.8%)           kuwahara::anisotropic::run

<    912,742,353  (2.6%, 89.7%)  /home/kevin/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/lib/rustlib/src/rust/library/std/src/f64.rs:
     904,459,026  (2.6%)           kuwahara::anisotropic::run

<    867,373,596  (2.5%, 92.1%)  /home/kevin/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/lib/rustlib/src/rust/library/core/src/iter/range.rs:
     851,152,791  (2.4%)           kuwahara::anisotropic::run

<    657,721,742  (1.9%, 94.0%)  /home/kevin/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/lib/rustlib/src/rust/library/core/src/ptr/mut_ptr.rs:
     654,293,826  (1.9%)           kuwahara::anisotropic::run

<    585,596,404  (1.7%, 95.6%)  ./malloc/./malloc/malloc.c:
     160,438,795  (0.5%)           _int_malloc
     142,508,812  (0.4%)           _int_free
      64,498,281  (0.2%)           free
      58,438,938  (0.2%)           calloc
      51,978,035  (0.1%)           malloc

<    352,040,491  (1.0%, 96.6%)  /home/kevin/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/lib/rustlib/src/rust/library/core/src/option.rs:
     328,870,434  (0.9%)           kuwahara::anisotropic::run

<    165,331,313  (0.5%, 97.1%)  /home/kevin/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/lib/rustlib/src/rust/library/core/src/slice/iter/macros.rs:
     151,108,331  (0.4%)           <ndarray::iterators::Iter<A,D> as core::iter::traits::iterator::Iterator>::fold

<    104,548,675  (0.3%, 97.4%)  /home/kevin/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/lib/rustlib/src/rust/library/core/src/ptr/non_null.rs:
      89,708,107  (0.3%)           <ndarray::iterators::Iter<A,D> as core::iter::traits::iterator::Iterator>::fold

<     92,988,248  (0.3%, 97.6%)  /home/kevin/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/lib/rustlib/src/rust/library/core/src/ops/arith.rs:
      88,224,966  (0.2%)           <ndarray::iterators::Iter<A,D> as core::iter::traits::iterator::Iterator>::fold

<     39,080,386  (0.1%, 97.8%)  ./math/../sysdeps/ieee754/dbl-64/e_atan2.c:__ieee754_atan2_fma

<     38,624,940  (0.1%, 97.9%)  /home/kevin/.cargo/registry/src/index.crates.io-6f17d22bba15001f/matrixmultiply-0.3.9/src/packing.rs:matrixmultiply::packing::pack_avx2

--------------------------------------------------------------------------------
-- Function:file summary
--------------------------------------------------------------------------------
  Ir___________________________  function:file

> 33,460,226,373 (94.7%, 94.7%)  kuwahara::anisotropic::run:
  23,174,825,800 (65.6%)           /home/kevin/01_rust/06_kuwahara/src/anisotropic/mod.rs
   3,202,973,352  (9.1%)           /home/kevin/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/lib/rustlib/src/rust/library/core/src/cmp.rs
   1,962,553,181  (5.6%)           /home/kevin/.cargo/registry/src/index.crates.io-6f17d22bba15001f/ndarray-0.16.1/src/dimension/dimension_trait.rs
   1,381,024,890  (3.9%)           /home/kevin/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/lib/rustlib/src/rust/library/core/src/ops/range.rs
     978,568,286  (2.8%)           /home/kevin/.cargo/registry/src/index.crates.io-6f17d22bba15001f/ndarray-0.16.1/src/dimension/mod.rs
     904,459,026  (2.6%)           /home/kevin/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/lib/rustlib/src/rust/library/std/src/f64.rs
     851,152,791  (2.4%)           /home/kevin/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/lib/rustlib/src/rust/library/core/src/iter/range.rs
     654,293,826  (1.9%)           /home/kevin/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/lib/rustlib/src/rust/library/core/src/ptr/mut_ptr.rs
     328,870,434  (0.9%)           /home/kevin/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/lib/rustlib/src/rust/library/core/src/option.rs

>    354,247,904  (1.0%, 95.7%)  <ndarray::iterators::Iter<A,D> as core::iter::traits::iterator::Iterator>::fold:
     151,108,331  (0.4%)           /home/kevin/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/lib/rustlib/src/rust/library/core/src/slice/iter/macros.rs
      89,708,107  (0.3%)           /home/kevin/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/lib/rustlib/src/rust/library/core/src/ptr/non_null.rs
      88,224,966  (0.2%)           /home/kevin/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/lib/rustlib/src/rust/library/core/src/ops/arith.rs

>    160,438,795  (0.5%, 96.1%)  _int_malloc:./malloc/./malloc/malloc.c

>    142,508,812  (0.4%, 96.5%)  _int_free:./malloc/./malloc/malloc.c

>     71,408,778  (0.2%, 96.7%)  free:
      64,498,281  (0.2%)           ./malloc/./malloc/malloc.c

>     61,065,497  (0.2%, 96.9%)  calloc:
      58,438,938  (0.2%)           ./malloc/./malloc/malloc.c

>     59,246,730  (0.2%, 97.1%)  matrixmultiply::packing::pack_avx2:
      38,624,940  (0.1%)           /home/kevin/.cargo/registry/src/index.crates.io-6f17d22bba15001f/matrixmultiply-0.3.9/src/packing.rs

>     52,635,040  (0.1%, 97.2%)  malloc:
      51,978,035  (0.1%)           ./malloc/./malloc/malloc.c

>     42,407,736  (0.1%, 97.4%)  __ieee754_atan2_fma:
      39,080,386  (0.1%)           ./math/../sysdeps/ieee754/dbl-64/e_atan2.c

--------------------------------------------------------------------------------
-- Annotated source file: ./malloc/./malloc/malloc.c
--------------------------------------------------------------------------------
Unannotated because one or more of these original files are unreadable:
- ./malloc/./malloc/malloc.c

--------------------------------------------------------------------------------
-- Annotated source file: ./math/../sysdeps/ieee754/dbl-64/e_atan2.c
--------------------------------------------------------------------------------
Unannotated because one or more of these original files are unreadable:
- ./math/../sysdeps/ieee754/dbl-64/e_atan2.c

--------------------------------------------------------------------------------
-- Annotated source file: /home/kevin/.cargo/registry/src/index.crates.io-6f17d22bba15001f/matrixmultiply-0.3.9/src/packing.rs
--------------------------------------------------------------------------------
Ir_______________ 

12,111,210 (0.0%)  <unknown (line 0)>

-- line 33 ----------------------------------------
         .         {
         .             pack_impl::<MR, T>(kc, mc, pack, a, rsa, csa)
         .         }
         .         
         .         /// Specialized for AVX2
         .         /// Safety: Requires AVX2
         .         #[cfg(any(target_arch="x86", target_arch="x86_64"))]
         .         #[target_feature(enable="avx2")]
 4,582,620 (0.0%)  pub(crate) unsafe fn pack_avx2<MR, T>(kc: usize, mc: usize, pack: &mut [T],
         .                                              a: *const T, rsa: isize, csa: isize)
         .             where T: Element,
         .                   MR: ConstNum,
         .         {
         .             pack_impl::<MR, T>(kc, mc, pack, a, rsa, csa)
 5,237,280 (0.0%)  }
         .         
         .         /// Pack implementation, see pack above for docs.
         .         ///
         .         /// Uses inline(always) so that it can be instantiated for different target features.
         .         #[inline(always)]
         .         unsafe fn pack_impl<MR, T>(kc: usize, mc: usize, pack: &mut [T],
         .                                    a: *const T, rsa: isize, csa: isize)
         .             where T: Element,
         .                   MR: ConstNum,
         .         {
         .             let pack = pack.as_mut_ptr();
         .             let mr = MR::VALUE;
         .             let mut p = 0; // offset into pack
         .         
 2,291,310 (0.0%)      if rsa == 1 {
         .                 // if the matrix is contiguous in the same direction we are packing,
         .                 // copy a kernel row at a time.
         .                 for ir in 0..mc/mr {
         .                     let row_offset = ir * mr;
         .                     for j in 0..kc {
         .                         let a_row = a.stride_offset(rsa, row_offset)
         .                                      .stride_offset(csa, j);
         .                         copy_nonoverlapping(a_row, pack.add(p), mr);
-- line 70 ----------------------------------------
-- line 85 ----------------------------------------
         .                     }
         .                 }
         .             }
         .         
         .             let zero = <_>::zero();
         .         
         .             // Pad with zeros to multiple of kernel size (uneven mc)
         .             let rest = mc % mr;
 5,237,280 (0.0%)      if rest > 0 {
 2,618,640 (0.0%)          let row_offset = (mc/mr) * mr;
         .                 for j in 0..kc {
         .                     for i in 0..mr {
 6,546,600 (0.0%)                  if i < rest {
         .                             let a_elt = a.stride_offset(rsa, i + row_offset)
         .                                          .stride_offset(csa, j);
         .                             copy_nonoverlapping(a_elt, pack.add(p), 1);
         .                         } else {
         .                             *pack.add(p) = zero;
         .                         }
         .                         p += 1;
         .                     }
-- line 105 ----------------------------------------

--------------------------------------------------------------------------------
-- Annotated source file: /home/kevin/.cargo/registry/src/index.crates.io-6f17d22bba15001f/ndarray-0.16.1/src/dimension/dimension_trait.rs
--------------------------------------------------------------------------------
Ir__________________ 

  337,909,259 (1.0%)  <unknown (line 0)>

-- line 126 ----------------------------------------
            .             /// the array is empty, the strides are all zeros.
            .             #[doc(hidden)]
            .             fn default_strides(&self) -> Self
            .             {
            .                 // Compute default array strides
            .                 // Shape (a, b, c) => Give strides (b * c, c, 1)
            .                 let mut strides = Self::zeros(self.ndim());
            .                 // For empty arrays, use all zero strides.
           41 (0.0%)          if self.slice().iter().all(|&d| d != 0) {
            .                     let mut it = strides.slice_mut().iter_mut().rev();
            .                     // Set first element to 1
            .                     if let Some(rs) = it.next() {
            .                         *rs = 1;
            .                     }
            .                     let mut cum_prod = 1;
            .                     for (rs, dim) in it.zip(self.slice().iter().rev()) {
           16 (0.0%)                  cum_prod *= *dim;
            .                         *rs = cum_prod;
            .                     }
            .                 }
           12 (0.0%)          strides
            .             }
            .         
            .             /// Returns the strides for a Fortran layout array with the given shape.
            .             ///
            .             /// If the array is non-empty, the strides result in contiguous layout; if
            .             /// the array is empty, the strides are all zeros.
            .             #[doc(hidden)]
            .             fn fortran_strides(&self) -> Self
-- line 154 ----------------------------------------
-- line 248 ----------------------------------------
            .             #[doc(hidden)]
            .             fn strides_equivalent<D>(&self, strides1: &Self, strides2: &D) -> bool
            .             where D: Dimension
            .             {
            .                 let shape_ndim = self.ndim();
            .                 shape_ndim == strides1.ndim()
            .                     && shape_ndim == strides2.ndim()
            .                     && izip!(self.slice(), strides1.slice(), strides2.slice())
      657,297 (0.0%)                  .all(|(&d, &s1, &s2)| d <= 1 || s1 as isize == s2 as isize)
            .             }
            .         
            .             #[doc(hidden)]
            .             /// Return stride offset for index.
            .             fn stride_offset(index: &Self, strides: &Self) -> isize
            .             {
            .                 let mut offset = 0;
            .                 for (&i, &s) in izip!(index.slice(), strides.slice()) {
-- line 264 ----------------------------------------
-- line 289 ----------------------------------------
            .             {
            .                 let nd = self.ndim();
            .                 self.slice_mut()[nd - 1] = i;
            .             }
            .         
            .             #[doc(hidden)]
            .             fn is_contiguous(dim: &Self, strides: &Self) -> bool
            .             {
      328,354 (0.0%)          let defaults = dim.default_strides();
      656,872 (0.0%)          if strides.equal(&defaults) {
            .                     return true;
            .                 }
            .                 if dim.ndim() == 1 {
            .                     // fast case for ndim == 1:
            .                     // Either we have length <= 1, then stride is arbitrary,
            .                     // or we have stride == 1 or stride == -1, but +1 case is already handled above.
            .                     dim[0] <= 1 || strides[0] as isize == -1
            .                 } else {
-- line 306 ----------------------------------------
-- line 313 ----------------------------------------
            .                         // a dimension of length 1 can have unequal strides
            .                         if dim_slice[i] != 1 && (strides[i] as isize).unsigned_abs() != cstride {
            .                             return false;
            .                         }
            .                         cstride *= dim_slice[i];
            .                     }
            .                     true
            .                 }
            6 (0.0%)      }
            .         
            .             /// Return the axis ordering corresponding to the fastest variation
            .             /// (in ascending order).
            .             ///
            .             /// Assumes that no stride value appears twice.
            .             #[doc(hidden)]
            .             fn _fastest_varying_stride_order(&self) -> Self
            .             {
-- line 329 ----------------------------------------
-- line 404 ----------------------------------------
            .         macro_rules! impl_insert_axis_array(
            .             ($n:expr) => (
            .                 #[inline]
            .                 fn insert_axis(&self, axis: Axis) -> Self::Larger {
            .                     debug_assert!(axis.index() <= $n);
            .                     let mut out = [1; $n + 1];
            .                     out[0..axis.index()].copy_from_slice(&self.slice()[0..axis.index()]);
            .                     out[axis.index()+1..=$n].copy_from_slice(&self.slice()[axis.index()..$n]);
           60 (0.0%)              Dim(out)
            .                 }
            .             );
            .         );
            .         
            .         impl Dimension for Dim<[Ix; 0]>
            .         {
            .             const NDIM: Option<usize> = Some(0);
            .             type Pattern = ();
-- line 420 ----------------------------------------
-- line 505 ----------------------------------------
            .                 } else {
            .                     None
            .                 }
            .             }
            .         
            .             #[inline]
            .             fn equal(&self, rhs: &Self) -> bool
            .             {
           42 (0.0%)          get!(self, 0) == get!(rhs, 0)
            .             }
            .         
            .             #[inline]
            .             fn size(&self) -> usize
            .             {
            .                 get!(self, 0)
            .             }
            .             #[inline]
-- line 521 ----------------------------------------
-- line 522 ----------------------------------------
            .             fn size_checked(&self) -> Option<usize>
            .             {
            .                 Some(get!(self, 0))
            .             }
            .         
            .             #[inline]
            .             fn default_strides(&self) -> Self
            .             {
    1,477,305 (0.0%)          if get!(self, 0) == 0 {
            .                     Ix1(0)
            .                 } else {
            .                     Ix1(1)
            .                 }
            .             }
            .         
            .             #[inline]
            .             fn _fastest_varying_stride_order(&self) -> Self
-- line 538 ----------------------------------------
-- line 568 ----------------------------------------
            .             {
            .                 stride_offset(get!(index, 0), get!(stride, 0))
            .             }
            .         
            .             /// Return stride offset for this dimension and index.
            .             #[inline]
            .             fn stride_offset_checked(&self, stride: &Self, index: &Self) -> Option<isize>
            .             {
  327,721,418 (0.9%)          if get!(index, 0) < get!(self, 0) {
            .                     Some(stride_offset(get!(index, 0), get!(stride, 0)))
            .                 } else {
            .                     None
            .                 }
            .             }
            .             impl_insert_axis_array!(1);
            .             #[inline]
            .             fn try_remove_axis(&self, axis: Axis) -> Self::Smaller
-- line 584 ----------------------------------------
-- line 631 ----------------------------------------
            .             }
            .             #[inline]
            .             fn next_for(&self, index: Self) -> Option<Self>
            .             {
            .                 let mut i = get!(&index, 0);
            .                 let mut j = get!(&index, 1);
            .                 let imax = get!(self, 0);
            .                 let jmax = get!(self, 1);
          689 (0.0%)          j += 1;
        1,960 (0.0%)          if j >= jmax {
            .                     j = 0;
        4,005 (0.0%)              i += 1;
        4,617 (0.0%)              if i >= imax {
            .                         return None;
            .                     }
            .                 }
            .                 Some(Ix2(i, j))
            .             }
            .         
            .             #[inline]
            .             fn equal(&self, rhs: &Self) -> bool
            .             {
      820,828 (0.0%)          get!(self, 0) == get!(rhs, 0) && get!(self, 1) == get!(rhs, 1)
            .             }
            .         
            .             #[inline]
            .             fn size(&self) -> usize
            .             {
      984,127 (0.0%)          get!(self, 0) * get!(self, 1)
            .             }
            .         
            .             #[inline]
            .             fn size_checked(&self) -> Option<usize>
            .             {
            .                 let m = get!(self, 0);
            .                 let n = get!(self, 1);
            .                 m.checked_mul(n)
-- line 667 ----------------------------------------
-- line 679 ----------------------------------------
            .                 getm!(self, 1) = i;
            .             }
            .         
            .             #[inline]
            .             fn default_strides(&self) -> Self
            .             {
            .                 let m = get!(self, 0);
            .                 let n = get!(self, 1);
    3,771,805 (0.0%)          if m == 0 || n == 0 {
            .                     Ix2(0, 0)
            .                 } else {
            .                     Ix2(n, 1)
            .                 }
            .             }
            .             #[inline]
            .             fn fortran_strides(&self) -> Self
            .             {
-- line 695 ----------------------------------------
-- line 724 ----------------------------------------
            .                 }
            .             }
            .         
            .             #[inline]
            .             fn first_index(&self) -> Option<Self>
            .             {
            .                 let m = get!(self, 0);
            .                 let n = get!(self, 1);
           85 (0.0%)          if m != 0 && n != 0 {
           51 (0.0%)              Some(Ix2(0, 0))
            .                 } else {
            .                     None
            .                 }
            .             }
            .         
            .             /// Self is an index, return the stride offset
            .             #[inline(always)]
            .             fn stride_offset(index: &Self, strides: &Self) -> isize
-- line 741 ----------------------------------------
-- line 752 ----------------------------------------
            .             fn stride_offset_checked(&self, strides: &Self, index: &Self) -> Option<isize>
            .             {
            .                 let m = get!(self, 0);
            .                 let n = get!(self, 1);
            .                 let i = get!(index, 0);
            .                 let j = get!(index, 1);
            .                 let s = get!(strides, 0);
            .                 let t = get!(strides, 1);
1,307,602,772 (3.7%)          if i < m && j < n {
            .                     Some(stride_offset(i, s) + stride_offset(j, t))
            .                 } else {
            .                     None
            .                 }
            .             }
            .             impl_insert_axis_array!(2);
            .             #[inline]
            .             fn try_remove_axis(&self, axis: Axis) -> Self::Smaller
-- line 768 ----------------------------------------
-- line 800 ----------------------------------------
            .             }
            .         
            .             #[inline]
            .             fn size(&self) -> usize
            .             {
            .                 let m = get!(self, 0);
            .                 let n = get!(self, 1);
            .                 let o = get!(self, 2);
          136 (0.0%)          m * n * o
            .             }
            .         
            .             #[inline]
            .             fn zeros(ndim: usize) -> Self
            .             {
            .                 assert_eq!(ndim, 3);
            .                 Self::default()
            .             }
-- line 816 ----------------------------------------
-- line 819 ----------------------------------------
            .             fn next_for(&self, index: Self) -> Option<Self>
            .             {
            .                 let mut i = get!(&index, 0);
            .                 let mut j = get!(&index, 1);
            .                 let mut k = get!(&index, 2);
            .                 let imax = get!(self, 0);
            .                 let jmax = get!(self, 1);
            .                 let kmax = get!(self, 2);
    1,477,305 (0.0%)          k += 1;
    3,939,477 (0.0%)          if k == kmax {
            .                     k = 0;
      492,435 (0.0%)              j += 1;
      984,870 (0.0%)              if j == jmax {
            .                         j = 0;
        1,059 (0.0%)                  i += 1;
        2,118 (0.0%)                  if i == imax {
            .                             return None;
            .                         }
            .                     }
            .                 }
            .                 Some(Ix3(i, j, k))
            .             }
            .         
            .             /// Self is an index, return the stride offset
-- line 842 ----------------------------------------
-- line 860 ----------------------------------------
            .                 let n = get!(self, 1);
            .                 let l = get!(self, 2);
            .                 let i = get!(index, 0);
            .                 let j = get!(index, 1);
            .                 let k = get!(index, 2);
            .                 let s = get!(strides, 0);
            .                 let t = get!(strides, 1);
            .                 let u = get!(strides, 2);
      984,870 (0.0%)          if i < m && j < n && k < l {
            .                     Some(stride_offset(i, s) + stride_offset(j, t) + stride_offset(k, u))
            .                 } else {
            .                     None
            .                 }
            .             }
            .         
            .             #[inline]
            .             fn _fastest_varying_stride_order(&self) -> Self
-- line 876 ----------------------------------------

--------------------------------------------------------------------------------
-- Annotated source file: /home/kevin/.cargo/registry/src/index.crates.io-6f17d22bba15001f/ndarray-0.16.1/src/dimension/mod.rs
--------------------------------------------------------------------------------
Ir________________ 

    819,559 (0.0%)  <unknown (line 0)>

-- line 42 ----------------------------------------
          .         mod remove_axis;
          .         pub(crate) mod reshape;
          .         mod sequence;
          .         
          .         /// Calculate offset from `Ix` stride converting sign properly
          .         #[inline(always)]
          .         pub fn stride_offset(n: Ix, stride: Ix) -> isize
          .         {
978,568,363 (2.8%)      (n as isize) * (stride as Ixs)
          .         }
          .         
          .         /// Check whether the given `dim` and `stride` lead to overlapping indices
          .         ///
          .         /// There is overlap if, when iterating through the dimensions in order of
          .         /// increasing stride, the current stride is less than or equal to the maximum
          .         /// possible offset along the preceding axes. (Axes of length ≤1 are ignored.)
          .         pub(crate) fn dim_stride_overlap<D: Dimension>(dim: &D, strides: &D) -> bool
-- line 58 ----------------------------------------
-- line 85 ----------------------------------------
          .         /// are met to construct an array from the data buffer, `dim`, and `strides`.
          .         /// (The data buffer being a slice or `Vec` guarantees that it contains no more
          .         /// than `isize::MAX` bytes.)
          .         pub fn size_of_shape_checked<D: Dimension>(dim: &D) -> Result<usize, ShapeError>
          .         {
          .             let size_nonzero = dim
          .                 .slice()
          .                 .iter()
        235 (0.0%)          .filter(|&&d| d != 0)
          .                 .try_fold(1usize, |acc, &d| acc.checked_mul(d))
          .                 .ok_or_else(|| from_kind(ErrorKind::Overflow))?;
        158 (0.0%)      if size_nonzero > isize::MAX as usize {
          .                 Err(from_kind(ErrorKind::Overflow))
          .             } else {
          .                 Ok(dim.size())
          .             }
          .         }
          .         
          .         /// Select how aliasing is checked
          .         ///
-- line 104 ----------------------------------------
-- line 134 ----------------------------------------
          .         ///    zero strides for empty arrays, this ensures that for empty arrays the
          .         ///    difference between the least address and greatest address accessible by
          .         ///    moving along all axes is ≤ the length of the slice.)
          .         ///
          .         /// Note that since slices cannot contain more than `isize::MAX` bytes,
          .         /// conditions 1 and 2 are sufficient to guarantee that the offset in units of
          .         /// `A` and in units of bytes between the least address and greatest address
          .         /// accessible by moving along all axes does not exceed `isize::MAX`.
          3 (0.0%)  pub(crate) fn can_index_slice_with_strides<A, D: Dimension>(
          .             data: &[A], dim: &D, strides: &Strides<D>, mode: CanIndexCheckMode,
          .         ) -> Result<(), ShapeError>
          .         {
         91 (0.0%)      if let Strides::Custom(strides) = strides {
          .                 can_index_slice(data, dim, strides, mode)
          .             } else {
          .                 // contiguous shapes: never aliasing, mode does not matter
          .                 can_index_slice_not_custom(data.len(), dim)
          .             }
         40 (0.0%)  }
          .         
          .         pub(crate) fn can_index_slice_not_custom<D: Dimension>(data_len: usize, dim: &D) -> Result<(), ShapeError>
          .         {
          .             // Condition 1.
          .             let len = size_of_shape_checked(dim)?;
          .             // Condition 2.
          6 (0.0%)      if len > data_len {
          .                 return Err(from_kind(ErrorKind::OutOfBounds));
          .             }
          .             Ok(())
          .         }
          .         
          .         /// Returns the absolute difference in units of `A` between least and greatest
          .         /// address accessible by moving along all axes.
          .         ///
-- line 167 ----------------------------------------
-- line 199 ----------------------------------------
          .                 .try_fold(0usize, |acc, (&d, &s)| {
          .                     let s = s as isize;
          .                     // Calculate maximum possible absolute movement along this axis.
          .                     let off = d.saturating_sub(1).checked_mul(s.unsigned_abs())?;
          .                     acc.checked_add(off)
          .                 })
          .                 .ok_or_else(|| from_kind(ErrorKind::Overflow))?;
          .             // Condition 2a.
         17 (0.0%)      if max_offset > isize::MAX as usize {
          .                 return Err(from_kind(ErrorKind::Overflow));
          .             }
          .         
          .             // Determine absolute difference in units of bytes between least and
          .             // greatest address accessible by moving along all axes
          .             let max_offset_bytes = max_offset
          .                 .checked_mul(elem_size)
          .                 .ok_or_else(|| from_kind(ErrorKind::Overflow))?;
-- line 215 ----------------------------------------
-- line 264 ----------------------------------------
          .             can_index_slice_impl(max_offset, data.len(), dim, strides, mode)
          .         }
          .         
          .         fn can_index_slice_impl<D: Dimension>(
          .             max_offset: usize, data_len: usize, dim: &D, strides: &D, mode: CanIndexCheckMode,
          .         ) -> Result<(), ShapeError>
          .         {
          .             // Check condition 3.
         34 (0.0%)      let is_empty = dim.slice().iter().any(|&d| d == 0);
          .             if is_empty && max_offset > data_len {
          .                 return Err(from_kind(ErrorKind::OutOfBounds));
          .             }
         34 (0.0%)      if !is_empty && max_offset >= data_len {
          .                 return Err(from_kind(ErrorKind::OutOfBounds));
          .             }
          .         
          .             // Check condition 4.
         34 (0.0%)      if !is_empty && mode != CanIndexCheckMode::ReadOnly && dim_stride_overlap(dim, strides) {
          .                 return Err(from_kind(ErrorKind::Unsupported));
          .             }
          .         
          .             Ok(())
          .         }
          .         
          .         /// Stride offset checked general version (slices)
          .         #[inline]
-- line 289 ----------------------------------------
-- line 339 ----------------------------------------
          .             fn axis(&self, axis: Axis) -> Ix
          .             {
          .                 self[axis.index()]
          .             }
          .         
          .             #[inline]
          .             fn set_axis(&mut self, axis: Axis, value: Ix)
          .             {
          4 (0.0%)          self[axis.index()] = value;
          .             }
          .         }
          .         
          .         impl DimensionExt for [Ix]
          .         {
          .             #[inline]
          .             fn axis(&self, axis: Axis) -> Ix
          .             {
-- line 355 ----------------------------------------
-- line 369 ----------------------------------------
          .         /// **Panics** if `index` is larger than the size of the axis
          .         #[track_caller]
          .         // FIXME: Move to Dimension trait
          .         pub fn do_collapse_axis<D: Dimension>(dims: &mut D, strides: &D, axis: usize, index: usize) -> isize
          .         {
          .             let dim = dims.slice()[axis];
          .             let stride = strides.slice()[axis];
          .             ndassert!(
        231 (0.0%)          index < dim,
          .                 "collapse_axis: Index {} must be less than axis length {} for \
          .                  array with shape {:?}",
          .                 index,
          .                 dim,
          .                 *dims
          .             );
          .             dims.slice_mut()[axis] = 1;
          .             stride_offset(index, stride)
          .         }
          .         
          .         /// Compute the equivalent unsigned index given the axis length and signed index.
          .         #[inline]
          .         pub fn abs_index(len: Ix, index: Ixs) -> Ix
          .         {
        136 (0.0%)      if index < 0 {
          .                 len - (-index as Ix)
          .             } else {
          .                 index as Ix
          .             }
          .         }
          .         
          .         /// Determines nonnegative start and end indices, and performs sanity checks.
          .         ///
          .         /// The return value is (start, end, step).
          .         ///
          .         /// **Panics** if stride is 0 or if any index is out of bounds.
          .         #[track_caller]
          .         fn to_abs_slice(axis_len: usize, slice: Slice) -> (usize, usize, isize)
          .         {
         34 (0.0%)      let Slice { start, end, step } = slice;
          .             let start = abs_index(axis_len, start);
          .             let mut end = abs_index(axis_len, end.unwrap_or(axis_len as isize));
         68 (0.0%)      if end < start {
          .                 end = start;
          .             }
          .             ndassert!(
         68 (0.0%)          start <= axis_len,
          .                 "Slice begin {} is past end of axis of length {}",
          .                 start,
          .                 axis_len,
          .             );
          .             ndassert!(
         68 (0.0%)          end <= axis_len,
          .                 "Slice end {} is past end of axis of length {}",
          .                 end,
          .                 axis_len,
          .             );
         68 (0.0%)      ndassert!(step != 0, "Slice stride must not be zero");
          .             (start, end, step)
          .         }
          .         
          .         /// This function computes the offset from the lowest address element to the
          .         /// logically first element.
          .         pub fn offset_from_low_addr_ptr_to_logical_ptr<D: Dimension>(dim: &D, strides: &D) -> usize
          .         {
          .             let offset = izip!(dim.slice(), strides.slice()).fold(0, |_offset, (&d, &s)| {
          .                 let s = s as isize;
  9,016,429 (0.0%)          if s < 0 && d > 1 {
          .                     _offset - s * (d as isize - 1)
          .                 } else {
          .                     _offset
          .                 }
          .             });
          .             debug_assert!(offset >= 0);
          .             offset as usize
          .         }
          .         
          .         /// Modify dimension, stride and return data pointer offset
          .         ///
          .         /// **Panics** if stride is 0 or if any index is out of bounds.
          .         #[track_caller]
         34 (0.0%)  pub fn do_slice(dim: &mut usize, stride: &mut usize, slice: Slice) -> isize
          .         {
         34 (0.0%)      let (start, end, step) = to_abs_slice(*dim, slice);
          .         
          .             let m = end - start;
         34 (0.0%)      let s = (*stride) as isize;
          .         
          .             // Compute data pointer offset.
        102 (0.0%)      let offset = if m == 0 {
          .                 // In this case, the resulting array is empty, so we *can* avoid performing a nonzero
          .                 // offset.
          .                 //
          .                 // In two special cases (which are the true reason for this `m == 0` check), we *must* avoid
          .                 // the nonzero offset corresponding to the general case.
          .                 //
          .                 // * When `end == 0 && step < 0`. (These conditions imply that `m == 0` since `to_abs_slice`
          .                 //   ensures that `0 <= start <= end`.) We cannot execute `stride_offset(end - 1, *stride)`
          .                 //   because the `end - 1` would underflow.
          .                 //
          .                 // * When `start == *dim && step > 0`. (These conditions imply that `m == 0` since
          .                 //   `to_abs_slice` ensures that `start <= end <= *dim`.) We cannot use the offset returned
          .                 //   by `stride_offset(start, *stride)` because that would be past the end of the axis.
          .                 0
        102 (0.0%)      } else if step < 0 {
          .                 // When the step is negative, the new first element is `end - 1`, not `start`, since the
          .                 // direction is reversed.
          .                 stride_offset(end - 1, *stride)
          .             } else {
          .                 stride_offset(start, *stride)
          .             };
          .         
          .             // Update dimension.
          .             let abs_step = step.unsigned_abs();
        102 (0.0%)      *dim = if abs_step == 1 {
          .                 m
          .             } else {
          .                 let d = m / abs_step;
          .                 let r = m % abs_step;
          .                 d + if r > 0 { 1 } else { 0 }
          .             };
          .         
          .             // Update stride. The additional check is necessary to avoid possible
          .             // overflow in the multiplication.
        136 (0.0%)      *stride = if *dim <= 1 { 0 } else { (s * step) as usize };
          .         
          .             offset
        102 (0.0%)  }
          .         
          .         /// Solves `a * x + b * y = gcd(a, b)` for `x`, `y`, and `gcd(a, b)`.
          .         ///
          .         /// Returns `(g, (x, y))`, where `g` is `gcd(a, b)`, and `g` is always
          .         /// nonnegative.
          .         ///
          .         /// See https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm
          .         fn extended_gcd(a: isize, b: isize) -> (isize, (isize, isize))
-- line 503 ----------------------------------------
-- line 682 ----------------------------------------
          .                 }
          .             }
          .             true
          .         }
          .         
          .         pub(crate) fn is_layout_c<D: Dimension>(dim: &D, strides: &D) -> bool
          .         {
          .             if let Some(1) = D::NDIM {
        346 (0.0%)          return strides[0] == 1 || dim[0] <= 1;
          .             }
          .         
          4 (0.0%)      for &d in dim.slice() {
        540 (0.0%)          if d == 0 {
          .                     return true;
          .                 }
          .             }
          .         
          .             let mut contig_stride = 1_isize;
          .             // check all dimensions -- a dimension of length 1 can have unequal strides
          .             for (&dim, &s) in izip!(dim.slice().iter().rev(), strides.slice().iter().rev()) {
        189 (0.0%)          if dim != 1 {
          .                     let s = s as isize;
        214 (0.0%)              if s != contig_stride {
          .                         return false;
          .                     }
         54 (0.0%)              contig_stride *= dim as isize;
          .                 }
          .             }
          .             true
          .         }
          .         
          .         pub(crate) fn is_layout_f<D: Dimension>(dim: &D, strides: &D) -> bool
          .         {
          .             if let Some(1) = D::NDIM {
-- line 715 ----------------------------------------

--------------------------------------------------------------------------------
-- Annotated source file: /home/kevin/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/lib/rustlib/src/rust/library/core/src/cmp.rs
--------------------------------------------------------------------------------
Ir__________________ 

    2,461,865 (0.0%)  <unknown (line 0)>

-- line 1436 ----------------------------------------
            .         ///
            .         /// let result = cmp::min_by(-2, 3, |x: &i32, y: &i32| x.abs().cmp(&y.abs()));
            .         /// assert_eq!(result, -2);
            .         /// ```
            .         #[inline]
            .         #[must_use]
            .         #[stable(feature = "cmp_min_max_by", since = "1.53.0")]
            .         pub fn min_by<T, F: FnOnce(&T, &T) -> Ordering>(v1: T, v2: T, compare: F) -> T {
            6 (0.0%)      match compare(&v1, &v2) {
            .                 Ordering::Less | Ordering::Equal => v1,
            .                 Ordering::Greater => v2,
            .             }
            .         }
            .         
            .         /// Returns the element that gives the minimum value from the specified function.
            .         ///
            .         /// Returns the first argument if the comparison determines them to be equal.
-- line 1452 ----------------------------------------
-- line 1628 ----------------------------------------
            .             use crate::cmp::Ordering::{self, Equal, Greater, Less};
            .             use crate::hint::unreachable_unchecked;
            .         
            .             macro_rules! partial_eq_impl {
            .                 ($($t:ty)*) => ($(
            .                     #[stable(feature = "rust1", since = "1.0.0")]
            .                     impl PartialEq for $t {
            .                         #[inline]
            7 (0.0%)                  fn eq(&self, other: &$t) -> bool { (*self) == (*other) }
            .                         #[inline]
          689 (0.0%)                  fn ne(&self, other: &$t) -> bool { (*self) != (*other) }
            .                     }
            .                 )*)
            .             }
            .         
            .             #[stable(feature = "rust1", since = "1.0.0")]
            .             impl PartialEq for () {
            .                 #[inline]
            .                 fn eq(&self, _other: &()) -> bool {
-- line 1646 ----------------------------------------
-- line 1709 ----------------------------------------
            .             partial_ord_impl! { f16 f32 f64 f128 }
            .         
            .             macro_rules! ord_impl {
            .                 ($($t:ty)*) => ($(
            .                     #[stable(feature = "rust1", since = "1.0.0")]
            .                     impl PartialOrd for $t {
            .                         #[inline]
            .                         fn partial_cmp(&self, other: &$t) -> Option<Ordering> {
      876,182 (0.0%)                      Some(crate::intrinsics::three_way_compare(*self, *other))
            .                         }
            .                         #[inline(always)]
1,827,966,656 (5.2%)                  fn lt(&self, other: &$t) -> bool { (*self) < (*other) }
            .                         #[inline(always)]
1,381,024,890 (3.9%)                  fn le(&self, other: &$t) -> bool { (*self) <= (*other) }
            .                         #[inline(always)]
            .                         fn ge(&self, other: &$t) -> bool { (*self) >= (*other) }
            .                         #[inline(always)]
            .                         fn gt(&self, other: &$t) -> bool { (*self) > (*other) }
            .                     }
            .         
            .                     #[stable(feature = "rust1", since = "1.0.0")]
            .                     impl Ord for $t {
-- line 1730 ----------------------------------------

--------------------------------------------------------------------------------
-- Annotated source file: /home/kevin/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/lib/rustlib/src/rust/library/core/src/iter/range.rs
--------------------------------------------------------------------------------
Ir________________ 

 17,592,631 (0.0%)  <unknown (line 0)>

-- line 744 ----------------------------------------
          .         
          .                 NonZero::new(n - taken).map_or(Ok(()), Err)
          .             }
          .         }
          .         
          .         impl<T: TrustedStep> RangeIteratorImpl for ops::Range<T> {
          .             #[inline]
          .             fn spec_next(&mut self) -> Option<T> {
849,779,917 (2.4%)          if self.start < self.end {
          .                     let old = self.start;
          .                     // SAFETY: just checked precondition
        353 (0.0%)              self.start = unsafe { Step::forward_unchecked(old, 1) };
          .                     Some(old)
          .                 } else {
          .                     None
          .                 }
          .             }
          .         
          .             #[inline]
          .             fn spec_nth(&mut self, n: usize) -> Option<T> {
-- line 763 ----------------------------------------
-- line 840 ----------------------------------------
          .         
          .             #[inline]
          .             fn next(&mut self) -> Option<A> {
          .                 self.spec_next()
          .             }
          .         
          .             #[inline]
          .             fn size_hint(&self) -> (usize, Option<usize>) {
         66 (0.0%)          if self.start < self.end {
          .                     let hint = Step::steps_between(&self.start, &self.end);
          .                     (hint.unwrap_or(usize::MAX), hint)
          .                 } else {
          .                     (0, Some(0))
          .                 }
          .             }
          .         
          .             #[inline]
-- line 856 ----------------------------------------
-- line 1148 ----------------------------------------
          .         
          .         impl<T: TrustedStep> RangeInclusiveIteratorImpl for ops::RangeInclusive<T> {
          .             #[inline]
          .             fn spec_next(&mut self) -> Option<T> {
          .                 if self.is_empty() {
          .                     return None;
          .                 }
          .                 let is_iterating = self.start < self.end;
        581 (0.0%)          Some(if is_iterating {
          .                     // SAFETY: just checked precondition
          .                     let n = unsafe { Step::forward_unchecked(self.start, 1) };
          .                     mem::replace(&mut self.start, n)
          .                 } else {
          .                     self.exhausted = true;
          .                     self.start
          .                 })
          .             }
-- line 1164 ----------------------------------------
-- line 1171 ----------------------------------------
          .                 R: Try<Output = B>,
          .             {
          .                 if self.is_empty() {
          .                     return try { init };
          .                 }
          .         
          .                 let mut accum = init;
          .         
         25 (0.0%)          while self.start < self.end {
          .                     // SAFETY: just checked precondition
          .                     let n = unsafe { Step::forward_unchecked(self.start, 1) };
          .                     let n = mem::replace(&mut self.start, n);
         23 (0.0%)              accum = f(accum, n)?;
          .                 }
          .         
          .                 self.exhausted = true;
          .         
          .                 if self.start == self.end {
          .                     accum = f(accum, self.start)?;
          .                 }
          .         
-- line 1191 ----------------------------------------

--------------------------------------------------------------------------------
-- Annotated source file: /home/kevin/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/lib/rustlib/src/rust/library/core/src/ops/arith.rs
--------------------------------------------------------------------------------
Ir_______________ 

-- line 95 ----------------------------------------
         .             ($($t:ty)*) => ($(
         .                 #[stable(feature = "rust1", since = "1.0.0")]
         .                 impl Add for $t {
         .                     type Output = $t;
         .         
         .                     #[inline]
         .                     #[track_caller]
         .                     #[rustc_inherit_overflow_checks]
   738,705 (0.0%)              fn add(self, other: $t) -> $t { self + other }
         .                 }
         .         
         .                 forward_ref_binop! { impl Add, add for $t, $t }
         .             )*)
         .         }
         .         
         .         add_impl! { usize u8 u16 u32 u64 u128 isize i8 i16 i32 i64 i128 f16 f32 f64 f128 }
         .         
-- line 111 ----------------------------------------
-- line 204 ----------------------------------------
         .             ($($t:ty)*) => ($(
         .                 #[stable(feature = "rust1", since = "1.0.0")]
         .                 impl Sub for $t {
         .                     type Output = $t;
         .         
         .                     #[inline]
         .                     #[track_caller]
         .                     #[rustc_inherit_overflow_checks]
   492,436 (0.0%)              fn sub(self, other: $t) -> $t { self - other }
         .                 }
         .         
         .                 forward_ref_binop! { impl Sub, sub for $t, $t }
         .             )*)
         .         }
         .         
         .         sub_impl! { usize u8 u16 u32 u64 u128 isize i8 i16 i32 i64 i128 f16 f32 f64 f128 }
         .         
-- line 220 ----------------------------------------
-- line 334 ----------------------------------------
         .             ($($t:ty)*) => ($(
         .                 #[stable(feature = "rust1", since = "1.0.0")]
         .                 impl Mul for $t {
         .                     type Output = $t;
         .         
         .                     #[inline]
         .                     #[track_caller]
         .                     #[rustc_inherit_overflow_checks]
47,316,308 (0.1%)              fn mul(self, other: $t) -> $t { self * other }
         .                 }
         .         
         .                 forward_ref_binop! { impl Mul, mul for $t, $t }
         .             )*)
         .         }
         .         
         .         mul_impl! { usize u8 u16 u32 u64 u128 isize i8 i16 i32 i64 i128 f16 f32 f64 f128 }
         .         
-- line 350 ----------------------------------------
-- line 492 ----------------------------------------
         .         
         .         macro_rules! div_impl_float {
         .             ($($t:ty)*) => ($(
         .                 #[stable(feature = "rust1", since = "1.0.0")]
         .                 impl Div for $t {
         .                     type Output = $t;
         .         
         .                     #[inline]
   328,316 (0.0%)              fn div(self, other: $t) -> $t { self / other }
         .                 }
         .         
         .                 forward_ref_binop! { impl Div, div for $t, $t }
         .             )*)
         .         }
         .         
         .         div_impl_float! { f16 f32 f64 f128 }
         .         
-- line 508 ----------------------------------------
-- line 751 ----------------------------------------
         .         
         .         macro_rules! add_assign_impl {
         .             ($($t:ty)+) => ($(
         .                 #[stable(feature = "op_assign_traits", since = "1.8.0")]
         .                 impl AddAssign for $t {
         .                     #[inline]
         .                     #[track_caller]
         .                     #[rustc_inherit_overflow_checks]
44,112,483 (0.1%)              fn add_assign(&mut self, other: $t) { *self += other }
         .                 }
         .         
         .                 forward_ref_op_assign! { impl AddAssign, add_assign for $t, $t }
         .             )+)
         .         }
         .         
         .         add_assign_impl! { usize u8 u16 u32 u64 u128 isize i8 i16 i32 i64 i128 f16 f32 f64 f128 }
         .         
-- line 767 ----------------------------------------

--------------------------------------------------------------------------------
-- Annotated source file: /home/kevin/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/lib/rustlib/src/rust/library/core/src/ops/range.rs
--------------------------------------------------------------------------------
Ir__________________ 

           39 (0.0%)  <unknown (line 0)>

-- line 538 ----------------------------------------
            .             /// let mut r = 3..=5;
            .             /// for _ in r.by_ref() {}
            .             /// // Precise field values are unspecified here
            .             /// assert!(r.is_empty());
            .             /// ```
            .             #[stable(feature = "range_is_empty", since = "1.47.0")]
            .             #[inline]
            .             pub fn is_empty(&self) -> bool {
          181 (0.0%)          self.exhausted || !(self.start <= self.end)
            .             }
            .         }
            .         
            .         /// A range only bounded inclusively above (`..=end`).
            .         ///
            .         /// The `RangeToInclusive` `..=end` contains all values with `x <= end`.
            .         /// It cannot serve as an [`Iterator`] because it doesn't have a starting point.
            .         ///
-- line 554 ----------------------------------------
-- line 814 ----------------------------------------
            .             /// assert!(!(f32::NAN..1.0).contains(&0.5));
            .             #[inline]
            .             #[stable(feature = "range_contains", since = "1.35.0")]
            .             fn contains<U>(&self, item: &U) -> bool
            .             where
            .                 T: PartialOrd<U>,
            .                 U: ?Sized + PartialOrd<T>,
            .             {
1,381,024,890 (3.9%)          (match self.start_bound() {
            .                     Included(start) => start <= item,
            .                     Excluded(start) => start < item,
            .                     Unbounded => true,
            .                 }) && (match self.end_bound() {
            .                     Included(end) => item <= end,
            .                     Excluded(end) => item < end,
            .                     Unbounded => true,
            .                 })
-- line 830 ----------------------------------------

--------------------------------------------------------------------------------
-- Annotated source file: /home/kevin/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/lib/rustlib/src/rust/library/core/src/option.rs
--------------------------------------------------------------------------------
Ir________________ 

  7,463,867 (0.0%)  <unknown (line 0)>

-- line 599 ----------------------------------------
          .             /// let x: Option<u32> = None;
          .             /// assert_eq!(x.is_some(), false);
          .             /// ```
          .             #[must_use = "if you intended to assert that this has a value, consider `.unwrap()` instead"]
          .             #[inline]
          .             #[stable(feature = "rust1", since = "1.0.0")]
          .             #[rustc_const_stable(feature = "const_option_basics", since = "1.48.0")]
          .             pub const fn is_some(&self) -> bool {
     58,137 (0.0%)          matches!(*self, Some(_))
          .             }
          .         
          .             /// Returns `true` if the option is a [`Some`] and the value inside of it matches a predicate.
          .             ///
          .             /// # Examples
          .             ///
          .             /// ```
          .             /// let x: Option<u32> = Some(2);
-- line 615 ----------------------------------------
-- line 698 ----------------------------------------
          .             /// // then consume *that* with `map`, leaving `text` on the stack.
          .             /// let text_length: Option<usize> = text.as_ref().map(|s| s.len());
          .             /// println!("still can print text: {text:?}");
          .             /// ```
          .             #[inline]
          .             #[rustc_const_stable(feature = "const_option_basics", since = "1.48.0")]
          .             #[stable(feature = "rust1", since = "1.0.0")]
          .             pub const fn as_ref(&self) -> Option<&T> {
  1,924,151 (0.0%)          match *self {
          .                     Some(ref x) => Some(x),
          .                     None => None,
          .                 }
          .             }
          .         
          .             /// Converts from `&mut Option<T>` to `Option<&mut T>`.
          .             ///
          .             /// # Examples
-- line 714 ----------------------------------------
-- line 721 ----------------------------------------
          .             /// }
          .             /// assert_eq!(x, Some(42));
          .             /// ```
          .             #[inline]
          .             #[stable(feature = "rust1", since = "1.0.0")]
          .             #[cfg_attr(bootstrap, rustc_allow_const_fn_unstable(const_mut_refs))]
          .             #[rustc_const_stable(feature = "const_option", since = "1.83.0")]
          .             pub const fn as_mut(&mut self) -> Option<&mut T> {
  5,252,714 (0.0%)          match *self {
          .                     Some(ref mut x) => Some(x),
          .                     None => None,
          .                 }
          .             }
          .         
          .             /// Converts from <code>[Pin]<[&]Option\<T>></code> to <code>Option<[Pin]<[&]T>></code>.
          .             ///
          .             /// [&]: reference "shared reference"
-- line 737 ----------------------------------------
-- line 923 ----------------------------------------
          .             /// Styles"](../../std/error/index.html#common-message-styles) in the [`std::error`](../../std/error/index.html) module docs.
          .             #[inline]
          .             #[track_caller]
          .             #[stable(feature = "rust1", since = "1.0.0")]
          .             #[cfg_attr(not(test), rustc_diagnostic_item = "option_expect")]
          .             #[rustc_allow_const_fn_unstable(const_precise_live_drops)]
          .             #[rustc_const_stable(feature = "const_option", since = "1.83.0")]
          .             pub const fn expect(self, msg: &str) -> T {
          7 (0.0%)          match self {
          .                     Some(val) => val,
          .                     None => expect_failed(msg),
          .                 }
          .             }
          .         
          .             /// Returns the contained [`Some`] value, consuming the `self` value.
          .             ///
          .             /// Because this function may panic, its use is generally discouraged.
-- line 939 ----------------------------------------
-- line 962 ----------------------------------------
          .             /// ```
          .             #[inline(always)]
          .             #[track_caller]
          .             #[stable(feature = "rust1", since = "1.0.0")]
          .             #[cfg_attr(not(test), rustc_diagnostic_item = "option_unwrap")]
          .             #[rustc_allow_const_fn_unstable(const_precise_live_drops)]
          .             #[rustc_const_stable(feature = "const_option", since = "1.83.0")]
          .             pub const fn unwrap(self) -> T {
      2,633 (0.0%)          match self {
         25 (0.0%)              Some(val) => val,
          .                     None => unwrap_failed(),
          .                 }
          .             }
          .         
          .             /// Returns the contained [`Some`] value or a provided default.
          .             ///
          .             /// Arguments passed to `unwrap_or` are eagerly evaluated; if you are passing
          .             /// the result of a function call, it is recommended to use [`unwrap_or_else`],
-- line 979 ----------------------------------------
-- line 985 ----------------------------------------
          .             ///
          .             /// ```
          .             /// assert_eq!(Some("car").unwrap_or("bike"), "car");
          .             /// assert_eq!(None.unwrap_or("bike"), "bike");
          .             /// ```
          .             #[inline]
          .             #[stable(feature = "rust1", since = "1.0.0")]
          .             pub fn unwrap_or(self, default: T) -> T {
    370,202 (0.0%)          match self {
          .                     Some(x) => x,
          .                     None => default,
          .                 }
          .             }
          .         
          .             /// Returns the contained [`Some`] value or computes it from a closure.
          .             ///
          .             /// # Examples
-- line 1001 ----------------------------------------
-- line 1007 ----------------------------------------
          .             /// ```
          .             #[inline]
          .             #[track_caller]
          .             #[stable(feature = "rust1", since = "1.0.0")]
          .             pub fn unwrap_or_else<F>(self, f: F) -> T
          .             where
          .                 F: FnOnce() -> T,
          .             {
328,549,155 (0.9%)          match self {
         32 (0.0%)              Some(x) => x,
          3 (0.0%)              None => f(),
          .                 }
          .             }
          .         
          .             /// Returns the contained [`Some`] value or a default.
          .             ///
          .             /// Consumes the `self` argument then, if [`Some`], returns the contained
          .             /// value, otherwise if [`None`], returns the [default value] for that
          .             /// type.
-- line 1025 ----------------------------------------
-- line 1038 ----------------------------------------
          .             /// [`parse`]: str::parse
          .             /// [`FromStr`]: crate::str::FromStr
          .             #[inline]
          .             #[stable(feature = "rust1", since = "1.0.0")]
          .             pub fn unwrap_or_default(self) -> T
          .             where
          .                 T: Default,
          .             {
          2 (0.0%)          match self {
          .                     Some(x) => x,
          .                     None => T::default(),
          .                 }
          .             }
          .         
          .             /// Returns the contained [`Some`] value, consuming the `self` value,
          .             /// without checking that the value is not [`None`].
          .             ///
-- line 1054 ----------------------------------------
-- line 1104 ----------------------------------------
          .             /// assert_eq!(x.map(|s| s.len()), None);
          .             /// ```
          .             #[inline]
          .             #[stable(feature = "rust1", since = "1.0.0")]
          .             pub fn map<U, F>(self, f: F) -> Option<U>
          .             where
          .                 F: FnOnce(T) -> U,
          .             {
  5,466,133 (0.0%)          match self {
     16,515 (0.0%)              Some(x) => Some(f(x)),
          2 (0.0%)              None => None,
          .                 }
          .             }
          .         
          .             /// Calls a function with a reference to the contained value if [`Some`].
          .             ///
          .             /// Returns the original option.
          .             ///
          .             /// # Examples
-- line 1122 ----------------------------------------
-- line 1163 ----------------------------------------
          .             /// ```
          .             #[inline]
          .             #[stable(feature = "rust1", since = "1.0.0")]
          .             #[must_use = "if you don't need the returned value, use `if let` instead"]
          .             pub fn map_or<U, F>(self, default: U, f: F) -> U
          .             where
          .                 F: FnOnce(T) -> U,
          .             {
         44 (0.0%)          match self {
          .                     Some(t) => f(t),
          .                     None => default,
          .                 }
          .             }
          .         
          .             /// Computes a default function result (if none), or
          .             /// applies a different function to the contained value (if any).
          .             ///
-- line 1179 ----------------------------------------
-- line 1267 ----------------------------------------
          .             /// assert_eq!(x.ok_or_else(|| 0), Err(0));
          .             /// ```
          .             #[inline]
          .             #[stable(feature = "rust1", since = "1.0.0")]
          .             pub fn ok_or_else<E, F>(self, err: F) -> Result<T, E>
          .             where
          .                 F: FnOnce() -> E,
          .             {
         18 (0.0%)          match self {
          .                     Some(v) => Ok(v),
          .                     None => Err(err()),
          .                 }
          .             }
          .         
          .             /// Converts from `Option<T>` (or `&Option<T>`) to `Option<&T::Target>`.
          .             ///
          .             /// Leaves the original Option in-place, creating a new one with a reference
-- line 1283 ----------------------------------------
-- line 1437 ----------------------------------------
          .             #[doc(alias = "flatmap")]
          .             #[inline]
          .             #[stable(feature = "rust1", since = "1.0.0")]
          .             #[rustc_confusables("flat_map", "flatmap")]
          .             pub fn and_then<U, F>(self, f: F) -> Option<U>
          .             where
          .                 F: FnOnce(T) -> Option<U>,
          .             {
         12 (0.0%)          match self {
          .                     Some(x) => f(x),
          .                     None => None,
          .                 }
          .             }
          .         
          .             /// Returns [`None`] if the option is [`None`], otherwise calls `predicate`
          .             /// with the wrapped value and returns:
          .             ///
-- line 1453 ----------------------------------------
-- line 1474 ----------------------------------------
          .             /// [`Some(t)`]: Some
          .             #[inline]
          .             #[stable(feature = "option_filter", since = "1.27.0")]
          .             pub fn filter<P>(self, predicate: P) -> Self
          .             where
          .                 P: FnOnce(&T) -> bool,
          .             {
          .                 if let Some(x) = self {
          2 (0.0%)              if predicate(&x) {
          .                         return Some(x);
          .                     }
          .                 }
          .                 None
          .             }
          .         
          .             /// Returns the option if it contains a value, otherwise returns `optb`.
          .             ///
-- line 1490 ----------------------------------------
-- line 1536 ----------------------------------------
          .             /// assert_eq!(None.or_else(nobody), None);
          .             /// ```
          .             #[inline]
          .             #[stable(feature = "rust1", since = "1.0.0")]
          .             pub fn or_else<F>(self, f: F) -> Option<T>
          .             where
          .                 F: FnOnce() -> Option<T>,
          .             {
         28 (0.0%)          match self {
          .                     x @ Some(_) => x,
          .                     None => f(),
          .                 }
          .             }
          .         
          .             /// Returns [`Some`] if exactly one of `self`, `optb` is [`Some`], otherwise returns [`None`].
          .             ///
          .             /// # Examples
-- line 1552 ----------------------------------------
-- line 1681 ----------------------------------------
          .             /// assert_eq!(x, Some(7));
          .             /// ```
          .             #[inline]
          .             #[stable(feature = "option_entry", since = "1.20.0")]
          .             pub fn get_or_insert_with<F>(&mut self, f: F) -> &mut T
          .             where
          .                 F: FnOnce() -> T,
          .             {
    680,097 (0.0%)          if let None = self {
  1,048,148 (0.0%)              *self = Some(f());
          .                 }
          .         
          .                 // SAFETY: a `None` variant for `self` would have been replaced by a `Some`
          .                 // variant in the code above.
          .                 unsafe { self.as_mut().unwrap_unchecked() }
          .             }
          .         
          .             /////////////////////////////////////////////////////////////////////////
-- line 1698 ----------------------------------------
-- line 1887 ----------------------------------------
          .             #[rustc_const_stable(feature = "const_option", since = "1.83.0")]
          .             pub const fn copied(self) -> Option<T>
          .             where
          .                 T: Copy,
          .             {
          .                 // FIXME(const-hack): this implementation, which sidesteps using `Option::map` since it's not const
          .                 // ready yet, should be reverted when possible to avoid code repetition
          .                 match self {
    984,868 (0.0%)              Some(&v) => Some(v),
          .                     None => None,
          .                 }
          .             }
          .         
          .             /// Maps an `Option<&T>` to an `Option<T>` by cloning the contents of the
          .             /// option.
          .             ///
          .             /// # Examples
-- line 1903 ----------------------------------------
-- line 1910 ----------------------------------------
          .             /// assert_eq!(cloned, Some(12));
          .             /// ```
          .             #[must_use = "`self` will be dropped if the result is not used"]
          .             #[stable(feature = "rust1", since = "1.0.0")]
          .             pub fn cloned(self) -> Option<T>
          .             where
          .                 T: Clone,
          .             {
      1,732 (0.0%)          match self {
          .                     Some(t) => Some(t.clone()),
          .                     None => None,
          .                 }
          .             }
          .         }
          .         
          .         impl<T> Option<&mut T> {
          .             /// Maps an `Option<&mut T>` to an `Option<T>` by copying the contents of the
-- line 1926 ----------------------------------------
-- line 2027 ----------------------------------------
          .         
          .         #[stable(feature = "rust1", since = "1.0.0")]
          .         impl<T> Clone for Option<T>
          .         where
          .             T: Clone,
          .         {
          .             #[inline]
          .             fn clone(&self) -> Self {
     25,717 (0.0%)          match self {
          .                     Some(x) => Some(x.clone()),
          .                     None => None,
          .                 }
          .             }
          .         
          .             #[inline]
          .             fn clone_from(&mut self, source: &Self) {
          .                 match (self, source) {
-- line 2043 ----------------------------------------
-- line 2178 ----------------------------------------
          .         #[stable(feature = "rust1", since = "1.0.0")]
          .         impl<T> crate::marker::StructuralPartialEq for Option<T> {}
          .         #[stable(feature = "rust1", since = "1.0.0")]
          .         impl<T: PartialEq> PartialEq for Option<T> {
          .             #[inline]
          .             fn eq(&self, other: &Self) -> bool {
          .                 // Spelling out the cases explicitly optimizes better than
          .                 // `_ => false`
        193 (0.0%)          match (self, other) {
          .                     (Some(l), Some(r)) => *l == *r,
          .                     (Some(_), None) => false,
          .                     (None, Some(_)) => false,
          .                     (None, None) => true,
          .                 }
          .             }
          .         }
          .         
-- line 2194 ----------------------------------------
-- line 2476 ----------------------------------------
          .         
          .             #[inline]
          .             fn from_output(output: Self::Output) -> Self {
          .                 Some(output)
          .             }
          .         
          .             #[inline]
          .             fn branch(self) -> ControlFlow<Self::Residual, Self::Output> {
    196,053 (0.0%)          match self {
          .                     Some(v) => ControlFlow::Continue(v),
          .                     None => ControlFlow::Break(None),
          .                 }
          .             }
          .         }
          .         
          .         #[unstable(feature = "try_trait_v2", issue = "84277")]
          .         // Note: manually specifying the residual type instead of using the default to work around
          .         // https://github.com/rust-lang/rust/issues/99940
          .         impl<T> ops::FromResidual<Option<convert::Infallible>> for Option<T> {
          .             #[inline]
          .             fn from_residual(residual: Option<convert::Infallible>) -> Self {
          .                 match residual {
          1 (0.0%)              None => None,
          .                 }
          .             }
          .         }
          .         
          .         #[diagnostic::do_not_recommend]
          .         #[unstable(feature = "try_trait_v2_yeet", issue = "96374")]
          .         impl<T> ops::FromResidual<ops::Yeet<()>> for Option<T> {
          .             #[inline]
-- line 2506 ----------------------------------------

--------------------------------------------------------------------------------
-- Annotated source file: /home/kevin/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/lib/rustlib/src/rust/library/core/src/ptr/mut_ptr.rs
--------------------------------------------------------------------------------
Ir________________ 

    983,215 (0.0%)  <unknown (line 0)>

-- line 422 ----------------------------------------
          .                         count: isize = count,
          .                         size: usize = size_of::<T>(),
          .                     ) => runtime_offset_nowrap(this, count, size)
          .                 );
          .         
          .                 // SAFETY: the caller must uphold the safety contract for `offset`.
          .                 // The obtained pointer is valid for writes since the caller must
          .                 // guarantee that it points to the same allocated object as `self`.
656,264,893 (1.9%)          unsafe { intrinsics::offset(self, count) }
          .             }
          .         
          .             /// Adds a signed offset in bytes to a pointer.
          .             ///
          .             /// `count` is in units of **bytes**.
          .             ///
          .             /// This is purely a convenience for casting to a `u8` pointer and
          .             /// using [offset][pointer::offset] on it. See that method for documentation
-- line 438 ----------------------------------------
-- line 997 ----------------------------------------
          .                     (
          .                         this: *const () = self as *const (),
          .                         count: usize = count,
          .                         size: usize = size_of::<T>(),
          .                     ) => runtime_add_nowrap(this, count, size)
          .                 );
          .         
          .                 // SAFETY: the caller must uphold the safety contract for `offset`.
    473,634 (0.0%)          unsafe { intrinsics::offset(self, count) }
          .             }
          .         
          .             /// Adds an unsigned offset in bytes to a pointer.
          .             ///
          .             /// `count` is in units of bytes.
          .             ///
          .             /// This is purely a convenience for casting to a `u8` pointer and
          .             /// using [add][pointer::add] on it. See that method for documentation
-- line 1013 ----------------------------------------

--------------------------------------------------------------------------------
-- Annotated source file: /home/kevin/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/lib/rustlib/src/rust/library/core/src/ptr/non_null.rs
--------------------------------------------------------------------------------
Ir_______________ 

 1,984,176 (0.0%)  <unknown (line 0)>

-- line 321 ----------------------------------------
         .             /// assert_eq!(x_value, 2);
         .             /// ```
         .             #[stable(feature = "nonnull", since = "1.25.0")]
         .             #[rustc_const_stable(feature = "const_nonnull_as_ptr", since = "1.32.0")]
         .             #[rustc_never_returns_null_ptr]
         .             #[must_use]
         .             #[inline(always)]
         .             pub const fn as_ptr(self) -> *mut T {
        17 (0.0%)          self.pointer as *mut T
         .             }
         .         
         .             /// Returns a shared reference to the value. If the value may be uninitialized, [`as_uninit_ref`]
         .             /// must be used instead.
         .             ///
         .             /// For the mutable counterpart see [`as_mut`].
         .             ///
         .             /// [`as_uninit_ref`]: NonNull::as_uninit_ref
-- line 337 ----------------------------------------
-- line 358 ----------------------------------------
         .             #[stable(feature = "nonnull", since = "1.25.0")]
         .             #[rustc_const_stable(feature = "const_nonnull_as_ref", since = "1.73.0")]
         .             #[must_use]
         .             #[inline(always)]
         .             pub const unsafe fn as_ref<'a>(&self) -> &'a T {
         .                 // SAFETY: the caller must guarantee that `self` meets all the
         .                 // requirements for a reference.
         .                 // `cast_const` avoids a mutable raw pointer deref.
   464,725 (0.0%)          unsafe { &*self.as_ptr().cast_const() }
         .             }
         .         
         .             /// Returns a unique reference to the value. If the value may be uninitialized, [`as_uninit_mut`]
         .             /// must be used instead.
         .             ///
         .             /// For the shared counterpart see [`as_ref`].
         .             ///
         .             /// [`as_uninit_mut`]: NonNull::as_uninit_mut
-- line 374 ----------------------------------------
-- line 471 ----------------------------------------
         .             pub const unsafe fn offset(self, count: isize) -> Self
         .             where
         .                 T: Sized,
         .             {
         .                 // SAFETY: the caller must uphold the safety contract for `offset`.
         .                 // Additionally safety contract of `offset` guarantees that the resulting pointer is
         .                 // pointing to an allocation, there can't be an allocation at null, thus it's safe to
         .                 // construct `NonNull`.
 4,435,558 (0.0%)          unsafe { NonNull { pointer: intrinsics::offset(self.pointer, count) } }
         .             }
         .         
         .             /// Calculates the offset from a pointer in bytes.
         .             ///
         .             /// `count` is in units of **bytes**.
         .             ///
         .             /// This is purely a convenience for casting to a `u8` pointer and
         .             /// using [offset][pointer::offset] on it. See that method for documentation
-- line 487 ----------------------------------------
-- line 547 ----------------------------------------
         .             pub const unsafe fn add(self, count: usize) -> Self
         .             where
         .                 T: Sized,
         .             {
         .                 // SAFETY: the caller must uphold the safety contract for `offset`.
         .                 // Additionally safety contract of `offset` guarantees that the resulting pointer is
         .                 // pointing to an allocation, there can't be an allocation at null, thus it's safe to
         .                 // construct `NonNull`.
 4,124,146 (0.0%)          unsafe { NonNull { pointer: intrinsics::offset(self.pointer, count) } }
         .             }
         .         
         .             /// Calculates the offset from a pointer in bytes (convenience for `.byte_offset(count as isize)`).
         .             ///
         .             /// `count` is in units of bytes.
         .             ///
         .             /// This is purely a convenience for casting to a `u8` pointer and
         .             /// using [`add`][NonNull::add] on it. See that method for documentation
-- line 563 ----------------------------------------
-- line 1701 ----------------------------------------
         .         #[stable(feature = "nonnull", since = "1.25.0")]
         .         impl<T: ?Sized> Eq for NonNull<T> {}
         .         
         .         #[stable(feature = "nonnull", since = "1.25.0")]
         .         impl<T: ?Sized> PartialEq for NonNull<T> {
         .             #[inline]
         .             #[allow(ambiguous_wide_pointer_comparisons)]
         .             fn eq(&self, other: &Self) -> bool {
93,540,053 (0.3%)          self.as_ptr() == other.as_ptr()
         .             }
         .         }
         .         
         .         #[stable(feature = "nonnull", since = "1.25.0")]
         .         impl<T: ?Sized> Ord for NonNull<T> {
         .             #[inline]
         .             #[allow(ambiguous_wide_pointer_comparisons)]
         .             fn cmp(&self, other: &Self) -> Ordering {
-- line 1717 ----------------------------------------

--------------------------------------------------------------------------------
-- Annotated source file: /home/kevin/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/lib/rustlib/src/rust/library/core/src/slice/iter/macros.rs
--------------------------------------------------------------------------------
Ir_______________ 

12,491,843 (0.0%)  <unknown (line 0)>

-- line 17 ----------------------------------------
         .                     let $len = unsafe { &mut *(&raw mut $this.end_or_len).cast::<usize>() };
         .                     $zst_body
         .                 } else {
         .                     // SAFETY: for non-ZSTs, the type invariant ensures it cannot be null
         .                     let $end = unsafe { &mut *(&raw mut $this.end_or_len).cast::<NonNull<T>>() };
         .                     $other_body
         .                 }
         .             }};
49,346,441 (0.1%)      ($this:ident, $len:ident => $zst_body:expr, $end:ident => $other_body:expr,) => {{
         .                 #![allow(unused_unsafe)] // we're sometimes used within an unsafe block
         .         
         .                 if T::IS_ZST {
         .                     let $len = $this.end_or_len.addr();
         .                     $zst_body
         .                 } else {
         .                     // SAFETY: for non-ZSTs, the type invariant ensures it cannot be null
         2 (0.0%)              let $end = unsafe { *(&raw const $this.end_or_len).cast::<NonNull<T>>() };
         .                     $other_body
         .                 }
         .             }};
         .         }
         .         
         .         // Inlining is_empty and len makes a huge performance difference
         .         macro_rules! is_empty {
         .             ($self: ident) => {
-- line 41 ----------------------------------------
-- line 100 ----------------------------------------
         .                         let old = self.ptr;
         .         
         .                         // SAFETY: the caller guarantees that `offset` doesn't exceed `self.len()`,
         .                         // so this new pointer is inside `self` and thus guaranteed to be non-null.
         .                         unsafe {
         .                             if_zst!(mut self,
         .                                 // Using the intrinsic directly avoids emitting a UbCheck
         .                                 len => *len = crate::intrinsics::unchecked_sub(*len, offset),
   329,541 (0.0%)                          _end => self.ptr = self.ptr.add(offset),
         .                             );
         .                         }
         .                         old
         .                     }
         .         
         .                     // Helper function for moving the end of the iterator backwards by `offset` elements,
         .                     // returning the new end.
         .                     // Unsafe because the offset must not exceed `self.len()`.
-- line 116 ----------------------------------------
-- line 221 ----------------------------------------
         .                         //   some optimizations, see #111603
         .                         // - avoids Option wrapping/matching
         .                         if is_empty!(self) {
         .                             return init;
         .                         }
         .                         let mut acc = init;
         .                         let mut i = 0;
         .                         let len = len!(self);
        12 (0.0%)                  loop {
         .                             // SAFETY: the loop iterates `i in 0..len`, which always is in bounds of
         .                             // the slice allocation
 4,678,181 (0.0%)                      acc = f(acc, unsafe { & $( $mut_ )? *self.ptr.add(i).as_ptr() });
         .                             // SAFETY: `i` can't overflow since it'll only reach usize::MAX if the
         .                             // slice had that length, in which case we'll break out of the loop
         .                             // after the increment
         .                             i = unsafe { i.unchecked_add(1) };
10,260,072 (0.0%)                      if i == len {
         .                                 break;
         .                             }
         .                         }
         .                         acc
         .                     }
         .         
         .                     // We override the default implementation, which uses `try_fold`,
         .                     // because this simple implementation generates less LLVM IR and is
-- line 245 ----------------------------------------
-- line 246 ----------------------------------------
         .                     // faster to compile.
         .                     #[inline]
         .                     fn for_each<F>(mut self, mut f: F)
         .                     where
         .                         Self: Sized,
         .                         F: FnMut(Self::Item),
         .                     {
         .                         while let Some(x) = self.next() {
88,224,966 (0.2%)                      f(x);
         .                         }
         .                     }
         .         
         .                     // We override the default implementation, which uses `try_fold`,
         .                     // because this simple implementation generates less LLVM IR and is
         .                     // faster to compile.
         .                     #[inline]
         .                     fn all<F>(&mut self, mut f: F) -> bool
         .                     where
         .                         Self: Sized,
         .                         F: FnMut(Self::Item) -> bool,
         .                     {
        15 (0.0%)                  while let Some(x) = self.next() {
        99 (0.0%)                      if !f(x) {
         .                                 return false;
         .                             }
         .                         }
         .                         true
         .                     }
         .         
         .                     // We override the default implementation, which uses `try_fold`,
         .                     // because this simple implementation generates less LLVM IR and is
         .                     // faster to compile.
         .                     #[inline]
         .                     fn any<F>(&mut self, mut f: F) -> bool
         .                     where
         .                         Self: Sized,
         .                         F: FnMut(Self::Item) -> bool,
         .                     {
        12 (0.0%)                  while let Some(x) = self.next() {
        64 (0.0%)                      if f(x) {
         .                                 return true;
         .                             }
         .                         }
         .                         false
         .                     }
         .         
         .                     // We override the default implementation, which uses `try_fold`,
         .                     // because this simple implementation generates less LLVM IR and is
-- line 293 ----------------------------------------
-- line 294 ----------------------------------------
         .                     // faster to compile.
         .                     #[inline]
         .                     fn find<P>(&mut self, mut predicate: P) -> Option<Self::Item>
         .                     where
         .                         Self: Sized,
         .                         P: FnMut(&Self::Item) -> bool,
         .                     {
         .                         while let Some(x) = self.next() {
        65 (0.0%)                      if predicate(&x) {
         .                                 return Some(x);
         .                             }
         .                         }
         .                         None
         .                     }
         .         
         .                     // We override the default implementation, which uses `try_fold`,
         .                     // because this simple implementation generates less LLVM IR and is
-- line 310 ----------------------------------------

--------------------------------------------------------------------------------
-- Annotated source file: /home/kevin/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/lib/rustlib/src/rust/library/std/src/f64.rs
--------------------------------------------------------------------------------
Ir________________ 

    500,376 (0.0%)  <unknown (line 0)>

-- line 66 ----------------------------------------
          .             /// assert_eq!(g.ceil(), 4.0);
          .             /// ```
          .             #[doc(alias = "ceiling")]
          .             #[rustc_allow_incoherent_impl]
          .             #[must_use = "method returns a new number and does not mutate the original value"]
          .             #[stable(feature = "rust1", since = "1.0.0")]
          .             #[inline]
          .             pub fn ceil(self) -> f64 {
      2,336 (0.0%)          unsafe { intrinsics::ceilf64(self) }
          .             }
          .         
          .             /// Returns the nearest integer to `self`. If a value is half-way between two
          .             /// integers, round away from `0.0`.
          .             ///
          .             /// This function always returns the precise result.
          .             ///
          .             /// # Examples
-- line 82 ----------------------------------------
-- line 94 ----------------------------------------
          .             /// assert_eq!(i.round(), 4.0);
          .             /// assert_eq!(j.round(), 5.0);
          .             /// ```
          .             #[rustc_allow_incoherent_impl]
          .             #[must_use = "method returns a new number and does not mutate the original value"]
          .             #[stable(feature = "rust1", since = "1.0.0")]
          .             #[inline]
          .             pub fn round(self) -> f64 {
      1,168 (0.0%)          unsafe { intrinsics::roundf64(self) }
          .             }
          .         
          .             /// Returns the nearest integer to a number. Rounds half-way cases to the number
          .             /// with an even least significant digit.
          .             ///
          .             /// This function always returns the precise result.
          .             ///
          .             /// # Examples
-- line 110 ----------------------------------------
-- line 145 ----------------------------------------
          .             /// assert_eq!(h.trunc(), -3.0);
          .             /// ```
          .             #[doc(alias = "truncate")]
          .             #[rustc_allow_incoherent_impl]
          .             #[must_use = "method returns a new number and does not mutate the original value"]
          .             #[stable(feature = "rust1", since = "1.0.0")]
          .             #[inline]
          .             pub fn trunc(self) -> f64 {
      1,167 (0.0%)          unsafe { intrinsics::truncf64(self) }
          .             }
          .         
          .             /// Returns the fractional part of `self`.
          .             ///
          .             /// This function always returns the precise result.
          .             ///
          .             /// # Examples
          .             ///
-- line 161 ----------------------------------------
-- line 168 ----------------------------------------
          .             /// assert!(abs_difference_x < 1e-10);
          .             /// assert!(abs_difference_y < 1e-10);
          .             /// ```
          .             #[rustc_allow_incoherent_impl]
          .             #[must_use = "method returns a new number and does not mutate the original value"]
          .             #[stable(feature = "rust1", since = "1.0.0")]
          .             #[inline]
          .             pub fn fract(self) -> f64 {
      1,167 (0.0%)          self - self.trunc()
          .             }
          .         
          .             /// Computes the absolute value of `self`.
          .             ///
          .             /// This function always returns the precise result.
          .             ///
          .             /// # Examples
          .             ///
-- line 184 ----------------------------------------
-- line 191 ----------------------------------------
          .             ///
          .             /// assert!(f64::NAN.abs().is_nan());
          .             /// ```
          .             #[rustc_allow_incoherent_impl]
          .             #[must_use = "method returns a new number and does not mutate the original value"]
          .             #[stable(feature = "rust1", since = "1.0.0")]
          .             #[inline]
          .             pub fn abs(self) -> f64 {
  1,969,768 (0.0%)          unsafe { intrinsics::fabsf64(self) }
          .             }
          .         
          .             /// Returns a number that represents the sign of `self`.
          .             ///
          .             /// - `1.0` if the number is positive, `+0.0` or `INFINITY`
          .             /// - `-1.0` if the number is negative, `-0.0` or `NEG_INFINITY`
          .             /// - NaN if the number is NaN
          .             ///
-- line 207 ----------------------------------------
-- line 389 ----------------------------------------
          .             ///
          .             /// assert!(abs_difference < 1e-10);
          .             /// ```
          .             #[rustc_allow_incoherent_impl]
          .             #[must_use = "method returns a new number and does not mutate the original value"]
          .             #[stable(feature = "rust1", since = "1.0.0")]
          .             #[inline]
          .             pub fn powi(self, n: i32) -> f64 {
904,143,105 (2.6%)          unsafe { intrinsics::powif64(self, n) }
          .             }
          .         
          .             /// Raises a number to a floating point power.
          .             ///
          .             /// # Unspecified precision
          .             ///
          .             /// The precision of this function is non-deterministic. This means it varies by platform, Rust version, and
          .             /// can even differ within the same execution from one invocation to the next.
-- line 405 ----------------------------------------
-- line 412 ----------------------------------------
          .             ///
          .             /// assert!(abs_difference < 1e-10);
          .             /// ```
          .             #[rustc_allow_incoherent_impl]
          .             #[must_use = "method returns a new number and does not mutate the original value"]
          .             #[stable(feature = "rust1", since = "1.0.0")]
          .             #[inline]
          .             pub fn powf(self, n: f64) -> f64 {
    182,433 (0.0%)          unsafe { intrinsics::powf64(self, n) }
          .             }
          .         
          .             /// Returns the square root of a number.
          .             ///
          .             /// Returns NaN if `self` is a negative number other than `-0.0`.
          .             ///
          .             /// # Precision
          .             ///
-- line 428 ----------------------------------------
-- line 441 ----------------------------------------
          .             /// assert!(negative.sqrt().is_nan());
          .             /// assert!(negative_zero.sqrt() == negative_zero);
          .             /// ```
          .             #[rustc_allow_incoherent_impl]
          .             #[must_use = "method returns a new number and does not mutate the original value"]
          .             #[stable(feature = "rust1", since = "1.0.0")]
          .             #[inline]
          .             pub fn sqrt(self) -> f64 {
  4,264,890 (0.0%)          unsafe { intrinsics::sqrtf64(self) }
          .             }
          .         
          .             /// Returns `e^(self)`, (the exponential function).
          .             ///
          .             /// # Unspecified precision
          .             ///
          .             /// The precision of this function is non-deterministic. This means it varies by platform, Rust version, and
          .             /// can even differ within the same execution from one invocation to the next.
-- line 457 ----------------------------------------
-- line 468 ----------------------------------------
          .             ///
          .             /// assert!(abs_difference < 1e-10);
          .             /// ```
          .             #[rustc_allow_incoherent_impl]
          .             #[must_use = "method returns a new number and does not mutate the original value"]
          .             #[stable(feature = "rust1", since = "1.0.0")]
          .             #[inline]
          .             pub fn exp(self) -> f64 {
     33,461 (0.0%)          unsafe { intrinsics::expf64(self) }
          .             }
          .         
          .             /// Returns `2^(self)`.
          .             ///
          .             /// # Unspecified precision
          .             ///
          .             /// The precision of this function is non-deterministic. This means it varies by platform, Rust version, and
          .             /// can even differ within the same execution from one invocation to the next.
-- line 484 ----------------------------------------
-- line 747 ----------------------------------------
          .             ///
          .             /// assert!(abs_difference < 1e-10);
          .             /// ```
          .             #[rustc_allow_incoherent_impl]
          .             #[must_use = "method returns a new number and does not mutate the original value"]
          .             #[stable(feature = "rust1", since = "1.0.0")]
          .             #[inline]
          .             pub fn cos(self) -> f64 {
  1,309,320 (0.0%)          unsafe { intrinsics::cosf64(self) }
          .             }
          .         
          .             /// Computes the tangent of a number (in radians).
          .             ///
          .             /// # Unspecified precision
          .             ///
          .             /// The precision of this function is non-deterministic. This means it varies by platform, Rust version, and
          .             /// can even differ within the same execution from one invocation to the next.
-- line 763 ----------------------------------------
-- line 902 ----------------------------------------
          .             /// assert!(abs_difference_1 < 1e-10);
          .             /// assert!(abs_difference_2 < 1e-10);
          .             /// ```
          .             #[rustc_allow_incoherent_impl]
          .             #[must_use = "method returns a new number and does not mutate the original value"]
          .             #[stable(feature = "rust1", since = "1.0.0")]
          .             #[inline]
          .             pub fn atan2(self, other: f64) -> f64 {
    333,162 (0.0%)          unsafe { cmath::atan2(self, other) }
          .             }
          .         
          .             /// Simultaneously computes the sine and cosine of the number, `x`. Returns
          .             /// `(sin(x), cos(x))`.
          .             ///
          .             /// # Unspecified precision
          .             ///
          .             /// The precision of this function is non-deterministic. This means it varies by platform, Rust version, and
-- line 918 ----------------------------------------

--------------------------------------------------------------------------------
-- Annotated source file: /home/kevin/01_rust/06_kuwahara/src/anisotropic/mod.rs
--------------------------------------------------------------------------------
Ir___________________ 

4,349,510,968 (12.3%)  <unknown (line 0)>

-- line 26 ----------------------------------------
            .              pub static FILTER_SMOOTHING_STD: f64 = 1.0;
            .              // Standard deviation for filter decay
            .              pub static FILTER_DECAY_STD: f64 = 3.0;
            .              // TODO: Something
            .              pub static SHARPNESS_COEFFICIENT: u64 = 8;
            .          }
            .          
            .          fn load_image(input_file: &PathBuf) -> ImageResult<Array3<f64>> {
           12  (0.0%)      let img = ImageReader::open(input_file)?.decode()?.to_rgb8();
            4  (0.0%)      let img = converters::image_to_ndarray(img);
            .              Ok(img)
            .          }
            .          
           28  (0.0%)  fn save_with_suffix<P, Container>(args: &Args, img: ImageBuffer<P, Container>, suffix: &str)
            .          where
            .              P: Pixel + PixelWithColorType,
            .              [P::Subpixel]: EncodableLayout,
            .              Container: Deref<Target = [P::Subpixel]>,
            .          {
           36  (0.0%)      let filepath = PathBuf::from(format!(
            .                  "{}/{}{}.png",
           12  (0.0%)          args.get_output_dir(),
            8  (0.0%)          args.get_file_stem(),
            .                  suffix
            .              ));
            .          
            8  (0.0%)      img.save(filepath).unwrap();
           16  (0.0%)  }
            .          
            .          /// The 2D Gaussian
            .          fn gaussian(x: f64, y: f64, std: f64) -> f64 {
        1,116  (0.0%)      let factor = 1.0 / (2.0 * PI * std.powi(2));
       13,204  (0.0%)      let exponent = -(x.powi(2) + y.powi(2)) / (2.0 * std.powi(2));
        3,306  (0.0%)      factor * exponent.exp()
            .          }
            .          /// Returns the partial derivative of the 2D Gaussian WRT x
            .          fn gaussian_derivative_x(x: f64, y: f64, std: f64) -> f64 {
            .              let factor = -x / std.powi(2);
           30  (0.0%)      factor * gaussian(x, y, std)
            .          }
            .          /// Returns the partial derivative of the 2D Gaussian WRT y
            .          fn gaussian_derivative_y(x: f64, y: f64, std: f64) -> f64 {
            .              gaussian_derivative_x(y, x, std)
            .          }
            .          
            .          fn gaussian_kernel(size: usize, std: f64) -> Array2<f64> {
          279  (0.0%)      let f = |x, y| gaussian(x, y, std);
            .              let kernel = converters::array2_from_fn(size, f);
            .              kernel
            .          }
            .          
            .          /// Represents the structure tensor with values:
            .          /// |e f|
            .          /// |f g|
            .          #[derive(Copy, Clone, Default, Debug, PartialEq)]
-- line 80 ----------------------------------------
-- line 86 ----------------------------------------
            .          
            .          impl StructureTensor {
            .              fn new(e: f64, f: f64, g: f64) -> Self {
            .                  StructureTensor { e, f, g }
            .              }
            .          
            .              fn approx_zero(&self) -> bool {
            .                  const THRESHOLD: f64 = 1e-9;
      988,782  (0.0%)          self.e.abs() < THRESHOLD && self.f.abs() < THRESHOLD && self.g.abs() < THRESHOLD
            .              }
            .          
            .              /// Finds the eigenvalue of a structure tensor. The largest eigenvalue is given first.
            .              fn get_eigenvalues(&self) -> (f64, f64) {
    1,309,320  (0.0%)          let determinant = (self.e - self.g).powi(2) + 4.0 * self.f.powi(2);
    1,963,980  (0.0%)          let eigenvalue1 = (self.e + self.g + determinant.sqrt()) / 2.0;
            .                  let eigenvalue2 = (self.e + self.g - determinant.sqrt()) / 2.0;
            .                  (eigenvalue1, eigenvalue2)
            .              }
            .          
    2,626,320  (0.0%)      fn into_anisotropy(self) -> Anisotropy {
            .                  if self.approx_zero() {
            .                      // This area is isotropic
        4,800  (0.0%)              let identity = array![[1.0, 0.0], [0.0, 1.0]];
        6,720  (0.0%)              return Anisotropy {
            .                          anisotropy: 0.0,
            .                          angle: 0.0,
            .                          transform: identity.clone(),
            .                      };
            .                  } else {
            .                      let (eigenvalue1, eigenvalue2) = self.get_eigenvalues();
            .          
      654,660  (0.0%)              let t = array![eigenvalue1 - self.e, -self.f];
            .                      let angle = t[1].atan2(t[0]);
            .          
            .                      let anisotropy = (eigenvalue1 - eigenvalue2) / (eigenvalue1 + eigenvalue2);
            .          
            .                      const ALPHA: f64 = 1.0;
    1,636,650  (0.0%)              let scale = array![
      654,660  (0.0%)                  [ALPHA / (ALPHA + anisotropy), 0.],
            .                          [0., (ALPHA + anisotropy) / ALPHA]
            .                      ];
            .          
    1,963,980  (0.0%)              let rotation = array![[angle.cos(), -angle.sin()], [angle.sin(), angle.cos()]];
            .          
            .                      let transform = scale.dot(&rotation);
            .          
    3,600,630  (0.0%)              Anisotropy {
            .                          anisotropy,
            .                          angle,
            .                          transform,
            .                      }
            .                  }
    2,954,610  (0.0%)      }
            .          }
            .          
            .          /// Represents the partial derivatives of each of the three channels with
            .          /// respect to x and y.  Dimensions are [channels, height, width]
            .          struct Gradients {
            .              x: Array3<f64>,
            .              y: Array3<f64>,
            .          }
            .          
            .          impl Gradients {
            .              fn new(img: &Array3<f64>, kernel_size: usize) -> Self {
            .                  /// Normalise kernels so that the sum of absolute values is equal to 2.
            .                  /// The sum is arbritary, but we simply want to avoid floating point error
            .                  /// by using kernels that have values that are too small.
           12  (0.0%)          fn normalise_kernel(arr: Array2<f64>) -> Array2<f64> {
            4  (0.0%)              let sum = arr.map(|x| x.abs()).sum();
           22  (0.0%)              arr / sum
            8  (0.0%)          }
            .          
            .                  let f_x = |x, y| gaussian_derivative_x(x, y, consts::PARTIAL_DERIVATIVE_STD);
            .                  let f_y = |x, y| gaussian_derivative_y(x, y, consts::PARTIAL_DERIVATIVE_STD);
            .          
            1  (0.0%)          let x_kernel = normalise_kernel(converters::array2_from_fn(kernel_size, f_x));
            1  (0.0%)          let y_kernel = normalise_kernel(converters::array2_from_fn(kernel_size, f_y));
            .          
            .                  /// Applies the kernel convolution to each of the three RGB channels
            .                  /// independently
           18  (0.0%)          fn apply_convolution_on_channels(img: &Array3<f64>, kernel: &Array2<f64>) -> Array3<f64> {
            .                      let gradients = img
            .                          .clone()
            .                          .axis_iter(Axis(0))
            .                          .map(|channel| {
           36  (0.0%)                      channel
           22  (0.0%)                          .conv(kernel, ConvMode::Same, PaddingMode::Replicate)
            .                                  .unwrap()
            .                          })
            .                          .collect::<Vec<_>>();
            .                      let gradient_views = gradients.iter().map(|x| x.view()).collect::<Vec<_>>();
            8  (0.0%)              stack(Axis(0), &gradient_views).unwrap()
           16  (0.0%)          }
            .          
            1  (0.0%)          let x_gradients = apply_convolution_on_channels(img, &x_kernel);
            1  (0.0%)          let y_gradients = apply_convolution_on_channels(img, &y_kernel);
            .          
           10  (0.0%)          Gradients {
           10  (0.0%)              x: x_gradients,
            .                      y: y_gradients,
            .                  }
            .              }
            .          
            .              /// For each pixel, computes the corresponding structure tensor
            .              fn into_structure_tensors(self) -> Array2<StructureTensor> {
            .                  #[inline]
            .                  fn dot_product(a1: &Array3<f64>, a2: &Array3<f64>) -> Array2<f64> {
           30  (0.0%)              (a1 * a2).sum_axis(Axis(0))
            .                  }
            .          
            .                  let fxx = dot_product(&self.x, &self.x);
            1  (0.0%)          let fxy = dot_product(&self.x, &self.y);
            .                  let fyy = dot_product(&self.y, &self.y);
            .          
            8  (0.0%)          Zip::from(&fxx)
            .                      .and(&fxy)
            .                      .and(&fyy)
            .                      .map_collect(|&e, &f, &g| StructureTensor::new(e, f, g))
            .              }
            .          }
            .          
            .          fn smooth_structure_tensors(
            .              structure_tensors: &Array2<StructureTensor>,
-- line 208 ----------------------------------------
-- line 212 ----------------------------------------
            .          
            .              let es = structure_tensors.mapv(|tensor| tensor.e);
            .              let fs = structure_tensors.mapv(|tensor| tensor.f);
            .              let gs = structure_tensors.mapv(|tensor| tensor.g);
            .          
            .              let kernel_size = consts::TENSOR_SMOOTHING_STD as usize * 3 * 2 + 1;
            .              let smoothing_kernel = gaussian_kernel(kernel_size, consts::TENSOR_SMOOTHING_STD);
            .          
           14  (0.0%)      let es = es
            2  (0.0%)          .conv(&smoothing_kernel, ConvMode::Same, PaddingMode::Replicate)
            .                  .unwrap();
           14  (0.0%)      let fs = fs
            .                  .conv(&smoothing_kernel, ConvMode::Same, PaddingMode::Replicate)
            .                  .unwrap();
            1  (0.0%)      let gs = gs
            .                  .conv(&smoothing_kernel, ConvMode::Same, PaddingMode::Replicate)
            .                  .unwrap();
            .          
            7  (0.0%)      let structure_tensors = Zip::from(&es)
            .                  .and(&fs)
            .                  .and(&gs)
            .                  .map_collect(|&e, &f, &g| StructureTensor::new(e, f, g));
            .              structure_tensors
            .          }
            .          
            .          #[derive(Clone, Debug)]
            .          struct Anisotropy {
-- line 238 ----------------------------------------
-- line 247 ----------------------------------------
            .          /// `i`` reprsent which segment of the disc we are considering.
            .          fn get_disc_space_weighting(i: usize) -> Array2<f64> {
            .              assert!(i < consts::NUM_SECTORS);
            .          
            .              // Calculate the charateristic function
            .              const SIZE: usize = 27;
            .              let characteristic = converters::array2_from_fn(SIZE, |x, y| {
            .                  let half_size = (SIZE / 2) as f64;
       26,814  (0.0%)          let inside_circle = x.powi(2) + y.powi(2) <= (half_size).powi(2);
            .          
            .                  let angle = y.atan2(x);
       75,816  (0.0%)          let lower = ((2.0 * i as f64 - 1.0) * PI) / consts::NUM_SECTORS as f64;
       17,496  (0.0%)          let upper = ((2.0 * i as f64 + 1.0) * PI) / consts::NUM_SECTORS as f64;
            .          
       12,827  (0.0%)          let inside_segment1 = lower < angle && angle <= upper;
       27,997  (0.0%)          let inside_segment2 = lower < angle + 2.0 * PI && angle + 2.0 * PI <= upper;
        1,163  (0.0%)          let inside_segment = inside_segment1 || inside_segment2;
            .          
       11,242  (0.0%)          if inside_circle && inside_segment {
            .                      1.0
            .                  } else {
            .                      0.0
            .                  }
            .              });
            .          
            .              // Smooth out the characteristic function
            .              let smoothing_diameter = (consts::FILTER_SMOOTHING_STD).ceil() as usize * 3 * 2 + 1;
            .              let smoothing_kernel = gaussian_kernel(smoothing_diameter, consts::FILTER_SMOOTHING_STD);
           48  (0.0%)      let weights = characteristic
           16  (0.0%)          .conv(&smoothing_kernel, ConvMode::Same, PaddingMode::Zeros)
            .                  .unwrap();
            .          
            .              // Ensure the characteristic function decays the further from (0,0) we go
            .              let decay_kernel = gaussian_kernel(SIZE, consts::FILTER_DECAY_STD);
          120  (0.0%)      let weights = weights * decay_kernel;
            .          
            .              weights
            .          }
            .          
            .          /// A quick and dirty function that just truncates the floating points of the
            .          /// coordinates in the point to get the array value.
            .          fn query_point_in_array2(point: &[f64; 2], arr: &Array2<f64>) -> f64 {
            .              let (height, width) = arr.dim();
-- line 289 ----------------------------------------
-- line 310 ----------------------------------------
            .              var: Array2<f64>,
            .          }
            .          
            .          #[inline(always)]
            .          fn transform_point(point: &[f64; 2], transform: &Array2<f64>) -> [f64; 2] {
            .              let mut result = [0.0, 0.0];
            .              for i in 0..2 {
            .                  for j in 0..2 {
4,916,745,603 (13.9%)              result[i] += transform[[i, j]] * point[j];
            .                  }
            .              }
            .              result
            .          }
            .          
            .          impl PixelStatistics {
            .              fn new(
            .                  x0: usize,
-- line 326 ----------------------------------------
-- line 331 ----------------------------------------
            .              ) -> Self {
            .                  const WINDOW_SIZE: isize = 27 * 2;
            .                  const HALF_WINDOW_SIZE: isize = WINDOW_SIZE / 2;
            .          
            .                  let (_, height, width) = img.dim();
            .          
            .                  let mut mean: Array2<f64> = Array2::zeros((consts::NUM_SECTORS, 3));
            .                  let mut var: Array2<f64> = Array2::zeros((consts::NUM_SECTORS, 3));
      492,435  (0.0%)          let mut divisor: Array1<f64> = Array1::zeros(consts::NUM_SECTORS);
            .          
    8,863,830  (0.0%)          for y in -HALF_WINDOW_SIZE..HALF_WINDOW_SIZE {
            .                      for x in -HALF_WINDOW_SIZE..HALF_WINDOW_SIZE {
            .                          // (y1, x1) is the actual position of the pixel
            .                          let y1 = y + y0 as isize;
            .                          let x1 = x + x0 as isize;
  920,683,260  (2.6%)                  if !(0..height as isize).contains(&y1) || !(0..width as isize).contains(&x1) {
            .                              continue;
            .                          }
            .                          // (y1, x1) are definitely inside the image, so  it's safe to
            .                          // type-cast them to usizes
            .                          let y1 = y1 as usize;
            .                          let x1 = x1 as usize;
            .          
  893,953,746  (2.5%)                  let offset = [x as f64, y as f64];
            .                          let disc_offset = transform_point(&offset, &anisotropy.transform);
            .          
            .                          // Optimisation: We don't bother to calculate the weight for this
            .                          // pixel if we know that it is outside of the effective radius
            .                          // of the sector kernels.
            .                          static EFFECTIVE_RADIUS_SQUARED: f64 =
            .                              (consts::FILTER_DECAY_STD * 3.0) * (consts::FILTER_DECAY_STD * 3.0);
2,234,884,365  (6.3%)                  if disc_offset[0].powi(2) + disc_offset[1].powi(2) > EFFECTIVE_RADIUS_SQUARED {
            .                              continue;
            .                          }
            .          
            .                          for i in 0..consts::NUM_SECTORS {
            .                              // let weight = query_point_in_array2(&disc_offset, &disc_weights[i]);
            .                              let weight =
  326,079,968  (0.9%)                          disc_weights[i][[disc_offset[0] as usize, disc_offset[1] as usize]];
            .          
            .                              for c in 0..3 {
2,934,719,712  (8.3%)                          mean[[i, c]] += weight * img[[c, y1, x1]];
5,869,439,424 (16.6%)                          var[[i, c]] += weight * img[[c, y1, x1]] * img[[c, y1, x1]];
            .                              }
  652,159,936  (1.8%)                      divisor[[i]] += weight;
            .                          }
            .                      }
            .                  }
            .          
            .                  // Normalise the mean and variance by the sum of weights.
            .                  for i in 0..consts::NUM_SECTORS {
            .                      for c in 0..3 {
   13,295,745  (0.0%)                  mean[[i, c]] /= divisor[[i]];
            .                      }
            .                  }
            .                  for i in 0..consts::NUM_SECTORS {
            .                      for c in 0..3 {
    7,878,960  (0.0%)                  var[[i, c]] /= divisor[[i]];
   11,818,440  (0.0%)                  var[[i, c]] -= mean[[i, c]] * mean[[i, c]];
            .                      }
            .                  }
            .          
            .                  PixelStatistics { mean, var }
            .              }
            .          }
            .          
            .          /// Calculates the final pixel value of (x,y) in the image as a Array1
-- line 397 ----------------------------------------
-- line 399 ----------------------------------------
            .          fn calculate_pixel_value(
            .              x: usize,
            .              y: usize,
            .              img: &Array3<f64>,
            .              anisotropy: &Anisotropy,
            .              disc_weights: &DiscWeights,
            .          ) -> Array1<f64> {
            .              // Calculate the mean and variance of each of the sectors in our image.
    1,313,160  (0.0%)      let PixelStatistics { mean, var } = PixelStatistics::new(x, y, img, &anisotropy, disc_weights);
            .          
      492,435  (0.0%)      let mut output: Array1<f64> = Array1::zeros(3);
      492,435  (0.0%)      let mut divisor: Array1<f64> = Array1::zeros(3);
            .          
            .              let std = var.sqrt();
            .              for i in 0..consts::NUM_SECTORS {
            .                  // We multiply the norm by 255, since the paper expects pixel values
            .                  // between [0, 255], whereas we currently use [0, 1]
    7,878,960  (0.0%)          let norm = (std[[i, 0]].powi(2) + std[[i, 1]].powi(2) + std[[i, 2]].powi(2)).sqrt() * 255.0;
    2,626,320  (0.0%)          let weighting_factor = 1.0 / (1.0 + norm.powi(consts::SHARPNESS_COEFFICIENT as i32));
            .          
            .                  for c in 0..3 {
   13,131,600  (0.0%)              output[c] += weighting_factor * mean[[i, c]];
    7,878,960  (0.0%)              divisor[c] += weighting_factor;
            .                  }
            .              }
            .          
    2,790,465  (0.0%)      output / divisor
            .          }
            .          
            .          fn apply_kuwahara_filter(
            .              img: &Array3<f64>,
            .              anisotropy: &Array2<Anisotropy>,
            .              disc_weights: &DiscWeights,
            .          ) -> Array3<f64> {
            .              let (_, height, width) = img.dim();
            .          
            6  (0.0%)      let mut output = Array3::zeros((3, height, width));
            .          
            4  (0.0%)      let pb = ProgressBar::new((height * width) as u64);
           12  (0.0%)      pb.set_style(ProgressStyle::default_bar()
            .                  .template("{spinner:.green} [{elapsed_precise}] [{bar:50.cyan/blue}] {pos}/{len} ({percent}%) ETA: {eta}")
            .                  .unwrap()
            .                  .progress_chars("#>-"));
            .          
            .              for y in 0..height {
            .                  for x in 0..width {
            .                      let rgb = calculate_pixel_value(x, y, img, &anisotropy[[y, x]], disc_weights);
            .                      for c in 0..3 {
    1,149,015  (0.0%)                  output[[c, y, x]] = rgb[c];
            .                      }
      492,435  (0.0%)              pb.inc(1);
            .                  }
            .              }
            .          
            6  (0.0%)      pb.finish_with_message("done!");
            .          
           11  (0.0%)      output
            1  (0.0%)  }
            .          
            9  (0.0%)  pub fn run(args: &Args) -> ImageResult<()> {
            7  (0.0%)      let img = load_image(&args.input)?;
            .          
            .              let gradients = Gradients::new(&img, consts::PARTIAL_DERIVATIVE_KERNEL_SIZE);
            .          
            3  (0.0%)      if args.intermediate_results {
            1  (0.0%)          let img_x_grad = converters::ndarray_to_rgbimage(converters::normalise(&gradients.x));
            1  (0.0%)          let img_y_grad = converters::ndarray_to_rgbimage(converters::normalise(&gradients.y));
            .          
            5  (0.0%)          save_with_suffix(args, img_x_grad, "_x_grad");
            5  (0.0%)          save_with_suffix(args, img_y_grad, "_y_grad");
            .              }
            .          
            3  (0.0%)      let structure_tensors = gradients.into_structure_tensors();
            .          
            3  (0.0%)      if args.intermediate_results {
      984,871  (0.0%)          let anisotropy = structure_tensors.map(|tensor| tensor.into_anisotropy());
            .          
            .                  let strength = anisotropy.mapv(|anisotropy| anisotropy.anisotropy);
      656,580  (0.0%)          let angle = anisotropy.mapv(|anisotropy| (anisotropy.angle + (PI / 2.0)) / PI);
            .                  // The elements of angle are normalised from [-PI/2, PI/2] to [0,1]
            .          
           12  (0.0%)          let img_anisotropy = converters::angle_and_strength_to_rgbimage(angle, strength);
            .          
            5  (0.0%)          save_with_suffix(args, img_anisotropy, "_unsmoothed_anisotropy");
            .              }
            .          
            .              let structure_tensors = smooth_structure_tensors(&structure_tensors);
            .          
            1  (0.0%)      let anisotropy = structure_tensors.map(|tensor| tensor.into_anisotropy());
            .          
            2  (0.0%)      if args.intermediate_results {
            .                  let strength = anisotropy.mapv(|anisotropy| anisotropy.anisotropy);
            .                  let angle = anisotropy.mapv(|anisotropy| (anisotropy.angle + (PI / 2.0)) / PI);
            .                  // The elements of angle are normalised from [-PI/2, PI/2] to [0,1]
            .          
           12  (0.0%)          let img_anisotropy = converters::angle_and_strength_to_rgbimage(angle, strength);
            .          
            5  (0.0%)          save_with_suffix(args, img_anisotropy, "_anisotropy");
            .              }
            .          
            .              let disc_weights: DiscWeights = std::array::from_fn(|i| get_disc_space_weighting(i));
            .          
            .              let output = apply_kuwahara_filter(&img, &anisotropy, &disc_weights);
            .          
            1  (0.0%)      let output_img = converters::ndarray_to_rgbimage(output);
            .          
           16  (0.0%)      let output_file = format!("{}/{}", args.get_output_dir(), args.get_output_filename());
            .              output_img.save(&output_file)?;
            .          
            2  (0.0%)      Ok(())
            9  (0.0%)  }

--------------------------------------------------------------------------------
-- Annotation summary
--------------------------------------------------------------------------------
Ir____________________ 

29,218,100,046 (82.7%)    annotated: files known & above threshold & readable, line numbers known
 4,743,829,008 (13.4%)    annotated: files known & above threshold & readable, line numbers unknown
             0          unannotated: files known & above threshold & two or more non-identical
   624,676,790  (1.8%)  unannotated: files known & above threshold & unreadable 
   747,781,740  (2.1%)  unannotated: files known & below threshold
     6,218,050  (0.0%)  unannotated: files unknown


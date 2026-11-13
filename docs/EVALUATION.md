# Evaluation protocol

Implemented in`evaluation/geometry_metrics.py`,`image_metrics.py`and`evaluation.py`.

## Geometry

-1356 chair test objects,input view0;10000 sampled prediction points and
  a fixed10000-point GT subset from100000. Seeds derive from object ID/view,
  shared between all groups.
- Camera coordinate normalization uses the GT test camera. Both before/after
  ICP therefore use GT information,not a wholly GT-free pose test.
- Before ICP: predicted/GT camera coordinate transform. After ICP: additionally
  AABB centering/max-extent normalization and100-iteration gradient ICP,
  LR.01,rotation,translation and **anisotropic scale**.
- CD-L1×10=`10*.5*(mean Euclidean NN pred→GT + mean Euclidean NN GT→pred)`.
  It is not squared Chamfer distance.
- F-score thresholds1%/2% of the **full GT AABB diagonal**,before subsampling.
  P=pred→GT,R=GT→pred,`F=2PR/(P+R)`. Average object F-scores,not the
  harmonic mean of averaged P/R.
- Normal score:symmetric nearest-point absolute cosine. After anisotropic ICP,
  apply inverse transpose and renormalize. Absolute cosine cannot detect flips.
- Mesh quality uses raw pre-ICP geometry,centered/divided by AABB diagonal.
  Triangle quality=`4*sqrt(3)*area/sum(edge_length**2)`;
  normalized near-zero area threshold1e-10. Also report minimum angle,
  quality percentiles and edge connectivity. No self-intersection detector.
  Fixed shared topology is not evidence of better geometry or no self-intersections.

## Images

Predicted input camera **before ICP**,64×64 full RGB,no mask. Clip prediction
to[0,1];save raw range and clipping fraction for every object/group.

- MSE:all pixels/channels.
- PSNR:`-10log10(max(MSE,1e-12))`,120dB cap with explicit floor-event count.
  Average per-object PSNR;not PSNR of average MSE.
- SSIM:11×11 Gaussian,sigma1.5,data range1,K1.01/K2.03,population covariance,
  valid spatial centers,RGB-channel mean.
- LPIPS is not implemented;no learned metric asset downloads occur at runtime.

## Pilot/cache/summary

Pilot:8 evenly spaced test-list positions ×7 final models=56 rows.
Full stage:1356/group. Identical pilot results may be reused. Caches bind
source/config/weights/data hashes,seed and object ID;reuse is not an
independent repeated measurement. Released full evaluation has9436 new rows
+56 pilot rows=9492 distinct group/object entries.

Each group writes`final_GROUP.json`. The summary script rejects missing/failed
groups,wrong object counts,inconsistent pairing or a different fingerprint.
It recomputes means from rows. Before/after ICP scores remain separate;do not
select alignment variants per model after inspecting their scores.

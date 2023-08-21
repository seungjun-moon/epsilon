<p align="center">

  <h1 align="center">EPSiloN: Efficient Point Sampling for Reducing Inference Latency of NeRF-based Avatar Video Generation
 </h1>
  <div align="center">
    <img src="assets/test.gif" alt="test" width="30%">
  </div>
</p>


  <h1 align="center">Abstract
 </h1>
  <div align="center">
   The rapid advancement of neural radiance fields (NeRF) combined with the existing statistical modeling, \eg, SMPL, has paved the way to generate animatable human avatars from a monocular video.
To train NeRF in canonical space, we need to deform the dynamic movement of humans in the video.
Unfortunately, backward deformation, which generates more accurate avatars than forward deformation, needs high computational costs on each sampled point.
We observe that since most of the sampled points are located in empty space, they do not affect the generation quality but result in inference latency with deformation.
In light of this observation, we propose EPSiloN, a NeRF-based avatar video generation scheme with novel efficient point sampling strategies for reducing inference latency.
In EPSiloN, we propose two methods to omit empty points at rendering; empty ray omission (ERO) and empty interval omission (EIO).
In ERO, we wipe out rays that progress through the empty space.
Then, EIO narrows down the sampling interval on the ray, which wipes out the region that cannot be occupied by either clothes or mesh.
The delicate sampling scheme of EPSiloN enables not only great computational cost reduction \sj{occurred by deformation} but also a single-stage inference without hierarchical sampling.
Compared to existing methods, EPSiloN maintains the generation quality only with 7\% of sampled points and achieves around $15\times$ lower inference latency.
  </div>
</p>

## Method

write down something.

Input structure image.



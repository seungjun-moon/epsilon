  <h1 align="center">EPSiloN: Efficient Point Sampling for Reducing Inference Latency of NeRF-based Avatar Video Generation
 </h1>
  <div align="center">
    <img src="assets/teaser.gif" alt="teaser" width="100%">
  </div>
</p>


  <h1 align="center">Abstract
 </h1>
  <div align="center">
   The rapid advancement of neural radiance fields (NeRF) combined with the existing statistical modeling, e.g., SMPL, has paved the way to generate animatable human avatars from a monocular video.
To train NeRF in canonical space, we need to deform the dynamic movement of humans in the video.
Unfortunately, backward deformation, which generates more accurate avatars than forward deformation, needs high computational costs on each sampled point.
We observe that since most of the sampled points are located in empty space, they do not affect the generation quality but result in inference latency with deformation.
In light of this observation, we propose EPSiloN, a NeRF-based avatar video generation scheme with novel efficient point sampling strategies for reducing inference latency.
In EPSiloN, we propose two methods to omit empty points at rendering; empty ray omission (ERO) and empty interval omission (EIO).
In ERO, we wipe out rays that progress through the empty space.
Then, EIO narrows down the sampling interval on the ray, which wipes out the region that cannot be occupied by either clothes or mesh.
The delicate sampling scheme of EPSiloN enables not only great computational cost reduction occurred by deformation but also a single-stage inference without hierarchical sampling.
Compared to existing methods, EPSiloN maintains the generation quality only with 7% of sampled points and achieves around 15× lower inference latency.
  </div>
</p>

## Method Overview

<div align="center">
  <img src="assets/structure.png" alt="visualize" width="100%">
</div>

EPSiloN proposes an efficient point sampling strategies in the avatar generation based on the monocular video, which results in comparable results to the state-of-the-art models while reducing the inference latency significantly.

## Results

<div align="center">
  <img src="assets/visualize.gif" alt="visualize" width="100%">
</div>

Along with the input image, we visualize the reconstructed image with RGB representation and mesh representation. Moreover, we show the depth image of the mesh, and obtained $T_n$ and $T_f$ together for the better understanding. We visualize (ground truth, rendered image, mesh representation, $D(M')$, $T_n$, $T_f$), respectively, in the figure above. $T_n$ and $T_f$ indeed find the appropriate interval using $D(M')$.


<div align="center">
  <img src="assets/new.gif" alt="visualize" width="100%">
</div>

In the figure above, we visualize the novel view generation and novel pose generation of four subjects in People Snapshot datasets. While achieving 15 times faster rendering speed compared to the baseline, our model robustly generates the novel contents of given subjects.
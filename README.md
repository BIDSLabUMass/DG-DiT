# DG-DiT: Dual-Branch Gating Diffusion Transformer for Multi-Tracer and Multi-Scanner Brain PET Image Denoising
Ziyuan Zhou, Fan Yang, Tzu-An Song, Bowen Lei, Yubo Zhang, and Joyita Dutta

Abstract——Diffusion probabilistic models (DPMs) have been demonstrated to be effective for denoising positron emission tomography (PET) images due to their ability to model complex data distributions. However, limitations in efficiency, accuracy, and generalizability remain open challenges in this area. In PET denoising, where high fidelity to the ground truth is critical, DPMs often require a large number of iterations and tend to offer limited quantitative accuracy. Moreover, traditional DPMs struggle to model variabilities in the data distribution arising from the use of multiple scanners and tracers. To address these issues, we propose a dual-branch gating diffusion transformer (DG-DiT) network for multi-tracer and multi-scanner PET denoising. The proposed DG-DiT exploits the strong distribution modeling capabilities of a diffusion transformer (DiT) to learn prior knowledge from a compact and regularized latent space. The design of the latent space enables efficient few-step diffusion. In addition, an image restoration transformer (IRT) model is employed for generating the final denoised image. The DiT backbone and the IRT both utilize a dual-branch gating mechanism to efficiently fuse information from multiple inputs. We conducted extensive experiments on multi-tracer and multi-scanner datasets. The results demonstrate that the proposed DG-DiT model achieves the highest quantitative accuracy across every scanner and tracer, with a PSNR improvement of up to 0.2 dB compared to several state-of-the-art deep learning models. Contrast-to-noise ratio evaluation shows that the proposed model is able to recover contrast in small and critical brain regions while effectively reducing noise. This suggests that the proposed DG-DiT model can consistently deliver superior denoising performance.

## Prerequisite
- Python 3.12.11
- Pytorch 2.4.1
- numpy 2.1.3
- scipy 1.16.0
- NVIDIA GPU
- CUDA 12.2
- CuDNN 9.1.0

## Dataset
Alzheimer’s Disease Neuroimaging Initiative (ADNI) (Clinical Database): http://adni.loni.usc.edu/

## Citation

## Condition of Use
If you find our work helpful for your research, please kindly cite our paper.

## UMASS_Amherst_BIDSLab

Biomedical Imaging & Data Science Laboratory

Lab's website: http://www.bidslab.org/index.html 

Email: ziyuanzhou(at)umass.edu, jdutta@umass.edu

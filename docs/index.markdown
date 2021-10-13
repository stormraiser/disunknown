---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: default
---

## **Abstract**
Disentangling data into interpretable and independent factors is critical
for controllable generation tasks. With the availability of labeled data,
supervision can help enforce the separation of specific factors as expected.
However, it is often expensive or even impossible to label every single
factor to achieve fully-supervised disentanglement. In this paper, we adopt
a general setting where all factors that are hard to label or identify are
encapsulated as a single unknown factor. Under this setting, we propose a
flexible weakly-supervised multi-factor disentanglement framework
*DisUnknown*, which *Dis*tills *Unknown* factors
for enabling multi-conditional generation regarding both labeled and
unknown factors. Specifically, a two-stage training approach is adopted
to first disentangle the unknown factor with an effective and robust
training method, and then train the final generator with the proper
disentanglement of all labeled factors utilizing the unknown distillation.
To demonstrate the generalization capacity and scalability of our method,
we evaluate it on multiple benchmark datasets qualitatively and
quantitatively and further apply it to various real-world applications
on complicated datasets.

## **Presentation Video**
<div style="margin:auto;max-width:1152px;aspect-ratio:16/9;">
<iframe src="https://www.youtube.com/embed/jEza9IKsANE" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen style="width:100%;height:100%;"></iframe>
</div>

## **Architecture**
<center><img src="figures/train_stage.jpg" width="1152"></center>

## **Application 1:** Anime Style Transfer.
Disentanglement of the artists' identity *(labeled)* and the content 
*(unknown)* in anime images,
and its application in anime style transfer.

**[Web demo](/disunknown/nekonetworks/)**

<center><img src="figures/app_anime_style_transfer.jpg" width="1152" >
<figcaption>The left column: content input; Top row: style input.</figcaption></center>

## **Application 2:** Portrait Relighting
Disentanglement of the lighting *(labeled)* and the remaining content 
*(unknown)* in portrait images,
and its application for portrait relighting.
<div style="margin:auto;max-width:1152px;aspect-ratio:16/9;">
<iframe src="https://www.youtube.com/embed/UOS5b1x9kzI" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen style="width:100%;height:100%;"></iframe>
</div>

## **Application 3:** Landmark-Based Face Reenactment
Disentanglement of the identity *(labeled)*, the pose *(labeled)* 
and the expression *(unknown)* in 2D face landmarks,
and its application in face reenactment.
<div style="margin:auto;max-width:1152px;aspect-ratio:16/9;">
<iframe src="https://www.youtube.com/embed/BN6LLJpw-v0" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen style="width:100%;height:100%;"></iframe>
</div>

## **Application 4:** Skeleton-Based Body Motion Retargeting
Disentanglement of the identity *(labeled)*, the view *(labeled)* 
and the motion *(unknown)* in 2D skeletons,
and its application in body motion retargeting.
<div style="margin:auto;max-width:1152px;aspect-ratio:16/9;">
<iframe src="https://www.youtube.com/embed/FIUzmOE2_2w" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen style="width:100%;height:100%;"></iframe>
</div>

## **BibTeX**
```
@inproceedings{xiang2021disunknown,
  title={DisUnknown: Distilling Unknown Factors for Disentanglement Learning},
  author={Xiang, Sitao and Gu, Yuming and Xiang, Pengda and Chai, Menglei and Li, Hao and Zhao, Yajie and He, Mingming},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={14810--14819},
  year={2021}
}
```

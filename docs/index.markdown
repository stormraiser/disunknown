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
<iframe width="1152" height="648" style="margin: auto; display: block;"
src="https://www.youtube.com/embed/jEza9IKsANE">
</iframe>

## **Architecture**
<center><img src="figures/train_stage.jpg" width="1152"></center>

## **Application 1:** Anime Style Transfer.
Disentanglement of the artists' identity (*known factor*) and the content 
(*unknown factor*) in anime images,
and its application in anime style transfer.

**[Web demo](/disunknown/nekonetworks/)**

<center><img src="figures/app_anime_style_transfer.jpg" width="1152" >
<figcaption>The left column: content input; Top row: style input.</figcaption></center>

## **Application 2:** Portrait Relighting
Disentanglement of the lighting (*known factor*) and the remaining content 
(unknown factor) in portrait images,
and its application for portrait relighting.
<iframe width="1152" height="648" style="margin: auto; display: block;"
src="https://www.youtube.com/embed/A0ymux6aciw">
</iframe>

## **Application 3:** Landmark-Based Face Reenactment
Disentanglement of the identity (*known factor*), the pose (*known factor*) 
and the expression (*unknown factor*) in 2D face landmarks,
and its application in face reenactment.
<iframe width="1152" height="648" style="margin: auto; display: block;"
src="https://www.youtube.com/embed/BN6LLJpw-v0">
</iframe>

## **Application 4:** Skeleton-Based Body Motion Retargeting
Disentanglement of the identity (*known factor*), the view (*known factor*) 
and the motion (*unknown factor*) in 2D skeletons,
and its application in body motion retargeting.
<iframe width="1152" height="648" style="margin: auto; display: block;"
src="https://www.youtube.com/embed/FIUzmOE2_2w">
</iframe>

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

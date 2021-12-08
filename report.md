<style>
.reveal {
  font-size: 24px;
}
</style>

# Image Watercolorization

B08902107 蘇柏瑄
B08902109 邱俊茗

---

## Overview

----

- We implemented a system for processing a static image into the style of watercolor paintings.
    - Mostly following the method from [1]
    - Some details cannot be found in the paper, thus we also referenced the implementation in [6].
    - We used python as the programming language, and used OpenCV's implementation for some well known algorithm.

----

- Some distinct features of watercolor paintings:
    - Detail abstraction
    - Hand tremor
    - Wet-in-wet technique
    - Edge darkening
    - Pigment density variation
    - Granulation

----

![](https://www.researchgate.net/profile/Kurt-Fleischer/publication/2917328/figure/fig1/AS:669955381002246@1536741218458/Real-watercolor-effects-drybrush-a-edge-darkening-b-backruns-c-granulation-d.jpg)

- Second from the left is "Edge darkening".
- Fifth from the left is "Wet-in-wet technique".

(Figure borrowed from [8])

----

### Pipeline

1. Abstraction
    - Salient region detection
    - Distance field
    - Image segmentation
    - Mean filtering
2. Hand tremor
    - Type 1 (without gaps and overlaps)
    - Type 2 (with gaps and overlaps)

----

3. Wet-in-wet effect
4. Other effects
    - Edge darkening
    - Pigment density variation
    - Granulation

----

There is a step "Color Adjustment" in [1] which we skipped because of our lack of a dataset of real watercolor paintings.

---

## Image input

![](https://i.imgur.com/WhI3fzw.jpg =300x400)

---

## Abstraction

----

### Salient region detection

- Follow the method of [2] to get the saliency map.
- Use Otsu's algorithm [3] to adaptively threshold the saliency map and get a binary map.

----

![](https://i.imgur.com/fhIV4KL.png =300x400) ![](https://i.imgur.com/BrjgtAa.png =300x400)

----

### Distance field

- From the binary map of the salient region, for every pixel not in the salient region, find the closest salient pixel.
- Normalize the distance to be between [0, 1].
    - Edge of image: 1
    - Neighbor of a salient pixel: 0
- Blur the distance field to reduce artifact.


----

![](https://i.imgur.com/Hjm10Mp.png =300x400)

----

### Image segmentation

- Use [4] [5] to segmentate the image into several regions.

----

![](https://i.imgur.com/Iv0XJEo.png =300x400)

----

### Mean filtering

- For every pixel $p$
    - Let $d$ be the distace field at $p$
    - If $p$ in salient region:
        - Only consider pixel in the same region within kernel of size 5.
    - Else:
        - Also consider pixel in the diffent region within kernel of size $\operatorname{clamp}(5 \cdot 2 \cdot (d + 0.3), 4, 9)$, whose color difference to $p$ is less than $0.3d$.

----

- Apply linear transformation to colors ($f \gets \frac{2}{3}f + \frac{1}{4}$) to get a more "pale and desaturated" look.

----

![](https://i.imgur.com/4p90mZp.png =300x400) ![](https://i.imgur.com/x1JKtA5.png =300x400)

---

## Hand tremor

----

- For all pixel $p$ at boundary of regions:
    - Let $\mathbf{g}_c$ be the averaged color gradient.
    - Let $\delta_h$ be the difference of hue between the adjacent regions.
    - If $\lvert \mathbf{g}_{cx} \rvert, \lvert \mathbf{g}_{cy} \rvert \geq 0.1$ and either (a) inside salient regions and $\delta_h < 20^\circ$, or (b) outside salient regions and $\delta_h < 90^\circ$:
        - Mark $p$ as wet-in-wet.
    - Else if $\delta_h < 40^\circ$:
        - Mark $p$ as Type 1.
    - Else:
        - Mark $p$ as Type 2.

----

### Type 1 (without gaps and overlaps)

- Get binary map of Type 1 pixels and blur it to get a displacement weight field $\mathbf{W}$.
- Use two perlin noise texture $\mathbf{N}_1, \mathbf{N}_2$.
- Displace those pixels $p$ around Type 1 pixels ($\mathbf{W}(p) > \epsilon$):
- $f(p) \leftarrow f(p_x + \mathbf{W}(p) \mathbf{N}_1(p), p_y + \mathbf{W}(p) \mathbf{N}_2(p))$

----

### Type 2 (with gaps and overlaps)

- Get binary map of Type 2 pixels and blur it to get a displacement weight field $\mathbf{W}$.
- For every region $R_i$:
    - Use two private perlin noise texture $\mathbf{N}_1, \mathbf{N}_2$.
    - Displace those pixels $p$ around Type 2 pixels:
    - $f_i(p) \leftarrow f(p_x + \mathbf{W}(p) \mathbf{N}_1(p), p_y + \mathbf{W}(p) \mathbf{N}_2(p))$
- Combine all $f_i$.

----

![](https://i.imgur.com/zBQg2TP.png =300x400)

---

## Wet-in-wet effect

----

- For every Wet-in-wet pixel $p$:
    - Let the intensity gradient be $\mathbf{g}$.
    - Let $r$ be a real number $3 < r < 5$ randomly picked.
    - Dot a pixel with color of $p$ at $p + r \mathbf{g}$:
    - $f(p + r \mathbf{g}) \gets f(p)$
    - For pixels in a elliptic window (long side aligned with $\mathbf{g}$), blur them with a elliptic kernel.

----

<!-- ![](https://i.imgur.com/EOaFSRV.png =300x400) -->
![](https://i.imgur.com/MF6Cutf.png =300x400)

---

## Other effects

----

### Edge darkening & Pigment density variation

- Let $\mathbf{N}$ be the sum of multiple perlin noise texture with geometrically increasing frequency and decreasing amplitude, which is used for pigment density variation effect.

----

- For every pixel $p$:
    - Let $\Delta d = a(\lvert \mathbf{g}_x \rvert + \lvert \mathbf{g}_y \rvert) + b \mathbf{N}(p)$, where $\mathbf{g}$ is the intensity gradient at $p$, and $a, b$ are parameters for controlling the intensity of the effects.
    - Adjust the pixel's "pigment density":
    - $f(p) \leftarrow f(p) - (f(p) - f(p)^2) \Delta d$
        - Taken from [7]

----

![](https://i.imgur.com/S4ppU3w.png =300x400) ![](https://i.imgur.com/KhBSiwr.png =300x400)

----

![](https://i.imgur.com/HToMvnU.png =300x400)

----

### Granulation

- Let $\mathbf{N}$ be a perlin noise texture with high frequency to simulate paper texture.
- For every pixel $p$:
    - Let $\textbf{g}$ be the gradient of $\mathbf{N}$ at $p$.
    - Displace the pixel:
    - $f(p) \leftarrow f(p + c \textbf{g})$, where $c$ is a parameter for controlling the intensity of the displacement.

----

![](https://i.imgur.com/L9AQoXN.png =300x400)

---

## Results

----

![](https://i.imgur.com/QWQ1h1T.jpg =420x280) ![](https://i.imgur.com/AakRXOc.png =420x280)

- Notice the hand tremor effect results in the horizon being no longer straight.

----

![](https://i.imgur.com/WhI3fzw.jpg =300x400) ![](https://i.imgur.com/L9AQoXN.png =300x400)

----

![](https://i.imgur.com/Mv1KMZ3.png)

- Notice the wet-in-wet effect on the edge of the platform.

----

![](https://i.imgur.com/vJrSJRL.jpg =400x300) ![](https://i.imgur.com/rnerSXu.jpg =400x300)

----

![](https://i.imgur.com/TIeEHpE.png =400x284)

- Notice the edge darkening effect on the shadow.

----

![](https://i.imgur.com/5bzSDyE.jpg =400x300) ![](https://i.imgur.com/agzGRqP.jpg =400x300)

----

![](https://i.imgur.com/zPwnvNL.jpg =400x300) ![](https://i.imgur.com/YzpnRMO.jpg =400x300)

----

![](https://i.imgur.com/dUPa4lT.jpg =400x300) ![](https://i.imgur.com/G16OFKk.png =400x300)

----

![](https://i.imgur.com/I3rOPQ5.jpg =400x300) ![](https://i.imgur.com/DtflC28.jpg =400x300)

---

## Conclusion and Future Work

----

- Our system is able to transform a static image into the style of watercolor painting with decent quality.
- Effects in the original paper such as hand tremor, wet-in-wet, edge darkening, etc., are (to some extent) successfully recreated.

----

- Our result is not quite as good as the original paper's:
    - "Color Adjustment" step in the original paper is for transforming the image into some color pallette commonly used by watercolor artists. We instead used a hardcoded linear transform. This makes the color pallette of our image rather uninteresting.

----

- Our result is not quite as good as the original paper's:
    - Our quality of hand tremor and wet-in-wet effect is not as good as the original paper's.
    - Difference between salient and unimportant region is not as clear as the original paper's.

----

- Our result is not quite as good as the original paper's:
    - One of the selling point of the original paper is that their method is efficient and highly parallelizable, which this implementation mostly neglected. As a result, our system run at a much slower speed.
    - For a 800x600 image, our system takes about 10 minutes to process, where the original paper's takes about 1.5 seconds.

---

## References

----

<!-- .slide: style="font-size: 20px; text-align: left;" -->

[1] M. Wang et al., "Towards Photo Watercolorization with Artistic Verisimilitude," in IEEE Transactions on Visualization and Computer Graphics, vol. 20, no. 10, pp. 1451-1460, Oct. 2014, doi: 10.1109/TVCG.2014.2303984.

[2] M. Cheng, N. J. Mitra, X. Huang, P. H. S. Torr and S. Hu, "Global Contrast Based Salient Region Detection," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 37, no. 3, pp. 569-582, 1 March 2015, doi: 10.1109/TPAMI.2014.2345401.

[3] https://docs.opencv.org/4.5.2/d7/d1b/group__imgproc__misc.html#ggaa9e58d2860d4afa658ef70a9b1115576a95251923e8e22f368ffa86ba8bce87ff

[4] Felzenszwalb, P.F., Huttenlocher, D.P. Efficient Graph-Based Image Segmentation. International Journal of Computer Vision 59, 167–181 (2004). https://doi.org/10.1023/B:VISI.0000022288.19776.77

[5] https://docs.opencv.org/4.5.2/dd/d19/classcv_1_1ximgproc_1_1segmentation_1_1GraphSegmentation.html

[6] https://github.com/devin6011/ICGproject-watercolorization

[7] Adrien Bousseau, Matt Kaplan, Joëlle Thollot, and François X. Sillion. 2006. Interactive watercolor rendering with temporal coherence and abstraction. In Proceedings of the 4th international symposium on Non-photorealistic animation and rendering (NPAR '06). Association for Computing Machinery, New York, NY, USA, 141–149. DOI:https://doi.org/10.1145/1124728.1124751

[8] Curtis, Cassidy & Anderson, Sean & Seims, Joshua & Fleischer, Kurt & Salesin, David. (1997). Computer-Generated Watercolor. Proc. SIGGRAPH1997. 97. 10.1145/258734.258896. 

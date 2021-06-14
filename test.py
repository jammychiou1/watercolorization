import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter

# load the input image
#image = cv2.imread('small_2.jpg')
image = cv2.imread('apple.jpeg')
print(image.shape)
print(image[0][0])
rectangle = (image.shape[0], image.shape[1])
N = image.shape[0] * image.shape[1]

def val_to_lab(i, j, k, steps):
    #c = sRGBColor(i / (steps - 1), j / (steps - 1), k / (steps - 1), is_upscaled=False)
    c = np.array([[[i * 255 // (steps - 1), j * 255 // (steps - 1), k * 255 // (steps - 1)]]], dtype=np.uint8)
    #print(c)
    c = cv2.cvtColor(c, cv2.COLOR_BGR2LAB).astype(np.float32).flatten()
    #print(c.dtype)
    c[0] *= 100 / 255
    return c

def reduce_color(image, steps):
    image = image.copy()
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    image_lab[:, :, 0] *= 100 / 255
    image_lab = image_lab.reshape(-1, 3)

    mns = np.full(image.shape[0] * image.shape[1], float('inf'), dtype=np.float32)
    bst = np.zeros(image.shape[0] * image.shape[1], dtype=np.int32)
    hist = []
    for i in range(steps):
        print(i)
        for j in range(steps):
            for k in range(steps):
                tmp = np.sum((image_lab - val_to_lab(i, j, k, steps)) ** 2, axis=1)
                #print(tmp.shape)
                swap = tmp < mns
                np.copyto(mns, tmp, where=swap)
                bst[swap] = np.ravel_multi_index((i, j, k), (steps, steps, steps))

    uniques, counts = np.unique(bst, return_counts=True)
    hist = list(zip(uniques, counts))
    ncolors = len(hist)

    hist.sort(reverse=True, key=lambda ele: ele[1])

    thr = int(image.shape[0] * image.shape[1] * 0.95)
    sm = 0
    values = np.zeros((image.shape[0], image.shape[1]), dtype=np.int32)
    for i, (val, cnt) in enumerate(hist):
        sm += cnt
        whr = np.unravel_index(np.nonzero(bst == val), (image.shape[0], image.shape[1]))
        clr = np.array(np.unravel_index(val, (steps, steps, steps))) * 255 // (steps - 1)
        image[whr] = clr
        values[whr] = val
        if sm >= thr:
            ncolors = i + 1
            break
    print(ncolors)
    for val, _ in hist[ncolors : ]:
        whr = np.unravel_index(np.nonzero(bst == val), (image.shape[0], image.shape[1]))
        i, j, k = np.unravel_index(val, (steps, steps, steps))
        mn = float('inf')
        for val2, _ in hist[ : ncolors]:
            i2, j2, k2 = np.unravel_index(val2, (steps, steps, steps))
            tmp = np.sum((val_to_lab(i2, j2, k2, steps) - val_to_lab(i, j, k, steps)) ** 2)
            if tmp < mn:
                bst_clr = np.array([i2, j2, k2]) * 255 // (steps - 1)
                bst_val = val2
                mn = tmp
        image[whr] = bst_clr
        values[whr] = bst_val
    return image, values, [ele[0] for ele in hist[ : ncolors]]

def get_salient(image):
    rectangle = (image.shape[0], image.shape[1])
    N = image.shape[0] * image.shape[1]

    steps = 12
    image_reduce, values, colors = reduce_color(image, steps)

    dc = {}
    for val1 in colors:
        for val2 in colors:
            i1, j1, k1 = np.unravel_index(val1, (steps, steps, steps))
            i2, j2, k2 = np.unravel_index(val2, (steps, steps, steps))
            dc[(val1, val2)] = np.sum((val_to_lab(i1, j1, k1, steps) - val_to_lab(i2, j2, k2, steps)) ** 2) ** 0.5

    plt.imshow(cv2.cvtColor(image_reduce, cv2.COLOR_BGR2RGB))
    plt.show()

    # Graph-Based Image Segmentation 分割器
    segmentator = cv2.ximgproc.segmentation.createGraphSegmentation(sigma=0.5, k=500, min_size=1000)

    # 分割圖形
    segment_map = segmentator.processImage(image)
    print(np.max(segment_map), np.min(segment_map))
    print(np.max(values))
    nseg = np.max(segment_map) + 1

    segments = [{'cnt': 0, 'x': 0, 'y': 0, 'clrs': {}, 'sal': 0} for i in range(nseg)]

    for idx, (val, seg_idx) in enumerate(zip(values.flat, segment_map.flat)):
        i, j = np.unravel_index(idx, (image.shape[0], image.shape[1]))
        #if j == 0:
            #print(i)
        segment = segments[seg_idx]
        segment['cnt'] += 1
        segment['x'] += i / (image.shape[0] - 1)
        segment['y'] += j / (image.shape[1] - 1)
        if val not in segment['clrs']:
            segment['clrs'][val] = 1
        else:
            segment['clrs'][val] += 1

    for segment in segments:
        segment['x'] /= segment['cnt']
        segment['y'] /= segment['cnt']

    def dis_seg(seg1, seg2):
        sm = 0
        for val1, cnt1 in seg1['clrs'].items():
            tmp = 0
            f1 = cnt1 / seg1['cnt']
            for val2, cnt2 in seg2['clrs'].items():
                f2 = cnt2 / seg2['cnt']
                d = dc[(val1, val2)]
                tmp += f2 * d
            sm += f1 * tmp
        return sm

    sigma_sq = 0.4
    for i in range(nseg):
        #print(i)
        segi = segments[i]
        for j in range(i + 1, nseg):
            segj = segments[j]
            ds = ((segi['x'] - segj['x']) ** 2 + (segi['y'] - segj['y']) ** 2) ** 0.5
            dr = dis_seg(segi, segj)
            spatial_w = np.exp(-ds / sigma_sq)
            segi['sal'] += spatial_w * segj['cnt'] * dr
            segj['sal'] += spatial_w * segi['cnt'] * dr

    saliency = np.zeros(rectangle, dtype=np.float32)
    for idx, seg_idx in enumerate(segment_map.flat):
        i, j = np.unravel_index(idx, rectangle)
        saliency[i, j] = segments[seg_idx]['sal']
    
    return saliency / np.max(saliency)

def get_salient_region(image):
    rectangle = (image.shape[0], image.shape[1])
    N = image.shape[0] * image.shape[1]
    if 1 == 1:
        saliency = get_salient(image)
        np.save('sal.npy', saliency)
    else:
        saliency = np.load('sal.npy')

    print(np.max(saliency), np.min(saliency))
    mask = (saliency > 0.6).astype(np.uint8)

    xs = np.arange(image.shape[0])
    ys = np.arange(image.shape[1])
    xs, ys = np.meshgrid(xs, ys, indexing='ij')
    #print(xs)
    edge = (xs > 10) & (xs < image.shape[0] - 10) & (ys > 10) & (ys < image.shape[1] - 10)
    #print(edge)
    plt.imshow(edge)
    plt.show()
    mask = (mask & edge).astype(np.uint8)

    kernel = np.ones((5, 5), dtype=np.uint8)
    for i in range(5):
        plt.imshow(mask)
        plt.show()
        trimap = np.full(rectangle, cv2.GC_PR_BGD, dtype=np.uint8)
        trimap[cv2.erode(mask, kernel, iterations=10) > 0.5] = cv2.GC_FGD
        trimap[cv2.dilate(mask, kernel, iterations=5) < 0.5] = cv2.GC_BGD
        plt.imshow(trimap)
        plt.show()

        bgdModel = np.zeros((1, 65), dtype=np.float64)
        fgdModel = np.zeros((1, 65), dtype=np.float64)
        cv2.grabCut(image, trimap, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
        plt.imshow(trimap)
        plt.show()
        mask = (((trimap == 1) | (trimap == 3)) & edge).astype(np.uint8)
    return mask

if 1 == 1:
    mask = get_salient_region(image)
    np.save('mask.npy', mask)
else:
    mask = np.load('mask.npy')

xs = np.arange(image.shape[0])
ys = np.arange(image.shape[1])
xs, ys = np.meshgrid(xs, ys, indexing='ij')

if 1 == 1:
    frmx = xs.copy()
    frmy = ys.copy()

    vsted = mask.copy()

    def dist(x1, y1, x2, y2):
        return (x1 - x2) ** 2 + (y1 - y2) ** 2

    mn_d = np.full(rectangle, float('inf'), dtype=np.float32)
    mn_d[mask == 1] = 0
    plt.imshow(mn_d)
    plt.show()
    queue = list(zip(*mask.nonzero()))
    now = 0
    dxs = [0, 1, 1, 1, 0, -1, -1, -1]
    dys = [1, 1, 0, -1, -1, -1, 0, 1]
    while len(queue) > now:
        x, y = queue[now]
        print(now)
        vsted[x, y] = 2
        now += 1
        for dx, dy in zip(dxs, dys):
            xx = x + dx
            yy = y + dy
            if not (0 <= xx and xx < image.shape[0] and 0 <= yy and yy < image.shape[1]):
                continue
            if vsted[xx, yy] != 2:
                if vsted[xx, yy] == 0:
                    vsted[xx, yy] = 1
                    queue.append((xx, yy))
                tmp = dist(xx, yy, frmx[x, y], frmy[x, y])
                if tmp < mn_d[xx, yy]:
                    mn_d[xx, yy] = tmp
                    frmx[xx, yy] = frmx[x, y]
                    frmy[xx, yy] = frmy[x, y]
    plt.imshow(mn_d)
    plt.show()
    np.save('frmx.npy', frmx)
    np.save('frmy.npy', frmy)
else:
    frmx = np.load('frmx.npy')
    frmy = np.load('frmy.npy')

box = np.array([[1, 0, 0], [0, 1, 0], [-1, 0, image.shape[0]], [0, -1, image.shape[1]]])
frm = np.stack((frmx, frmy, np.ones_like(mask)), axis=-1)
here = np.stack((xs, ys, np.ones_like(mask)), axis=-1)

plt.imshow(mask)
a = np.einsum('xyc, rc -> xyr', frm, box)
b = np.einsum('xyc, rc -> xyr', here, box)
dist = (1 - np.max((a - b) / a, axis=2))
dist = gaussian_filter(dist, sigma=3)
dist = gaussian_filter(dist, sigma=3)
plt.imshow(dist)
plt.show()

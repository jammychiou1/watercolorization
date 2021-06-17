import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from perlin_noise import PerlinNoise
import random

# load the input image
image = cv2.imread('small_2.jpg')
#image = cv2.imread('landscape29.png')
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
    print('reduced color count', ncolors)
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
    segmentator = cv2.ximgproc.segmentation.createGraphSegmentation(sigma=0.5, k=300, min_size=200)

    # 分割圖形
    segment_map = segmentator.processImage(image)
    #print(np.max(segment_map), np.min(segment_map))
    #print(np.max(values))
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
    #return saliency / N

def get_salient_region(image):
    rectangle = (image.shape[0], image.shape[1])
    N = image.shape[0] * image.shape[1]
    if 1 == 1:
        saliency = get_salient(image)
        np.save('sal.npy', saliency)
    else:
        saliency = np.load('sal.npy')

    plt.imshow(saliency)
    plt.show()
    print(np.max(saliency), np.min(saliency))
    saliency = (saliency * 255).astype(np.uint8)
    #t = (np.max(saliency) + np.min(saliency)) / 2
    #mask = (saliency > t).astype(np.uint8)
    _, mask = cv2.threshold(saliency, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    plt.imshow(mask)
    plt.show()

    xs = np.arange(image.shape[0])
    ys = np.arange(image.shape[1])
    xs, ys = np.meshgrid(xs, ys, indexing='ij')
    #print(xs)
    edge = (xs > 10) & (xs < image.shape[0] - 10) & (ys > 10) & (ys < image.shape[1] - 10)
    #print(edge)
    #plt.imshow(edge)
    #plt.show()
    '''
    mask = (mask & edge).astype(np.uint8)

    kernel = np.ones((5, 5), dtype=np.uint8)
    for i in range(4):
        plt.imshow(mask)
        plt.show()
        trimap = np.full(rectangle, cv2.GC_PR_BGD, dtype=np.uint8)
        trimap[cv2.erode(mask, kernel, iterations=5) > 0.5] = cv2.GC_FGD
        trimap[cv2.dilate(mask, kernel, iterations=5) < 0.5] = cv2.GC_BGD
        plt.imshow(trimap)
        plt.show()

        bgdModel = np.zeros((1, 65), dtype=np.float64)
        fgdModel = np.zeros((1, 65), dtype=np.float64)
        cv2.grabCut(image, trimap, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
        plt.imshow(trimap)
        plt.show()
        mask = (((trimap == 1) | (trimap == 3)) & edge).astype(np.uint8)
    '''

    '''
    trimap = np.full(rectangle, cv2.GC_PR_BGD, dtype=np.uint8)
    trimap[mask != 0] = cv2.GC_PR_FGD
    bgdModel = np.zeros((1, 65), dtype=np.float64)
    fgdModel = np.zeros((1, 65), dtype=np.float64)
    plt.imshow(trimap)
    plt.show()
    cv2.grabCut(image, trimap, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
    plt.imshow(trimap)
    plt.show()
    mask = ((trimap == 1) | (trimap == 3)) & edge
    '''
    return (mask != 0) & edge

def get_dist_field(image, mask):
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
        #plt.imshow(mn_d)
        #plt.show()
        queue = list(zip(*mask.nonzero()))
        now = 0
        dxs = [0, 1, 1, 1, 0, -1, -1, -1]
        dys = [1, 1, 0, -1, -1, -1, 0, 1]
        while len(queue) > now:
            x, y = queue[now]
            #print(now)
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
    dist = np.max((a - b) / a, axis=2)
    dist = ndimage.gaussian_filter(dist, sigma=10)
    #for i in range(20):
    #    dist = gaussian_filter(dist, sigma=5)
    plt.imshow(dist)
    plt.show()

    return dist

def abstraction(image):
    if 1 == 1:
        mask = get_salient_region(image)
        np.save('mask.npy', mask)
    else:
        mask = np.load('mask.npy')

    if 1 == 1:
        dist_field = get_dist_field(image, mask)
        print(type(dist_field))
        np.save('dist.npy', dist_field)
    else:
        dist_field = np.load('dist.npy')

    image_float = image / 255.0

    # Graph-Based Image Segmentation 分割器
    segmentator = cv2.ximgproc.segmentation.createGraphSegmentation(sigma=0.5, k=300, min_size=200)

    # 分割圖形
    segment_map = segmentator.processImage(image)
    #print(np.max(segment_map), np.min(segment_map))
    nseg = np.max(segment_map) + 1
    pix_in_sal = np.bincount(segment_map.flat, weights=mask.flat)
    pix = np.bincount(segment_map.flat)
    #print(pix)
    #print(pix_in_sal)
    #print(pix.shape)
    inside = pix_in_sal * 2 > pix
    inside_image = inside[segment_map].astype(np.int32)
    avg_color = np.zeros((1, nseg, 3))
    for i in range(3):
        avg_color[0, :, i] = np.bincount(segment_map.flat, weights=image[:, :, i].flat)
    avg_color /= pix[:, np.newaxis]
    print(avg_color.dtype)
    avg_color = avg_color.astype(np.uint8)
    #plt.imshow(avg_color[0][segment_map])
    #plt.show()
    avg_color = cv2.cvtColor(avg_color, cv2.COLOR_BGR2HSV)
    avg_hue = avg_color[0, :, 0].astype(np.float32) * 2
    avg_brightness = avg_color[0, :, 2].astype(np.float32)

    colors = np.random.rand(nseg, 3)
    plt.imshow(colors[segment_map])
    plt.show()
    plt.imshow(dist_field)
    plt.show()
    plt.imshow(inside_image)
    plt.show()

    abstract = np.zeros_like(image_float)
    cases = [0, 0, 0]
    for i in range(image.shape[0]):
        print(i)
        for j in range(image.shape[1]):
            cnt = 0
            mean = np.zeros(3)
            if inside_image[i, j]:
                for ii in range(i - 3, i + 4):
                    for jj in range(j - 3, j + 4):
                        if 0 <= ii and ii < image.shape[0] and 0 <= jj and jj < image.shape[1]:
                            if segment_map[i, j] == segment_map[ii, jj]:
                                cnt += 1
                                mean += image_float[ii, jj]
                                cases[0] += 1
            else:
                sz = round(np.clip(2 * 7 * (dist_field[i, j] + 0.3), 6, 13))
                for ii in range(i - sz // 2, i - sz // 2 + sz):
                    for jj in range(j - sz // 2, j - sz // 2 + sz):
                        if 0 <= ii and ii < image.shape[0] and 0 <= jj and jj < image.shape[1]:
                            if segment_map[i, j] == segment_map[ii, jj]:
                                cnt += 1
                                mean += image_float[ii, jj]
                                cases[1] += 1
                            else:
                                color_diff = np.sum((image_float[i, j] - image_float[ii, jj] ) ** 2) ** 0.5
                                if color_diff < 0.3 * dist_field[i, j]:
                                    cnt += 1
                                    mean += image_float[ii, jj]
                                    cases[2] += 1
            abstract[i, j] = mean / (cnt + 1e-7)
    print(cases)
    return abstract, inside_image, segment_map, avg_hue, avg_brightness

if 1 == 1:
    abstract, inside_image, segment_map, avg_hue, avg_brightness = abstraction(image)
    np.save('abstract.npy', abstract) 
    np.save('inside_image.npy', inside_image) 
    np.save('segment_map.npy', segment_map) 
    np.save('avg_hue.npy', avg_hue) 
    np.save('avg_brightness.npy', avg_brightness) 
else:
    abstract = np.load('abstract.npy')
    inside_image = np.load('inside_image.npy')
    segment_map = np.load('segment_map.npy')
    avg_hue = np.load('avg_hue.npy')
    avg_brightness = np.load('avg_brightness.npy')

print(avg_hue.shape)
plt.imshow(abstract[:, :, ::-1])
plt.show()

gradx = np.mean(cv2.Sobel(image, cv2.CV_32F, 1, 0) / 8.0 / 255, axis=2)
grady = np.mean(cv2.Sobel(image, cv2.CV_32F, 0, 1) / 8.0 / 255, axis=2)
print(gradx.shape)
print(np.max(gradx))

def classify_edge(abstract, inside_image, segment_map, avg_hue, avg_brightness, gradx, grady):
    def deg_diff(h1, h2):
        #print(h1, h2, min(abs(h1 - h2 - 360), abs(h1 - h2), abs(h1 - h2 + 360)))
        return min(abs(h1 - h2 - 360), abs(h1 - h2), abs(h1 - h2 + 360))

    def set_type1(segment_map, avg_brightness, edge_type, feather_source, x1, y1, x2, y2):
        edge_type[x1, y1] = 3
        edge_type[x2, y2] = 3
        if avg_brightness[segment_map[x1, y1]] < avg_brightness[segment_map[x2, y2]]:
            feather_source[x1, y1] = 1
        else:
            feather_source[x2, y2] = 1

    def process_edge(segment_map, inside_image, avg_hue, avg_brightness, gradx, grady, edge_type, feather_source, x1, y1, x2, y2):
        if segment_map[x1, y1] == segment_map[x2, y2]:
            return
        if (gradx[x1, y1] > 0.1) and (grady[x1, y1] > 0.1):
            if inside_image[x1, y1] or inside_image[x2, y2]:
                if deg_diff(avg_hue[segment_map[x1, y1]], avg_hue[segment_map[x2, y2]]) < 20:
                    set_type1(segment_map, avg_brightness, edge_type, feather_source, x1, y1, x2, y2)
                    return
            else:
                if deg_diff(avg_hue[segment_map[x1, y1]], avg_hue[segment_map[x2, y2]]) < 90:
                    set_type1(segment_map, avg_brightness, edge_type, feather_source, x1, y1, x2, y2)
                    return
        if deg_diff(avg_hue[segment_map[x1, y1]], avg_hue[segment_map[x2, y2]]) < 40:
            edge_type[x1, y1] = max(edge_type[x1, y1], 2)
            edge_type[x2, y2] = max(edge_type[x2, y2], 2)
            return
        edge_type[x1, y1] = max(edge_type[x1, y1], 1)
        edge_type[x2, y2] = max(edge_type[x2, y2], 1)

    edge_type = np.zeros(image.shape[:2], dtype=np.uint8)
    feather_source = np.zeros(image.shape[:2], dtype=np.uint8)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if i < image.shape[0] - 1:
                process_edge(segment_map, inside_image, avg_hue, avg_brightness, gradx, grady, edge_type, feather_source, i, j, i + 1, j)
            if j < image.shape[1] - 1:
                process_edge(segment_map, inside_image, avg_hue, avg_brightness, gradx, grady, edge_type, feather_source, i, j, i, j + 1)
    edge_color = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    plt.imshow(edge_color[edge_type])
    plt.show()

    kernel = np.ones((8, 8), dtype=np.uint8)
    type3 = edge_type == 3
    type3_gap = cv2.dilate(type3.astype(np.uint8), kernel, iterations=1)
    type2 = (edge_type == 2) & (~type3_gap)
    kernel = np.ones((16, 16), dtype=np.uint8)
    type2_gap = cv2.dilate(type2.astype(np.uint8), kernel, iterations=1)
    type1 = (edge_type == 1) & (~type3_gap) & (~type2_gap)
    edge_color_gapped = np.zeros_like(image, dtype=np.float32)
    type1 = ndimage.gaussian_filter(type1.astype(np.float32), sigma=1)
    type2 = ndimage.gaussian_filter(type2.astype(np.float32), sigma=1)
    type3 = type3.astype(np.float32)
    edge_color_gapped = type3[:, :, np.newaxis] * np.array([0.0, 0.0, 1.0]) + type2[:, :, np.newaxis] * np.array([0.0, 1.0, 0.0]) + type1[:, :, np.newaxis] * np.array([1.0, 0.0, 0.0])
    print(np.min(edge_color_gapped), np.max(edge_color_gapped))
    edge_color_gapped *= 255
    plt.imshow(edge_color_gapped)
    plt.show()
    return type1, type2, type3, feather_source

def hand_tremor(abstract, segment_map, type1, type2):
    nseg = np.max(segment_map) + 1
    segment_pixs = [[] for i in range(nseg)]
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            segment_pixs[segment_map[i, j]].append((i, j))

    def distort(abstract, segment_map, seg_id, type1, type2, type1_noisex, type1_noisey, type2_noisex, type2_noisey, modified_pix, x, y, displace=None):
        if type2[x, y] < 1 / 30 and type1[x, y] < 1 / 30:
            xx = x
            yy = y
        elif type2[x, y] > type1[x, y]:
            dx = round(15 * type2[x, y] * type2_noisex([x, y]))
            dy = round(15 * type2[x, y] * type2_noisey([x, y]))
            xx = np.clip(x + dx, 0, image.shape[0] - 1)
            yy = np.clip(y + dy, 0, image.shape[1] - 1)
            #displace[x, y] = 30 * type2[x, y] * type2_noisex([x, y])
        else:
            dx = round(15 * type1[x, y] * type1_noisex([x, y]))
            dy = round(15 * type1[x, y] * type1_noisey([x, y]))
            xx = np.clip(x + dx, 0, image.shape[0] - 1)
            yy = np.clip(y + dy, 0, image.shape[1] - 1)
            #displace[x, y] = 30 * type1[x, y] * type1_noisex([x, y])
        '''
        if type(displace) != type(None):
            displace[x, y] = dx
        '''
        w = int(segment_map[xx, yy] == seg_id)
        if w == 0:
            return
        if (x, y) not in modified_pix:
            modified_pix[(x, y)] = {'w_sum': 0, 'color': np.zeros(3)}
        modified_pix[(x, y)]['w_sum'] += w
        modified_pix[(x, y)]['color'] += w * abstract[xx, yy]
        tmp = np.zeros_like(image, dtype=np.float32)
        
    def edge_darken(x, y, modified_pix):
        for dx in range(-2, 3):
            xx = x + dx
            if xx < 0 or xx >= image.shape[0]:
                continue
            for dy in range(-2, 3):
                yy = y + dy
                if yy < 0 or yy >= image.shape[1]:
                    continue
                if (xx, yy) not in modified_pix:
                    c = modified_pix[x, y]['color']
                    c -= 0.5 * (c - c ** 2)
                    return

    type2_noisex = PerlinNoise(octaves=0.015, seed=1)
    type2_noisey = PerlinNoise(octaves=0.015, seed=2)
    canvas = np.ones_like(image, dtype=np.float32) * 0.001
    canvas_w = np.ones(image.shape[:2], dtype=np.float32) * 0.001
    #type2_noisex = np.array([[type2_noisex([i, j]) for j in range(image.shape[1])] for i in range(image.shape[0])])
    #type2_noisey = np.array([[type2_noisey([i, j]) for j in range(image.shape[1])] for i in range(image.shape[0])])
    kernel = np.ones((2, 2), dtype=np.uint8)

    abstract = abstract * 2 / 3 + 0.25
    #feather_layer = np.ones_like(image, dtype=np.float32) * 0.001
    #feather_w = np.ones_like(image[:2], dtype=np.float32) * 0.001

    for i in range(nseg):
        print('{} / {}'.format(i+1, nseg))
        modified_pix = {}
        visited_pix = {}
        #displace = np.zeros(image.shape[:2])
        #region = segment_map == i
        #region_distorted = np.zeros(image.shape[:2])
        type1_noisex = PerlinNoise(octaves=0.015, seed=2 * i + 1)
        type1_noisey = PerlinNoise(octaves=0.015, seed=2 * i + 2)
        #processing = cv2.dilate(region.astype(np.uint8), kernel, iterations=1)
        for x, y in segment_pixs[i]:
            for dx in range(-4, 5):
                xx = x + dx
                if xx < 0 or xx >= image.shape[0]:
                    continue
                for dy in range(-4, 5):
                    yy = y + dy
                    if yy < 0 or yy >= image.shape[1]:
                        continue
                    if (xx, yy) not in visited_pix:
                        visited_pix[(xx, yy)] = True
                        #distort(abstract, segment_map, i, type1, type2, type1_noisex, type1_noisey, type2_noisex, type2_noisey, modified_pix, xx, yy, displace)
                        distort(abstract, segment_map, i, type1, type2, type1_noisex, type1_noisey, type2_noisex, type2_noisey, modified_pix, xx, yy)
        '''
        region_color = np.zeros_like(image, dtype=np.float32)
        for x, y in modified_pix:
            #print(modified_pix[(x, y)])
            #region_color[x, y] = modified_pix[(x, y)]['color']
            region_color[x, y, 0] = 1
        region_color += (segment_map == i)[:, :, np.newaxis] * np.array([0.0, 1.0, 0.0])
        plt.imshow(region_color)
        plt.show()
        plt.imshow(displace)
        plt.show()
        '''
        for x, y in modified_pix:
            w = modified_pix[(x, y)]['w_sum']
            wc = np.clip(w, 0, 1)
            canvas[x, y] += modified_pix[(x, y)]['color'] / w * wc
            canvas_w[x, y] += wc

    plt.imshow(canvas_w)
    plt.show()
    canvas /= canvas_w[:, :, np.newaxis]
    return canvas

if 1 == 1:
    type1, type2, type3, feather_source = classify_edge(abstract, inside_image, segment_map, avg_hue, avg_brightness, gradx, grady)
    canvas = hand_tremor(abstract, segment_map, type1, type2)
    np.save('feather_source.npy', feather_source) 
    np.save('canvas.npy', canvas)
else:
    feather_source = np.load('feather_source.npy')
    canvas = np.load('canvas.npy')
    
plt.imshow(canvas[:, :, ::-1])
plt.show()

canvas_w = np.ones(image.shape[:2], dtype=np.float32)

def get_kernel(r, a, b):
    xs = np.arange(r).astype(np.float32)
    ys = np.arange(r).astype(np.float32)
    xs, ys = np.meshgrid(xs, ys, indexing='ij')
    tmp = ((xs - (r - 1) / 2) / a) ** 2 + ((ys - (r - 1) / 2) / b) ** 2
    return np.clip(1 - tmp, 0, 1)

feather = get_kernel(15, 7, 3.5)
feather_sum = np.sum(feather)
plt.imshow(feather)
plt.show()
feather_image = np.zeros_like(image, dtype=np.float32)
feather_w = np.zeros(image.shape[:2], dtype=np.float32)

for x in range(image.shape[0]):
    for y in range(image.shape[1]):
        if feather_source[x, y]:
            gx = gradx[x, y]
            gy = grady[x, y]
            angle = np.arctan2(gy, gx)
            r = np.random.uniform(3, 5)
            dx = np.cos(angle)
            dy = np.sin(angle)
            basex = round(x + dx * r)
            basey = round(y + dy * r)
            if 0 <= basex < image.shape[0] and 0 <= basey < image.shape[1]:
                canvas[basex, basey] = canvas[x, y]
            
for x in range(image.shape[0]):
    for y in range(image.shape[1]):
        if feather_source[x, y]:
            gx = gradx[x, y]
            gy = grady[x, y]
            angle = np.arctan2(gy, gx)
            feather_rot = ndimage.rotate(feather, angle * 180 / np.pi, reshape=False)
            tmp = np.zeros((29, 29, 3))
            for dx in range(-14, 15):
                for dy in range(-14, 15):
                    tmp[dx + 14, dy + 14] = canvas[np.clip(x + dx, 0, image.shape[0] - 1), np.clip(y + dy, 0, image.shape[1] - 1)]
            for c in range(3):
                #print(feather_rot.shape)
                #print(tmp[:, :, c].shape)
                filtered = ndimage.convolve(tmp[:, :, c], feather_rot)[7 : 22, 7 : 22] / feather_sum
                prod = filtered * feather_rot
                for dx in range(-7, 8):
                    for dy in range(-7, 8):
                        if 0 <= x + dx < image.shape[0] and 0 <= y + dy < image.shape[1]:
                            feather_image[x + dx, y + dy, c] += prod[dx + 7, dy + 7]
            for dx in range(-7, 8):
                for dy in range(-7, 8):
                    if 0 <= x + dx < image.shape[0] and 0 <= y + dy < image.shape[1]:
                        feather_w[x + dx, y + dy] += feather_rot[dx + 7, dy + 7]

feather_image /= feather_w[:, :, np.newaxis] + 0.0000001
feather_w = np.clip(feather_w, 0, 1)
#plt.imshow((1 - feather_w[:, :, np.newaxis]) + feather_image * feather_w[:, :, np.newaxis])
#plt.show()
canvas = canvas * (1 - feather_w[:, :, np.newaxis]) + feather_image * feather_w[:, :, np.newaxis]
print(np.min(canvas), np.max(canvas))
plt.imshow(canvas[:, :, ::-1])
plt.show()

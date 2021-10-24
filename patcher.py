import openslide
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import cv2
import copy

class Patcher:
    def __init__(self, svs_path, xml_path):
        self.slide = openslide.OpenSlide(svs_path)
        cancer, c_idx = svs_path.split('_PNI2021chall_train_')
        self.cancer = cancer.split('/')[-1]
        self.cancer_idx = int(c_idx.split('.')[0])

        self.svs_path = svs_path
        self.xml_path = xml_path

        self.w_ratio = []
        self.h_ratio = []
        src_w, src_h = self.slide.level_dimensions[0]
        for i in [0,1,2]:
            dest_w, dest_h = self.slide.level_dimensions[i]
            self.w_ratio.append(src_w/dest_w)
            self.h_ratio.append(src_h/dest_h)

    def thumnail(self, region_idx, level, thickness=16, margin_x=100, margin_y=100, show=False):
        pts_raw, pts = self.pts_dict[region_idx]
        # print(pts_raw, pts)
        min_raw_x, min_raw_y = np.min(pts_raw, axis=0)
        max_raw_x, max_raw_y = np.max(pts_raw, axis=0)
        min_x, min_y = np.min(pts, axis=0)
        max_x, max_y = np.max(pts, axis=0)
        # print(min_x, min_y)
        raw_w = max_raw_x - min_raw_x
        raw_h = max_raw_y - min_raw_y
        w = max_x - min_x
        h = max_y - min_y

        if level == 0: # If level 2 is needed, Change the code. not work.
            w = raw_w
            h = raw_h

        pts = self.pts_dict[region_idx][level]
        pts = pts.reshape((-1,1,2))
        if level == 0:
            pts[:,0,0] -= min_raw_x - margin_x
            pts[:,0,1] -= min_raw_y - margin_y
        elif level == 1:
            pts[:,0,0] -= min_x - margin_x
            pts[:,0,1] -= min_y - margin_y

        if min_y - round(margin_y*self.h_ratio[level]) < 0:
            pts[:,0,1] -= margin_y
            img = self.slide.read_region((min_raw_x-round(margin_x*self.w_ratio[level]), min_raw_y),
                                         level, (w+margin_x*2, h+margin_y*2))
        else:
            img = self.slide.read_region((min_raw_x-round(margin_x*self.w_ratio[level]), min_raw_y-round(margin_y*self.h_ratio[level])),
                                         level, (w+margin_x*2, h+margin_y*2))
        img = img.convert('RGB')
        img = np.asarray(img, dtype=np.uint8)
        if show:
            img_mark = copy.deepcopy(img)
            img_mark = cv2.polylines(img_mark, [pts], isClosed=False, color=(255,0,0), thickness=thickness)
        else:
            map = np.zeros(img.shape[:-1], dtype=np.int8)
            map = np.expand_dims(map, axis=-1)
            cv2.polylines(map, [pts], isClosed=False, color=1, thickness=thickness)

            # img = np.multiply(img, map)
            # plt.imshow(img)
            # plt.show()

            #print(img.shape, map.shape)

            return img, map

        if show:
            print("show image", self.cancer, self.cancer_idx, region_idx)
            fig = plt.figure()
            rows = 1
            cols = 2

            ax1 = fig.add_subplot(rows, cols, 1)
            ax1.imshow(img)
            ax1.set_title('Raw Image')
            ax1.axis("off")

            ax2 = fig.add_subplot(rows, cols, 2)
            ax2.imshow(img_mark)
            ax2.set_title('Image with Label2')
            ax2.axis("off")

            plt.show()

    def thumnail_weighted(self, region_idx, level, thickness=16, margin_x=100, margin_y=100, show=False):
        pts_raw, pts = self.pts_dict[region_idx]
        # print(pts_raw, pts)
        min_raw_x, min_raw_y = np.min(pts_raw, axis=0)
        max_raw_x, max_raw_y = np.max(pts_raw, axis=0)
        min_x, min_y = np.min(pts, axis=0)
        max_x, max_y = np.max(pts, axis=0)
        # print(min_x, min_y)
        raw_w = max_raw_x - min_raw_x
        raw_h = max_raw_y - min_raw_y
        w = max_x - min_x
        h = max_y - min_y

        if level == 0: # If level 2 is needed, Change the code. not work.
            w = raw_w
            h = raw_h

        pts = self.pts_dict[region_idx][level]
        pts = pts.reshape((-1,1,2))
        if level == 0:
            pts[:,0,0] -= min_raw_x - margin_x
            pts[:,0,1] -= min_raw_y - margin_y
        elif level == 1:
            pts[:,0,0] -= min_x - margin_x
            pts[:,0,1] -= min_y - margin_y

        if min_y - round(margin_y*self.h_ratio[level]) < 0:
            pts[:,0,1] -= margin_y
            img = self.slide.read_region((min_raw_x-round(margin_x*self.w_ratio[level]), min_raw_y),
                                         level, (w+margin_x*2, h+margin_y*2))
        else:
            img = self.slide.read_region((min_raw_x-round(margin_x*self.w_ratio[level]), min_raw_y-round(margin_y*self.h_ratio[level])),
                                         level, (w+margin_x*2, h+margin_y*2))
        img = img.convert('RGB')
        img = np.asarray(img, dtype=np.uint8)
        if show:
            img_mark = copy.deepcopy(img)
            img_mark = cv2.polylines(img_mark, [pts], isClosed=False, color=(255,0,0), thickness=thickness)
        else:
            map = np.zeros(img.shape[:-1], dtype=np.int8)
            map = np.expand_dims(map, axis=-1)
            for w, tk in enumerate(range(thickness, 0, -1)):
                cv2.polylines(map, [pts], isClosed=False, color=w+1, thickness=tk)
            map = map / thickness
            # plt.imshow(map)
            # plt.show()

            # img = np.multiply(img, map)
            # plt.imshow(img)
            # plt.show()

            #print(img.shape, map.shape)

            return img, map

        if show:
            print("show image", self.cancer, self.cancer_idx, region_idx)
            fig = plt.figure()
            rows = 1
            cols = 2

            ax1 = fig.add_subplot(rows, cols, 1)
            ax1.imshow(img)
            ax1.set_title('Raw Image')
            ax1.axis("off")

            ax2 = fig.add_subplot(rows, cols, 2)
            ax2.imshow(img_mark)
            ax2.set_title('Image with Label2')
            ax2.axis("off")

            plt.show()


    def find_region(self, level=1):
        etree = ET.parse(self.xml_path)

        dest_w, dest_h = self.slide.level_dimensions[level]

        annotations = etree.getroot()
        label = int(annotations[1].get("Id"))
        regions = annotations[1].findall("Regions")[0]

        self.pts_dict = {}

        for r_idx, region in enumerate(regions.findall("Region")):
            pts = list()
            raw_pts = list()

            vertices = region.findall("Vertices")[0]
            for vertex in vertices.findall("Vertex"):
                x = round(float(vertex.get("X")))
                y = round(float(vertex.get("Y")))
                raw_pts.append((x,y))

                x = np.clip(round(x/self.w_ratio[level]), 0, dest_w-1)
                y = np.clip(round(y/self.h_ratio[level]), 0, dest_h-1)
                pts.append((x,y))
            raw_pts = np.array(raw_pts, dtype=np.int32)
            pts = np.array(pts, dtype=np.int32)
            self.pts_dict[r_idx] = [raw_pts, pts]
        self.num_regions = len(self.pts_dict)

class SmallPatcher:
    def __init__(self, img, map, weighted_map=None):
        self.img = img
        self.map = map
        if weighted_map is not None:
            self.weighted_map = weighted_map
        else:
            self.weighted_map = None

    def get_candidate(self):
        if self.weighted_map is not None:
            return self.candidate, self.candidate_mask, self.candidate_weighted_mask
        else:
            return self.candidate, self.candidate_mask

    def get_not_candidate(self):
        if self.weighted_map is not None:
            return self.not_candidate, self.not_candidate_mask, self.not_candidate_weighted_mask
        else:
            return self.not_candidate, self.not_candidate_mask

    def show_candidate(self, filter=False):
        print(len(self.candidate))
        fig = plt.figure()
        rows = 1
        cols = len(self.candidate)

        axes = [fig.add_subplot(rows, cols, i+1) for i in range(cols)]
        for idx, ax in enumerate(axes):
            if not filter:
                ax.imshow(self.candidate[idx])
            else:
                ax.imshow(self.candidate[idx]*self.candidate_mask[idx])
            ax.axis("off")
        plt.show()

    def random_cropping(self, crop_size=224, threshold=0.25, window_step=None):
        # print(self.img.shape, self.map.shape)
        W, H = self.img.shape[:-1]
        if window_step == None:
            window_step = int(crop_size//4)
        elif type(window_step) != type(1):
            print("window_step size must be type 'int'")
            return

        total_area = crop_size ** 2
        wh_pts = []
        wh_pts_not = []
        for w in range(0, W-crop_size+1, window_step):
            for h in range(0,H-crop_size+1, window_step):
                cropped_map = self.map[w:w+crop_size,h:h+crop_size]
                indices, counts = np.unique(cropped_map, return_counts=True)
                if 1 not in indices:
                    wh_pts_not.append((w,h))
                elif 0 not in indices:
                    wh_pts.append((w,h))
                else:
                    if counts[-1] / total_area >= threshold:
                        wh_pts.append((w,h))
                    else:
                        wh_pts_not.append((w,h))
                    #     print(f'({w},{h}) is over threshold')
                    # else:
                    #     print(f'({w},{h}) is not over threshold')
        self.candidate = [self.img[w:w+crop_size, h:h+crop_size, :] for w,h in wh_pts]
        self.candidate_mask = [self.map[w:w+crop_size, h:h+crop_size, :] for w,h in wh_pts]
        if self.weighted_map is not None:
            self.candidate_weighted_mask = [self.weighted_map[w:w+crop_size, h:h+crop_size, :] for w,h in wh_pts]
            self.not_candidate_weighted_mask = [self.weighted_map[w:w+crop_size, h:h+crop_size, :] for w,h in wh_pts_not]
        self.not_candidate = [self.img[w:w+crop_size, h:h+crop_size, :] for w,h in wh_pts_not]
        self.not_candidate_mask = [self.map[w:w + crop_size, h:h + crop_size, :] for w, h in wh_pts_not]


if __name__ == '__main__':
    d_path = '/home/jinhee/jinhee/OC/dataset'
    file_name = 'Col_PNI2021chall_train_0001'
    svs_path = d_path+f'/svs_folder/{file_name}.svs'
    xml_path = d_path+f'/xml_folder/{file_name}.xml'

    ptc = Patcher(svs_path, xml_path)
    ptc.find_region(level=0)
    # ptc.thumnail(0, level=0, thickness=1, show=True)
    # img, lab = ptc.thumnail(0, level=1, thickness=48)
    img, lab = ptc.thumnail(0, level=0, thickness=48, margin_x = 300, margin_y = 300)
    # weighted_img, weighted_lab = ptc.thumnail_weighted(0, level=1, thickness=48)

    # fig = plt.figure()
    # ax1 = fig.add_subplot(2,1,1)
    # # # ax2 = fig.add_subplot(2,1,2)
    # ax1.imshow(lab)
    # # # ax2.imshow(weighted_lab)
    # ax1.axis('off')
    # # # ax2.axis('off')
    # plt.show()

    # print(np.unique(lab))
    # print(img, lab)
    # SM = SmallPatcher(img, lab)
    SM = SmallPatcher(img, lab) #, weighted_map=weighted_lab)
    SM.random_cropping(crop_size=512, threshold=0.05)
    SM.show_candidate(filter=True)
    cand_img, cand_map = SM.get_candidate()
    not_cand_img, not_cand_map = SM.get_not_candidate()
    print(len(cand_img), len(not_cand_img))
    # print(len(cand_img), len(cand_map))
    # for i in range(len(cand_img)):
    #     print(cand_img[i], cand_map[i])


    # To check all of region per one WSI image.
    # for i in range(ptc.num_regions):
    #     ptc.thumnail(i, level=1, show=True)



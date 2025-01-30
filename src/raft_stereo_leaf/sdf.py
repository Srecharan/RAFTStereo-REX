#!/usr/bin/env python
import sys
import os
import matplotlib
import rospy
import glob
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
# matplotlib.use('agg')
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from message_filters import ApproximateTimeSynchronizer, Subscriber
# import open3d as o3d
import time
import cv2 as cv
from raftstereo.msg import depth
from yoloV8_seg.msg import masks
import skfmm
from skimage import measure
from paretoset import paretoset
import pandas as pd
import sklearn.metrics.pairwise as pdist
from scipy import signal
import time
from joblib import Parallel, delayed

HOME_DIR = os.path.expanduser('~')
# matplotlib.use('agg')
os.system("")


# from cv_bridge import CvBridge  This is causing issues with python 3 in melodic using custom msg instead

# Class of different styles
class style():
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'


class SDFSync:
    def __init__(self):
        self.image_masks = Subscriber('/leaves_masks', masks)
        self.image_depth = Subscriber('/depth_image', depth)
        self.subscriber_count = 0
        self.stereo_ = ApproximateTimeSynchronizer([self.image_masks, self.image_depth], queue_size=1, slop=0.05)
        self.stereo_.registerCallback(self.sdf_callback)
        self.img_width = 1440
        self.img_height = 1080
        self.optimal_li = -1
        self.depth = np.zeros((self.img_width, self.img_height))
        self.depth_mask = np.zeros((self.img_width, self.img_height))
        self.masks = np.zeros((self.img_width, self.img_height))
        self.kernels_ = []
        self.mn_dim = 5 / 1000  # physical dim of the micro needle ~ 5mm square
        self.graspable_mask = np.zeros((self.img_width, self.img_height))
        self.grasping_points = []

    def init_(self):  # TBD: get all of these from a launch file not from args cmd
        rospy.set_param('SDF_sub', False)
        # return args, model

    def clean_masks(self, MASKS):  # TDB debug this section. somehow small blobs are passing through
        # masks_ = []
        # index_ = np.unique(MASKS)  # TBD use unique rather than iterating through the max label
        # print('unique labels: ', index_)
        masks = MASKS
        coordinates = []
        for i in range(1, np.amax(masks).astype('uint8')):
            # for i in range(1, 2):
            # for i in range(1, 2):
            # mask_local = mask == i
            mask_local = np.where(masks, masks == i, 0)
            # labels = measure.label(mask_local)
            labels = measure.label(mask_local)
            props = measure.regionprops(labels)
            # print('no. of labels: ', len(props))
            for prop in props:
                # print('area: ', prop.area)
                area = prop.area
                if area <= 300:
                    pixels = prop.coords
                    # print('coord', pixels[0][::-1])
                    # print('coord:', pixels)
                    # mask_local[(pixels[:, 0], pixels[:, 1])] = 0
                    coordinates.append(pixels)
                    # print('------------------------')
            # masks_.append(mask_local)
            # plt.imshow(mask_local)
            # plt.show()
        # print('shape: ', np.array(masks_).shape)
        # print('coordinates: ', coordinates)
        for i in range(len(coordinates)):
            coor_ = np.array(coordinates[i])
            # print(coor_.shape)
            masks[(coor_[:, 0], coor_[:, 1])] = 0
        #
        # plt.subplot(121)
        # plt.imshow(MASKS)
        # plt.subplot(122)
        # plt.imshow(np.array(masks))
        # plt.show()
        self.masks = masks
        return masks

    def sdf_callback(self, imageMasks, imageDepth):
        print('SDF pair received ...')
        rospy.set_param('/SDF_sub', True)
        depth_image = np.asarray(imageDepth.imageData).astype('float32')
        masks_image = np.asarray(imageMasks.imageData).astype('uint8')

        depth_image_ = np.reshape(depth_image, newshape=(self.img_height, self.img_width))
        masks_image_ = np.reshape(masks_image, newshape=(self.img_height, self.img_width))

        self.depth = depth_image_
        self.image_masks = masks_image_
        self.kernels_ = self.get_kernels()

        plt.imsave(HOME_DIR + "/SDF_OUT/temp/depth_.png", depth_image_, cmap='jet')
        plt.imsave(HOME_DIR + "/SDF_OUT/temp/masks_.png", masks_image_, cmap='jet')

        # plt.subplot(121)
        # plt.imshow(depth_image_, cmap='jet')
        # plt.subplot(122)
        # plt.imshow(masks_image_, cmap='jet')
        # plt.show()

        cleaned_masks = self.clean_masks(masks_image_)
        no_leaves = np.unique(cleaned_masks)

        self.do_convolution()
        print(style.GREEN + 'no. of leaves cleaned masks: ' + style.RESET,
              len(no_leaves) - 1)  # as it includes 0 as background
        #
        # # masker = np.where(cleaned_masks, cleaned_masks == 0, 1)
        # # SDF_X = skfmm.distance(masker)
        # # plt.imshow(SDF_X)
        # # plt.imsave("/home/abhi/SDF_OUT/temp/sdf_ALL.png", SDF_X)
        # # plt.show()
        #
        # md = self.get_mean_depth(cleaned_masks, depth_image_)
        # # print('mean value for this leaf is: ', md)
        # # labels_ = np.unique(cleaned_masks)
        # # labels_ = np.delete(labels_, 0, axis=0)  # this is to remove background with val = 0
        # # DATA = np.concatenate(([md], [labels_]), axis=0)
        # # DATA_SORTED = DATA[:, DATA[0, :].argsort()]
        # bins, elements = np.histogram(md,
        #                               bins=3)  # will depend on the no of leaves in the images. This could be ittertive until non zero elements: 1-3-5-7-...
        # # print('bins: ', bins)
        # # i_ = 0
        # sum_ = 0
        # SDF = np.zeros(cleaned_masks.shape)
        # counter = 0
        # for i in bins:
        #     # print(i)
        #     sum = sum_ + i
        #     msk_ind = np.arange(sum_, sum)
        #     # mask_ind_val = np.array([for i__ in msk_ind])
        #     msk_img = np.isin(cleaned_masks, msk_ind) * cleaned_masks
        #     # print(msk_ind)
        #     sdf = self.get_sdf(msk_img, counter)
        #     SDF = SDF + sdf
        #     # plt.imshow(msk_img)
        #     # plt.imshow(sdf, alpha=0.5)
        #     # plt.show()
        #     sum_ = sum
        #     counter = counter + 1
        # SDF = SDF / counter
        # SDF = (SDF - np.amin(SDF)) / (np.amax(SDF) - np.amin(SDF))  # minmax normalization
        # plt.imsave(HOME_DIR + "/SDF_OUT/temp/sdf_FINAL.png", SDF)
        # self.get_optimal_leaf(SDF, cleaned_masks)

    def pareto_front(self, data, plot=False, option=1):
        mask = paretoset(data, sense=["max", "min"], distinct=True)
        paretoset_sols = data[mask]
        # res = np.where(mask == True)[0][0]  # or the last [-1]
        res = mask
        # print(mask)
        # print(paretoset_sols)

        if plot:
            # print(mask)
            # print(paretoset_sols)
            plt.figure(figsize=(6, 2.5))
            plt.title("Leaves in the Pareto set")
            if option == 0:
                plt.scatter(data[:, 0], data[:, 1], zorder=10, label="All leaves", s=50, alpha=0.8)
                plt.scatter(
                    paretoset_sols[:, 0],
                    paretoset_sols[:, 1],
                    zorder=5,
                    label="Optimal leaf",
                    s=150,
                    alpha=1,
                )
            else:
                plt.scatter(data["dist2minima"], data["dist2maxima"], zorder=10, label="All leaves", s=50, alpha=0.8)
                plt.scatter(
                    paretoset_sols["dist2minima"],
                    paretoset_sols["dist2maxima"],
                    zorder=5,
                    label="Optimal leaf",
                    s=150,
                    alpha=1,
                )

            plt.legend()
            plt.xlabel("dist2minima-[Maximize]")
            plt.ylabel("dist2maxima-[Minimize]")
            plt.grid(True, alpha=0.5, ls="--", zorder=0)
            plt.tight_layout()
            # plt.savefig("example_hotels.png", dpi=100)
            plt.show()
        return res

    def get_centroid(self, blob):

        index_ = np.unique(blob)
        # print('unique blobs: ', index_)

        # contours, hierarchy = cv.findContours(blob, cv.RETR_LIST, cv.CHAIN_APPROX_TC89_L1) # this is unreliable
        # print('no. of contours: ', len(contours))
        centroids = []

        for i in range(1, len(index_)):
            # calculate moments for each contour
            blob_ = blob == index_[i]
            blob_ = np.where(blob_, blob_ >= 1, 0)
            c, h = cv.findContours(blob_.astype('uint8'), cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
            M = cv.moments(c[0])
            # calculate x,y coordinate of center
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centroids.append((cX, cY))
        # print('no. of centroids: ', len(centroids))

        if len(centroids) != len(index_) - 1:
            print(style.RED + 'Warning.... some blobs are missing centroids!!!' + style.RESET)
        return centroids

    # def get_centroid(self, blob):
    #
    #     index_ = np.unique(blob)
    #     print('unique blobs: ', index_)
    #
    #     # contours, hierarchy = cv.findContours(blob, cv.RETR_LIST, cv.CHAIN_APPROX_TC89_L1)
    #     contours, hierarchy = cv.findContours(blob, 0, 2)
    #     print('no. of contours: ', len(contours))
    #     centroids = []
    #
    #     for c in contours:
    #         # calculate moments for each contour
    #         M = cv.moments(c)
    #         # calculate x,y coordinate of center
    #         cX = int(M["m10"] / M["m00"])
    #         cY = int(M["m01"] / M["m00"])
    #         centroids.append((cX, cY))
    #
    #     return centroids

    def get_optimal_leaf(self, sdf, mask):

        min_global = np.unravel_index(sdf.argmin(), sdf.shape)
        max_global = np.unravel_index(sdf.argmax(), sdf.shape)
        print('global max: ', max_global)
        print('global min: ', min_global)

        leaf_centroids = self.get_centroid(mask.astype('uint8'))
        # print(len(leaf_centroids))
        print(leaf_centroids)
        # ----------------plot results -------------------------
        plt.subplot(121)
        plt.imshow(mask)
        plt.plot(min_global[1], min_global[0], 'r*')
        plt.plot(max_global[1] + 5, max_global[0] + 5, 'y*')
        for j in range(0, len(leaf_centroids)):
            plt.plot(leaf_centroids[j][0], leaf_centroids[j][1], 'g.')
            plt.text(leaf_centroids[j][0], leaf_centroids[j][1] - 15, str(j))
            plt.text(leaf_centroids[j][0], leaf_centroids[j][1] + 15,
                     str(np.round(leaf_centroids[j][0])) + ' ,' + str(np.round(leaf_centroids[j][1])))
            plt.arrow(min_global[1], min_global[0], leaf_centroids[j][0] - min_global[1],
                      leaf_centroids[j][1] - min_global[0])

        plt.subplot(122)
        plt.imshow(mask)
        plt.plot(min_global[1], min_global[0], 'r*')
        plt.plot(max_global[1] + 5, max_global[0] + 5, 'y*')
        for j in range(0, len(leaf_centroids)):
            plt.plot(leaf_centroids[j][0], leaf_centroids[j][1], 'g.')
            plt.text(leaf_centroids[j][0], leaf_centroids[j][1] - 15, str(j))
            plt.text(leaf_centroids[j][0], leaf_centroids[j][1] + 15,
                     str(np.round(leaf_centroids[j][0])) + ' ,' + str(np.round(leaf_centroids[j][1])))
            plt.arrow(max_global[1], max_global[0], leaf_centroids[j][0] - max_global[1],
                      leaf_centroids[j][1] - max_global[0])
        plt.show()

        # ------------------ pairwise distance from minima  ----------------------

        B = np.asarray(leaf_centroids)
        B = np.insert(B, 0, values=(min_global[1], min_global[0]), axis=0)  # Python is an F*ed-up language
        pdist_B = np.array(pdist.euclidean_distances(B))
        print(B)
        np.savetxt(HOME_DIR + '/SDF_OUT/temp/pdist_to_minima.txt', pdist_B, fmt='%.2f')

        # ------------------ pairwise distance from maxima  ----------------------
        A = np.asarray(leaf_centroids)
        A = np.insert(A, 0, values=(max_global[1], max_global[0]), axis=0)  # Python is an F*ed-up language
        pdist_A = np.array(pdist.euclidean_distances(A))
        print(A)
        np.savetxt(HOME_DIR + '/SDF_OUT/temp/pdist_to_maxima.txt', pdist_A, fmt='%.2f')

        # print(pdist_B[0, :])
        # print(pdist_A[0, :])

        data = np.vstack(([pdist_B[0, :], pdist_A[0, :]])).transpose()  # first is zeros so avoid
        data = np.delete(data, 0, axis=0)
        print('data: \n', data)
        # print(data.shape)

        res_ = self.pareto_front(data, plot=True, option=0)
        opt_leaves = np.where(res_ == True)[0]
        print('optimal leaf: ', opt_leaves)

        plt.imshow(mask)
        plt.plot(min_global[1], min_global[0], 'r*')
        plt.plot(max_global[1] + 5, max_global[0] + 5, 'y*')
        for i_ in opt_leaves:
            plt.plot(leaf_centroids[i_][0], leaf_centroids[i_][1], 'g*')
        plt.show()

        # if len(res_)>1:
        # self.optimal_li = res_
        # plt.imshow(mask)
        # plt.plot(leaf_centroids[self.optimal_li][0], leaf_centroids[self.optimal_li][1], 'g*')
        # plt.plot(min_global[1], min_global[0], 'r*')
        # plt.plot(max_global[1] + 5, max_global[0] + 5, 'y*')
        # plt.show()

    def get_mean_depth(self, masks, depth_):
        index_ = np.unique(masks)
        # print('unique labels: ', index_)
        mean_depth = []

        for i in range(1, len(index_)):  # avoid the zero index
            mask_local_ = masks == index_[i]
            mask_local_ = np.where(mask_local_, mask_local_ >= 1, 0)
            mask_local = mask_local_ * depth_
            # print(mask_local.shape)
            mean_ = np.true_divide(mask_local.sum(), (mask_local != 0).sum())
            # mean_ = np.array(mean_[~numpy.isnan(mean_)])
            mean_depth.append(mean_)
        return np.array(mean_depth)

    def get_sdf(self, mask__, count):
        mask_ = np.where(mask__, mask__ == 0, 1)
        # sdf = np.zeros(mask_.shape)
        if np.count_nonzero(mask_ == 0) == 0:
            # print('len:', len(mask_ == 0))
            return np.zeros(mask_.shape)

        sdf_ = skfmm.distance(mask_, dx=1)
        # sdf__ = (sdf_ - np.amin(sdf_)) / (np.amax(sdf_) - np.amin(sdf_))  # min-max norm  !!! should not do this
        # plt.subplot(121)
        # plt.imshow(sdf__)
        # plt.subplot(122)
        # plt.hist(sdf__)
        # plt.show()
        # print("/home/buggspray/Downloads/SDF_OUT/sdf_" + str(count) + '.png')
        # plt.figure(figsize=(1080, 1440))

        # plt.imshow(im_rgb, alpha=0.5)
        # plt.imshow(mask__, alpha=0.7)
        # plt.imshow(sdf_, alpha=0.5)

        # plt.imsave("/home/abhi/SDF_OUT/temp/sdf_" + str(count) + '.png', sdf_)

        # plt.savefig("/home/buggspray/Downloads/SDF_OUT/temp/sdf_fig" + str(count) + '.png')
        return sdf_

    def get_kernels(self):
        depth_masks = self.depth
        leaves_masks = self.image_masks  # instance seg masks
        # masks_ = np.where(leaves_masks, leaves_masks > np.amin(leaves_masks), 0)  # do I need this?
        # binary_masks = leaves_masks * masks_
        masks_ind = np.unique(leaves_masks)
        # print('unique values: ', masks_ind)
        kernels = []
        plt.imshow(leaves_masks)
        plt.show()

        for i in range(1, len(masks_ind)):
            mask_a = np.where(leaves_masks, leaves_masks == masks_ind[i], 0)
            mask_b = mask_a * depth_masks
            # print(mask_b.shape)
            # plt.imshow(mask_b)
            # plt.show()
            kernel_ = self.kernel_size(mask_b, plot=False)
            kernels.append(kernel_)
        print('all kernels: ', kernels)
        return kernels

    def kernel_size(self, depth_mask, plot=False):  # depth_mask is for individual leaf
        # expects depth masked images
        # calibP = np.array(rospy.get_param("/theia/right/projection_matrix"))

        calibP = np.array(([1722.235253, 0, 584.315697, 0],
                           [0, 1722.235253, 488.690098, 0],
                           [0, 0, 1, 0, ]))
        # print('calibP: ', calibP)

        P = np.reshape(calibP, (3, 4))  # projection Matrix
        X = np.ones((4, 1))  # okay since we assume flat leaf. this should be 3x1 matrix
        mean_depth = np.true_divide(depth_mask.sum(), (depth_mask != 0).sum())
        # mean_depth_ind = np.where(np.abs())
        depth = np.array(([0, 0, mean_depth]))
        # corners of the micro needle                        z
        #      D1*---------* D2                             /
        #        |         |                               /____________ x
        #        |    *    |                               |
        #        |    Dc   |                               |
        #     D3 *---------* D4                            y

        d1 = np.array(([-self.mn_dim / 2, -self.mn_dim / 2, mean_depth]))
        d2 = np.array(([self.mn_dim / 2, -self.mn_dim / 2, mean_depth]))
        d3 = np.array(([-self.mn_dim / 2, self.mn_dim / 2, mean_depth]))
        d4 = np.array(([self.mn_dim / 2, self.mn_dim / 2, mean_depth]))

        Dc = np.array([depth]).transpose()
        D1 = np.array([d1]).transpose()
        D2 = np.array([d2]).transpose()
        D3 = np.array([d3]).transpose()
        D4 = np.array([d4]).transpose()

        # print(Dc)
        # print(D.shape)
        X[0:3, :] = np.array(Dc)
        x = np.matmul(P, X)
        xc = x / x[-1:]
        # print(np.round(xc))
        # print('---------------------------------')

        # print(D1.transpose())
        # print(D.shape)
        X[0:3, :] = np.array(D1)
        x = np.matmul(P, X)
        x1 = x / x[-1:]
        # print(np.round(x1))
        # print('---------------------------------')

        # print(D2)
        # print(D.shape)
        X[0:3, :] = np.array(D2)
        x = np.matmul(P, X)
        x2 = x / x[-1:]
        # print(np.round(x2))
        # print('---------------------------------')

        # print(D3)
        # print(D.shape)
        X[0:3, :] = np.array(D3)
        x = np.matmul(P, X)
        x3 = x / x[-1:]
        # print(np.round(x3))
        # print('---------------------------------')

        # print(D4)
        # print(D.shape)
        X[0:3, :] = np.array(D4)
        x = np.matmul(P, X)
        x4 = x / x[-1:]
        # print(np.round(x4))

        sz_1 = np.abs(np.round(x1[0]) - np.round(x2[0]))
        sz_2 = np.abs(np.round(x1[1]) - np.round(x3[1]))
        sz_3 = np.abs(np.round(x4[0]) - np.round(x3[0]))
        sz_4 = np.abs(np.round(x4[1]) - np.round(x2[1]))

        kernel_ = np.round(np.average(([sz_1, sz_2, sz_3, sz_4])))  # this is just a length of a square

        print('average kernel size: ', kernel_)
        # print('kernel width (Px):', np.round(x1[0]) - np.round(x2[0]))
        # print('kernel height(Px):', np.round(x1[1]) - np.round(x3[1]))
        # IMG = 255*np.ones((1080, 1440))
        if plot:
            plt.imshow(depth_mask)
            plt.plot(np.round(xc[0]), np.round(xc[1]), 'r.')
            plt.plot(np.round(x1[0]), np.round(x1[1]), 'r.')
            plt.plot(np.round(x2[0]), np.round(x2[1]), 'r.')
            plt.plot(np.round(x3[0]), np.round(x3[1]), 'r.')
            plt.plot(np.round(x4[0]), np.round(x4[1]), 'r.')
            plt.show()

        return kernel_

    def do_convolution(self):  # Note: 10/23/2023 this should be called at each mask at each instance of leaves
        # TBD. be sure that the index of the masks and kernel are the same
        # else, you're screwed
        # im = cv.imread('/home/buggspray/Downloads/SDF_OUT/temp/aggrigated_masks.png', cv.IMREAD_GRAYSCALE)
        # im = np.where(im, im > np.amin(im), 1)
        # kernel = np.ones((31, 31))  # thi should be dynamic based on leaf distance

        kernel_ = np.asarray(self.kernels_).astype('uint8')
        print('received kernels: ', kernel_)
        mask_ = self.masks

        print(len(kernel_))
        print(mask_.shape)

        index_ = np.unique(mask_)

        graspable_areas = np.zeros((self.img_height, self.img_width))

        for i in range(1, len(index_)):
            # print('conv mask size: ', np.ones((kernel_[i - 1], kernel_[i - 1])).astype('uint8'))
            mask_local_ = mask_ == index_[i]
            mask_local_ = np.where(mask_local_, mask_local_ >= 1, 0)
            graspable_area = signal.convolve2d(mask_local_, np.ones((kernel_[i - 1], kernel_[i - 1])), boundary='symm',
                                               mode='same')
            graspable_area = np.where(graspable_area, graspable_area < np.amax(graspable_area),
                                      1)  # remove blurry parts
            graspable_area_ = np.logical_not(graspable_area).astype(
                int)  # for some weird issue, in have to do this... never in C++
            i_, j_ = np.where(graspable_area_ == np.amax(graspable_area_))
            graspable_areas[i_, j_] = i

            # plt.imshow(mask_local_, alpha=0.9)
            # plt.subplot(121)
            # plt.imshow(mask_local_, alpha=1)
            # plt.subplot(122)
            # plt.imshow(graspable_area_)
            # plt.show()
        self.graspable_mask = graspable_areas

        print(np.unique(graspable_areas))
        plt.subplot(121)
        plt.imshow(mask_)
        plt.subplot(122)
        plt.imshow(graspable_areas)
        plt.show()
        # return graspable_area

    def do_convs(self):
        print('no. of conv kernels: ', np.array(self.kernels_).shape[0])

        grasp_areas = Parallel(n_jobs=threads)(delayed(don_conv)(i) for i in range(np.array(self.kernels_).shape[0]))  # 4X speed

    def don_conv(self, i_):
        conv_ = signal.convolve2d(self.graspable_mask, np.ones((self.kernels_[i_], self.kernels_[i_])), boundary='symm',
                                  mode='same')
        return conv_


def init():
    # print('inside the init function...')
    sdf = SDFSync()
    rospy.init_node('sdf_msg_sync', anonymous=False)
    rospy.spin()


if __name__ == '__main__':
    init()

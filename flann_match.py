import os
import cv2
import numpy as np
from matplotlib import pyplot as plt


def feature_match(template, target, save_picture=False, picture_name=None):
    """
    基于FLANN的匹配器(FLANN based Matcher)定位图片
    :param template: 模板照片
    :param target: 目标照片
    :param save_picture: 是否保存图像
    :param picture_name: 需要保存的图像名称
    :return:
    """
    # 设置最低特征点匹配数量为4(至少四个才能确定位置)
    min_match_count = 4
    # 第二个参数(-1:完整图片, 0:灰度图片, 1:彩色图片)
    template = cv2.imdecode(np.fromfile(template, dtype=np.uint8), 0)
    target = cv2.imdecode(np.fromfile(target, dtype=np.uint8), 0)
    # Initiate SIFT detector创建sift检测器
    sift = cv2.xfeatures2d.SIFT_create()
    # find the key_points and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(template, None)
    kp2, des2 = sift.detectAndCompute(target, None)
    # 创建设置FLANN匹配
    flann_index_kdtree = 0
    index_params = dict(algorithm=flann_index_kdtree, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    # 舍弃大于0.7的匹配
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    # 特征点数量至少四个才能找出匹配位置
    if len(good) >= min_match_count:
        # 获取关键点的坐标
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        # 计算变换矩阵和MASK
        m, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()
        h, w = template.shape
        # 使用得到的变换矩阵对原图像的四个角进行变换，获得在目标图像上对应的坐标
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, m)
        # 透视变换后的模板四个顶点
        coordinate_list = np.int32(dst)
        # 匹配正确
        result = True
        # 轮廓匹配的真实坐标顶点
        position = [tuple(coordinate_list[0][0]),
                    tuple(coordinate_list[1][0]),
                    tuple(coordinate_list[2][0]),
                    tuple(coordinate_list[3][0])]
        # 获取将模板包含在内的正长方形四个顶点
        upper_left_point = [min(position[0][0], position[1][0]),
                            min(position[0][1], position[3][1])]
        lower_right_point = [max(position[2][0], position[3][0]),
                             max(position[1][1], position[2][1])]
        width = lower_right_point[0] - upper_left_point[0]
        height = lower_right_point[1] - upper_left_point[1]
        lower_left_point = [upper_left_point[0], upper_left_point[1] + height]
        upper_right_point = [upper_left_point[0] + width, upper_left_point[1]]
        center_x = upper_left_point[0] + int(width / 2)
        center_y = upper_left_point[1] + int(height / 2)
        # 中点坐标
        center = (center_x, center_y)
        # 矫正后的正四边形顶点
        correction_boundary = [tuple(upper_left_point),
                               tuple(lower_left_point),
                               tuple(lower_right_point),
                               tuple(upper_right_point)]
        '''画图(画出轮廓, 中心点, 校正后的正长方形)'''
        # 标记匹配位置
        cv2.polylines(target, [coordinate_list], True, 255, 1, cv2.LINE_AA)
        # 画出中心点
        cv2.circle(target, center, 5, (0, 0, 255), 4)
        # 画出将模板包含在内的正长方形
        contour_point = np.array([[upper_left_point, lower_left_point, lower_right_point, upper_right_point]],
                                 dtype=np.int32)
        cv2.polylines(target, contour_point, 1, 255)
    # 未匹配到
    else:
        print("Not enough matches are found - %d/%d" % (len(good), min_match_count))
        matches_mask = None
        result = False
        position = [(0, 0), (0, 0), (0, 0), (0, 0)]
        center = (0, 0)
        correction_boundary = [(0, 0), (0, 0), (0, 0), (0, 0)]
    # 画出特征映射图
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=None,
                       matchesMask=matches_mask,
                       flags=2)
    result_pic = cv2.drawMatches(template, kp1, target, kp2, good, None, **draw_params)
    # 判断是否需要保存图片
    if save_picture is True:
        # 保存对比图片
        if picture_name is not None:
            # 确保存放文件的路径存在
            picture_path = os.path.dirname(picture_name)
            if picture_path != '':
                if os.path.exists(picture_path) is False:
                    os.makedirs(picture_path)
            # 处理后缀问题
            if '.' in picture_name:
                picture_type = picture_name.split('.')[1]
                cv2.imencode('.' + picture_type, result_pic)[1].tofile(picture_name)
            else:
                cv2.imencode('.jpg', result_pic)[1].tofile(picture_name + '.jpg')
        else:
            cv2.imencode('.jpg', result_pic)[1].tofile('match.jpg')
    # result: 是否匹配上, feature_points_number: 特征点数量, center: 中心坐标, position: 匹配到的位置, correction_boundary: 矫正后的正长方形
    return {'result': result,
            'feature_points_number': len(good),
            'center': center,
            'position': position,
            'correction_boundary': correction_boundary}


if __name__ == '__main__':
    # template_pic = 'picture/img3.jpg'
    # target_pic = 'picture/img4.jpg'
    template_pic = 'picture/template.jpg'
    target_pic = 'picture/target_bevel.jpg'
    match_result = feature_match(template_pic, target_pic, save_picture=False, picture_name='picture/123/match')
    print(match_result)

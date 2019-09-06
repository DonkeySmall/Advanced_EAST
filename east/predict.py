import argparse

import numpy as np
from PIL import Image, ImageDraw
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import cv2
import cfg
from .label import point_inside_of_quad
#from .network import East
from .preprocess import resize_image
from .nms import nms


def sigmoid(x):
    """`y = 1 / (1 + exp(-x))`"""
    return 1 / (1 + np.exp(-x))


def cut_text_line(geo, scale_ratio_w, scale_ratio_h, im_array, img_path, s, result_im,x_offset):
    geo /= [scale_ratio_w, scale_ratio_h]
    p_min = np.amin(geo, axis=0)
    p_max = np.amax(geo, axis=0)
    min_xy = p_min.astype(int)
    max_xy = p_max.astype(int) + 2
    LUpper = geo[0][0]-min_xy[0], geo[0][1]-min_xy[1]
    LDown = geo[1][0]-min_xy[0], geo[1][1]-min_xy[1]
    RDown = geo[2][0]-min_xy[0], geo[2][1]-min_xy[1]
    RUpper = geo[3][0]-min_xy[0], geo[3][1]-min_xy[1]
    
    sub_im_arr = im_array[min_xy[1]:max_xy[1], min_xy[0]:max_xy[0], :].copy()
    sub_im_arr = np.array(sub_im_arr, dtype=np.uint8)
    W, H, _ = sub_im_arr.shape
    mask = Image.new('RGB',(H, W))
    draw = ImageDraw.Draw(mask)
    draw.polygon([LUpper, RUpper, RDown, LDown], fill=(255,255,255))
    mask = np.array(mask,dtype=np.uint8)
    sub_im_arr = np.where(mask==(255,255,255),sub_im_arr,0)
    #sub_im = cv2.bitwise_and(sub_im_arr,sub_im_arr,mask = mask)
    sub_im = Image.fromarray(sub_im_arr)
    """
    #sub_im.show()
    #draw.line((0, 0) + im.size, fill=128)
    #draw.line((0, im.size[1], im.size[0], 0), fill=128)
    for m in range(min_xy[1], max_xy[1]):
        for n in range(min_xy[0], max_xy[0]):
            if not point_inside_of_quad(n, m, geo, p_min, p_max):
                sub_im_arr[m - min_xy[1], n - min_xy[0], :] = 255
    sub_im = image.array_to_img(sub_im_arr, scale=False)
    """
    result_im.paste(sub_im, (0,x_offset))
    coord = [0, x_offset, max_xy[0]-min_xy[0], x_offset + max_xy[1]-min_xy[1]]
    # Update X coordinate
    x_offset += max_xy[1]-min_xy[1]
    
    return x_offset, coord

def get_pillow_image_size(scale_ratio_w, scale_ratio_h, quad_scores, quad_after_nms,maxWidth,maxHeight):
    for score, geo in zip(quad_scores, quad_after_nms):
        if np.amin(score) > 0:
            geo_tmp = geo / [scale_ratio_w, scale_ratio_h]
            p_min = np.amin(geo_tmp, axis=0)
            p_max = np.amax(geo_tmp, axis=0)
            min_xy = p_min.astype(int)
            max_xy = p_max.astype(int) + 2
            maxWidth = max(maxWidth,max_xy[0]-min_xy[0])            
            maxHeight += max_xy[1]-min_xy[1]
    return maxWidth, maxHeight


def predict(east_detect, img_path, pixel_threshold, quiet=False):
    img = image.load_img(img_path)
    d_wight, d_height = resize_image(img, cfg.max_predict_img_size)
    img = img.resize((d_wight, d_height), Image.NEAREST).convert('RGB')
    img = image.img_to_array(img)
    img = preprocess_input(img, mode='tf')
    x = np.expand_dims(img, axis=0)
    y = east_detect.predict(x)

    y = np.squeeze(y, axis=0)
    y[:, :, :3] = sigmoid(y[:, :, :3])
    cond = np.greater_equal(y[:, :, 0], pixel_threshold)
    activation_pixels = np.where(cond)
    quad_scores, quad_after_nms = nms(y, activation_pixels)
    with Image.open(img_path) as im:
        im_array = image.img_to_array(im.convert('RGB'))
        d_wight, d_height = resize_image(im, cfg.max_predict_img_size)
        scale_ratio_w = d_wight / im.width
        scale_ratio_h = d_height / im.height
        im = im.resize((d_wight, d_height), Image.NEAREST).convert('RGB')
        quad_im = im.copy()

        draw = ImageDraw.Draw(im)
        for i, j in zip(activation_pixels[0], activation_pixels[1]):
            px = (j + 0.5) * cfg.pixel_size
            py = (i + 0.5) * cfg.pixel_size
            line_width, line_color = 1, 'red'
            if y[i, j, 1] >= cfg.side_vertex_pixel_threshold:
                if y[i, j, 2] < cfg.trunc_threshold:
                    line_width, line_color = 2, 'yellow'
                elif y[i, j, 2] >= 1 - cfg.trunc_threshold:
                    line_width, line_color = 2, 'green'
            draw.line([(px - 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size),
                       (px + 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size),
                       (px + 0.5 * cfg.pixel_size, py + 0.5 * cfg.pixel_size),
                       (px - 0.5 * cfg.pixel_size, py + 0.5 * cfg.pixel_size),
                       (px - 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size)],
                      width=line_width, fill=line_color)
            
            
        im.save(img_path + '_act.jpg')
        quad_draw = ImageDraw.Draw(quad_im)
        
        
        maxWidth = -1
        maxHeight = 0
        
        maxWidth, maxHeight = get_pillow_image_size(scale_ratio_w, scale_ratio_h, quad_scores, quad_after_nms,maxWidth,maxHeight)
            
        result_im = Image.new('RGB',(maxWidth, maxHeight))
        
        
        x_offset = 0
        
        txt_items = []
        four_txt_items=[]
        for score, geo, s in zip(quad_scores, quad_after_nms,range(len(quad_scores))):
            if np.amin(score) > 0:
                quad_draw.line([tuple(geo[0]),
                                tuple(geo[1]),
                                tuple(geo[2]),
                                tuple(geo[3]),
                                tuple(geo[0])], width=2, fill='red')
                
                x_offset,coord = cut_text_line(geo, scale_ratio_w, scale_ratio_h, im_array,
                              img_path, s, result_im, x_offset)
                rescaled_geo = geo / [scale_ratio_w, scale_ratio_h]
                rescaled_geo_list = np.reshape(rescaled_geo, (8,)).tolist()
                txt_item = ','.join(map(str, rescaled_geo_list))
                four_txt_item = ','.join(map(str, coord))
                txt_items.append(txt_item + '\n')
                four_txt_items.append(four_txt_item + '\n')
            elif not quiet:
                print('quad invalid with vertex num less then 4.')
        quad_im.save(img_path + '_predict.jpg')
        if cfg.predict_write2txt and len(txt_items) > 0:
            with open(img_path[:-4] + '.txt', 'w') as f_txt:
                f_txt.writelines(txt_items)
        # Convert to parse format
        result_txt = []
        for i in range(len(four_txt_items)):
            txt = four_txt_items[i]
            result_txt.append(list(txt.strip().split(',')))
        
        if cfg.predict_cut_text_line:
            result_im.save('demo.jpg')
        return result_txt, quad_im


def predict_txt(east_detect, img_path, txt_path, pixel_threshold, quiet=False):
    img = image.load_img(img_path)
    d_wight, d_height = resize_image(img, cfg.max_predict_img_size)
    scale_ratio_w = d_wight / img.width
    scale_ratio_h = d_height / img.height
    img = img.resize((d_wight, d_height), Image.NEAREST).convert('RGB')
    img = image.img_to_array(img)
    img = preprocess_input(img, mode='tf')
    x = np.expand_dims(img, axis=0)
    y = east_detect.predict(x)

    y = np.squeeze(y, axis=0)
    y[:, :, :3] = sigmoid(y[:, :, :3])
    cond = np.greater_equal(y[:, :, 0], pixel_threshold)
    activation_pixels = np.where(cond)
    quad_scores, quad_after_nms = nms(y, activation_pixels)

    txt_items = []
    for score, geo in zip(quad_scores, quad_after_nms):
        if np.amin(score) > 0:
            rescaled_geo = geo / [scale_ratio_w, scale_ratio_h]
            rescaled_geo_list = np.reshape(rescaled_geo, (8,)).tolist()
            txt_item = ','.join(map(str, rescaled_geo_list))
            txt_items.append(txt_item + '\n')
        elif not quiet:
            print('quad invalid with vertex num less then 4.')
    if cfg.predict_write2txt and len(txt_items) > 0:
        with open(txt_path, 'w') as f_txt:
            f_txt.writelines(txt_items)


"""
if __name__ == '__main__':
    args = parse_args()
    img_path = args.path
    threshold = float(args.threshold)
    print(img_path, threshold)
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]='5'

    east = East()
    east_detect = east.east_network()
    east_detect.load_weights(cfg.saved_model_weights_file_path)
    predict(east_detect, img_path, threshold)
"""
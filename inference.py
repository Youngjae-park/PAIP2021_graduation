import numpy as np
import torch
import segmentation_models_pytorch as smp
from configparser import ConfigParser
import os, time, openslide, tqdm
import cv2
import torch.nn as nn

parser = ConfigParser()
parser.read('config.ini')



def inference(train_case, model_type, encoder_name,
              pooling_type, loss_type, lr, batch_size,
              experiment_name, multiple, patch_size=512):

    ckpt_path = f'./saved_model/{experiment_name}_best.pth'
    checkpoint = torch.load(ckpt_path)

    model = checkpoint['model']
    model.cuda()
    model.eval()

    if pooling_type == 'max':
        pool = nn.AdaptiveMaxPool2d(1)

    # svs path setting
    v_path = '/home/dhodwo/young/PAIP2021/dataset/svs_folder'

    # Param setting
    patch_size = 512
    window_step = patch_size//2
    out_size = patch_size//16
    threshold = 0.9

    cancer_list = ['Col', 'Pan', 'Pros']
    idx_list = [46, 47, 48, 49, 50]

    with torch.no_grad():
        for cancer in cancer_list:
            for idx in idx_list:
                slide = openslide.OpenSlide(v_path+f'/{cancer}_PNI2021chall_train_{str(idx).zfill(4)}.svs')
                print(v_path+f'/{cancer}_PNI2021chall_train_{str(idx).zfill(4)}.svs')

                img = slide.read_region((0,0), 0, slide.level_dimensions[0])
                print("image loaded")
                rgb_img = np.asarray(img.convert('RGB'))
                img = np.asarray(img.convert('L'))
                w, h, _ = rgb_img.shape
                max_w = (w//patch_size)*patch_size
                max_h = (h//patch_size)*patch_size
                accum = np.zeros(slide.level_dimensions[2][::-1], dtype=np.float32)
                counts = np.zeros(slide.level_dimensions[2][::-1], dtype=np.float32)

                _, th_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

                patch_list = []
                coord_list = []

                for h in tqdm.tqdm(range(0, max_h-patch_size+1, window_step)):
                    for w in range(0, max_w-patch_size+1, window_step):
                        indices, cnt = np.unique(np.where(th_otsu[w:w+patch_size, h:h+patch_size]<10, 0, 1), return_counts=True)
                        # print(idx, cnt)
                        if len(cnt) > 1:
                            if not cnt[1] > (cnt[0]+cnt[1]) * threshold: # Threshold value
                                cropped_img = rgb_img[w:w+patch_size, h:h+patch_size, :]
                                cropped_img = np.expand_dims(cropped_img, axis=0)

                                coord = np.asarray([w//16, h//16], dtype=np.int32)
                                coord = np.expand_dims(coord, axis=0)
                                patch_list.append(cropped_img)
                                coord_list.append(coord)
                        else:
                            if 1 in indices:
                                cropped_img = rgb_img[w:w + patch_size, h:h + patch_size, :]
                                cropped_img = np.expand_dims(cropped_img, axis=0)

                                coord = np.asarray([w//16, h//16], dtype=np.int32)
                                coord = np.expand_dims(coord, axis=0)
                                patch_list.append(cropped_img)
                                coord_list.append(coord)
                        if len(coord_list)==64:
                            patch_list = np.concatenate(patch_list, axis=0)
                            coord_list = np.concatenate(coord_list, axis=0)
                            cropped_img = np.transpose(patch_list/255., [0, 3, 1, 2])
                            cropped_img = torch.tensor(cropped_img, dtype=torch.float32).cuda()
                            out_img = model(cropped_img)
                            # pooled_img = pool(out_img).detach().cpu().numpy()
                            out_img = out_img.detach().cpu().numpy()
                            # print(out_img.shape)
                            for i in range(64):
                                # if min(pooled_img[i])>0.5:
                                # if pooled_img[i, 1]>0.5:
                                x, y = coord_list[i]
                                accum[x:x+out_size, y:y+out_size] += out_img[i,0]
                                counts[x:x+out_size, y:y+out_size] += 1
                            patch_list = []
                            coord_list = []

                counts += 1e-7
                final_output = accum/counts
                s_path = f'./results/{experiment_name}/'
                f_name = 'lv2_{}_{}'.format(cancer, str(idx))
                if not os.path.isdir(s_path):
                    os.makedirs(s_path, exist_ok=True)
                cv2.imwrite(s_path+f'{f_name}_0.25.jpg', (final_output>0.25).astype(np.int32)*255)
                cv2.imwrite(s_path+f'{f_name}_0.5.jpg', (final_output>0.5).astype(np.int32)*255)
                cv2.imwrite(s_path+f'{f_name}_0.75.jpg', (final_output>0.75).astype(np.int32)*255)
                cv2.imwrite(s_path+f'{f_name}_raw.jpg', (final_output*255).astype(np.int32))

def experiment_info(section):
    print(f"##### Experiment: {section} #####")
    for key in parser.options(section):
        data = parser.get(section, key)
        print(f'{key} => {data}')
    print('###############################\n')

if __name__ == "__main__":
    experiment_idx_list = [10, 11, 9, 5, 6, 7, 8]
    for i in experiment_idx_list:
        experiment_info(f'train{i}')

        learning_rate = parser.get(f'train{i}', 'learning_rate')
        batch_size = parser.get(f'train{i}', 'batch_size')
        model_type = parser.get(f'train{i}', 'model_type')
        encoder_name = parser.get(f'train{i}', 'encoder_name')
        loss_type = parser.get(f'train{i}', 'loss_type')
        pooling_type = parser.get(f'train{i}', 'pooling_type')
        experiment_name = parser.get(f'train{i}', 'experiment_name')
        multiple = parser.get(f'train{i}', 'multiple')

        if multiple == 'True':
            multiple = True
        elif multiple == 'False':
            multiple = False
        else:
            print("Not valid multiple parameter!!")

        try:
            print("Inference started!!")
            inference(f'train{i}', model_type, encoder_name, pooling_type, loss_type, learning_rate, batch_size, experiment_name, multiple)
        except Exception as ex:
            print("Error: ",ex)
            print("Skipped the experiment!!")

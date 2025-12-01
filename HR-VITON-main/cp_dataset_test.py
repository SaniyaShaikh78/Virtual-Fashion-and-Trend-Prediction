import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import os.path as osp
import numpy as np
import json

def safe_paste(base_img, paste_img, mask_array):
    mask = Image.fromarray(np.uint8(mask_array * 255), 'L')
    if paste_img.size != base_img.size:
        paste_img = paste_img.resize(base_img.size)
    if mask.size != base_img.size:
        mask = mask.resize(base_img.size)
    base_img.paste(paste_img, None, mask)

class CPDatasetTest(data.Dataset):
    def __init__(self, opt):
        super(CPDatasetTest, self).__init__()
        self.opt = opt
        self.root = opt.dataroot
        self.datamode = opt.datamode
        self.data_list = opt.data_list
        self.fine_height = opt.fine_height
        self.fine_width = opt.fine_width
        self.semantic_nc = opt.semantic_nc
        self.data_path = osp.join(opt.dataroot, opt.datamode)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        im_names, c_names = [], []
        with open(osp.join(opt.dataroot, opt.data_list), 'r') as f:
            for line in f.readlines():
                im_name, c_name = line.strip().split()
                im_names.append(im_name)
                c_names.append(c_name)

        self.im_names = im_names
        self.c_names = {"paired": im_names, "unpaired": c_names}

    def name(self):
        return "CPDataset"

    def __len__(self):
        return len(self.im_names)

    def get_parse_agnostic(self, parse, pose_data):
        parse_array = np.array(parse)
        parse_upper = ((parse_array == 5).astype(np.float32) +
                       (parse_array == 6).astype(np.float32) +
                       (parse_array == 7).astype(np.float32))
        parse_neck = (parse_array == 10).astype(np.float32)

        r = 10
        agnostic = parse.copy()

        for parse_id, pose_ids in [(14, [2, 5, 6, 7]), (15, [5, 2, 3, 4])]:
            mask_arm = Image.new('L', (self.fine_width, self.fine_height), 'black')
            mask_arm_draw = ImageDraw.Draw(mask_arm)
            i_prev = pose_ids[0]
            for i in pose_ids[1:]:
                if np.any(pose_data[[i_prev, i], :2] == 0.0):
                    continue
                mask_arm_draw.line([tuple(pose_data[j]) for j in [i_prev, i]], 'white', width=r*10)
                pointx, pointy = pose_data[i]
                radius = r*4 if i == pose_ids[-1] else r*15
                mask_arm_draw.ellipse((pointx-radius, pointy-radius, pointx+radius, pointy+radius), 'white', 'white')
                i_prev = i
            parse_arm = (np.array(mask_arm) / 255) * (parse_array == parse_id).astype(np.float32)
            safe_paste(agnostic, parse, parse_arm)

        safe_paste(agnostic, parse, parse_upper)
        safe_paste(agnostic, parse, parse_neck)
        return agnostic

    def get_agnostic(self, im, im_parse, pose_data):
        parse_array = np.array(im_parse)
        parse_head = ((parse_array == 4).astype(np.float32) + (parse_array == 13).astype(np.float32))
        parse_lower = ((parse_array == 9).astype(np.float32) +
                       (parse_array == 12).astype(np.float32) +
                       (parse_array == 16).astype(np.float32) +
                       (parse_array == 17).astype(np.float32) +
                       (parse_array == 18).astype(np.float32) +
                       (parse_array == 19).astype(np.float32))

        agnostic = im.copy()
        agnostic_draw = ImageDraw.Draw(agnostic)

        length_a = np.linalg.norm(pose_data[5] - pose_data[2])
        length_b = np.linalg.norm(pose_data[12] - pose_data[9])
        point = (pose_data[9] + pose_data[12]) / 2
        pose_data[9] = point + (pose_data[9] - point) / length_b * length_a
        pose_data[12] = point + (pose_data[12] - point) / length_b * length_a
        r = int(length_a / 16) + 1

        for i in [9, 12]:
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse((pointx-r*3, pointy-r*6, pointx+r*3, pointy+r*6), 'gray', 'gray')
        agnostic_draw.line([tuple(pose_data[i]) for i in [2, 9]], 'gray', width=r*6)
        agnostic_draw.line([tuple(pose_data[i]) for i in [5, 12]], 'gray', width=r*6)
        agnostic_draw.line([tuple(pose_data[i]) for i in [9, 12]], 'gray', width=r*12)
        agnostic_draw.polygon([tuple(pose_data[i]) for i in [2, 5, 12, 9]], 'gray', 'gray')

        pointx, pointy = pose_data[1]
        agnostic_draw.rectangle((pointx-r*5, pointy-r*9, pointx+r*5, pointy), 'gray', 'gray')

        agnostic_draw.line([tuple(pose_data[i]) for i in [2, 5]], 'gray', width=r*12)
        for i in [2, 5]:
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse((pointx-r*5, pointy-r*6, pointx+r*5, pointy+r*6), 'gray', 'gray')
        for i in [3, 4, 6, 7]:
            if np.any(pose_data[[i-1, i], :2] == 0.0):
                continue
            agnostic_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 'gray', width=r*10)
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'gray', 'gray')

        for parse_id, pose_ids in [(14, [5, 6, 7]), (15, [2, 3, 4])]:
            mask_arm = Image.new('L', (768, 1024), 'white')
            mask_arm_draw = ImageDraw.Draw(mask_arm)
            for i in pose_ids:
                if i != pose_ids[0] and np.any(pose_data[[i-1, i], :2] == 0.0):
                    continue
                mask_arm_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 'black', width=r*10)
                pointx, pointy = pose_data[i]
                mask_arm_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'black', 'black')
            parse_arm = (np.array(mask_arm) / 255) * (parse_array == parse_id).astype(np.float32)
            safe_paste(agnostic, im, parse_arm)

        safe_paste(agnostic, im, parse_head)
        safe_paste(agnostic, im, parse_lower)
        return agnostic
    
    def __getitem__(self, index):
        im_name = self.im_names[index]
        c_name = {}
        c = {}
        cm = {}

        for key in self.c_names:
                c_name[key] = self.c_names[key][index]
                if key == "paired":
                    c[key] = Image.open(osp.join(self.data_path, 'image', c_name[key])).convert('RGB')
                else:
                    c[key] = Image.open(osp.join(self.data_path, 'cloth', c_name[key])).convert('RGB')
                c[key] = transforms.Resize(self.fine_width, interpolation=2)(c[key])

                if key == "paired":
                    cm[key] = Image.open(osp.join(self.data_path, 'image-parse-v3', c_name[key]).replace('.jpg', '.png'))
                else:
                    cm[key] = Image.open(osp.join(self.data_path, 'cloth-mask', c_name[key]))

                cm[key] = transforms.Resize(self.fine_width, interpolation=0)(cm[key])

                c[key] = self.transform(c[key])  # [-1,1]
                cm_array = np.array(cm[key])
                cm_array = (cm_array >= 128).astype(np.float32)
                cm[key] = torch.from_numpy(cm_array).unsqueeze(0)

        im_pil_big = Image.open(osp.join(self.data_path, 'image', im_name))
        im_pil = transforms.Resize(self.fine_width, interpolation=2)(im_pil_big)
        im = self.transform(im_pil)

        parse_name = im_name.replace('.jpg', '.png')
        im_parse_pil_big = Image.open(osp.join(self.data_path, 'image-parse-v3', parse_name))
        im_parse_pil = transforms.Resize(self.fine_width, interpolation=0)(im_parse_pil_big)

        parse = torch.from_numpy(np.array(im_parse_pil)[None]).long()
        im_parse = self.transform(im_parse_pil.convert('RGB'))

        pose_name = im_name.replace('.jpg', '_keypoints.json')
        with open(osp.join(self.data_path, 'openpose_json', pose_name), 'r') as f:
                pose_label = json.load(f)
                pose_data = np.array(pose_label['people'][0]['pose_keypoints_2d']).reshape((-1, 3))[:, :2]

        densepose_name = im_name.replace('image', 'image-densepose')
        densepose_map = Image.open(osp.join(self.data_path, 'image-densepose', densepose_name))
        densepose_map = transforms.Resize(self.fine_width, interpolation=2)(densepose_map)
        densepose_map = self.transform(densepose_map)

        agnostic = self.get_agnostic(im_pil_big, im_parse_pil_big, pose_data)
        agnostic = transforms.Resize(self.fine_width, interpolation=2)(agnostic)
        agnostic = self.transform(agnostic)

        parse = Image.open(osp.join(self.data_path, 'image-parse-v3', parse_name))
        parse = transforms.Resize(self.fine_width, interpolation=0)(parse)
        parse_agnostic = self.get_parse_agnostic(parse, pose_data)
        parse_agnostic = torch.from_numpy(np.array(parse_agnostic)[None]).long()

        parse_agnostic_map = torch.zeros(20, self.fine_height, self.fine_width, dtype=torch.float)
        parse_agnostic_map.scatter_(0, parse_agnostic, 1.0)

        new_parse_agnostic_map = torch.zeros(self.semantic_nc, self.fine_height, self.fine_width)
        labels = {
                0: ['background', [0, 10]],
                1: ['hair', [1, 2]],
                2: ['face', [4, 13]],
                3: ['upper', [5, 6, 7]],
                4: ['bottom', [9, 12]],
                5: ['left_arm', [14]],
                6: ['right_arm', [15]],
                7: ['left_leg', [16]],
                8: ['right_leg', [17]],
                9: ['left_shoe', [18]],
                10: ['right_shoe', [19]],
                11: ['socks', [8]],
                12: ['noise', [3, 11]]
            }

        for i in range(len(labels)):
                for label in labels[i][1]:
                    new_parse_agnostic_map[i] += parse_agnostic_map[label]

        return {
            'c_name': c_name,
            'im_name': im_name,
            'cloth': c,
            'cloth_mask': cm,
            'parse_agnostic': new_parse_agnostic_map,
            'densepose': densepose_map,
            'image': im,
            'agnostic': agnostic,
            'parse': im_parse  # ✅ This fixes the KeyError
        }

class CPDataLoader(object):
    def __init__(self, opt, dataset):
        super(CPDataLoader, self).__init__()
        if opt.shuffle:
            train_sampler = torch.utils.data.sampler.RandomSampler(dataset)
        else:
            train_sampler = None

        self.data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
            num_workers=opt.workers, pin_memory=True, drop_last=True, sampler=train_sampler)
        self.dataset = dataset
        self.data_iter = iter(self.data_loader)

    def next_batch(self):
        try:
            return next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.data_loader)
            return next(self.data_iter)

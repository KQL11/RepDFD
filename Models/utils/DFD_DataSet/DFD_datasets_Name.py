import os
import random
import numpy as np
import cv2
import math
import glob
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
import torchvision
import torch
import imgaug.augmenters as iaa


def jpeg_compression(image):
    if random.random() < 0.5:
        return image
    quality = random.randint(60, 100)
    tmp = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    tmp = cv2.imencode('.jpg', tmp, encode_param)[1]
    tmp = cv2.imdecode(tmp, cv2.IMREAD_COLOR)
    image_jpeg = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
    return image_jpeg


class SimplyDataset(Dataset):
    def __init__(self, txt_path, transform=None, target_transform=None):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.open(fn).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.imgs)


class SimpleDatasets(Dataset):
    def __init__(self, video_path, transform=None):
        self.video_path = video_path
        self.face_images = self.load_image()
        self.transform = transform

    def load_image(self):
        rgb_frames = []
        for i, filename in enumerate(glob.glob(self.video_path + '/*')):
            try:
                frame = Image.open(filename).convert('RGB')
                rgb_frames.append(frame)
            except:
                pass
        return rgb_frames

    def __getitem__(self, index):
        image = self.face_images[index]
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.face_images)


class FFDataset(Dataset):
    def __init__(self, root_path="", video_names=[], phase='train', test_frame_nums=5, vid_sample=1, is_pillow=True,
                 num_class=2, transform=None, aug=False):
        assert phase in ['train', 'valid', 'test']
        self.is_pillow = is_pillow
        self.vid_sample = vid_sample
        self.root_path = root_path
        self.video_names = video_names
        self.phase = phase
        self.num_classes = num_class
        self.transform = transform
        self.test_frame_nums = test_frame_nums
        self.videos_by_class = self.load_video_name()  # load all video name
        if phase != 'train':
            image_name = self.load_image_name()
            self.image_name = []
            for name in image_name:
                new_name = name.replace('/data2/linkaiqing/code/Reprogramming/dataset/', 
                                        '/data2/linkaiqing/code/Reprogramming/dataset/Face_Embedding/Transface/')
                if os.path.exists(new_name.replace('.png', '.pt')):
                    self.image_name.append(name)
            # print(1)
        else:
            self.num_real_class = len(self.videos_by_class['real'])
            self.num_fake_class = len(self.videos_by_class['fake'])

        self.aug = aug
        # self.augmenter = iaa.JpegCompression(
        #     compression=(60, 100), seed=7
        # )
        self.augmenter = jpeg_compression

    def load_video_name(self):
        videos_by_class = {}

        real_videos = []
        fake_videos = []
        for video_name in tqdm(self.video_names):
            all_image_name = os.listdir(video_name)
            if len(all_image_name) < 1:
                continue

            if video_name.find('mp4') == -1:
                continue
            if video_name.find('youtube') != -1:  # video is from youtube
                real_videos.append(video_name)
            else:  # video is fake
                fake_videos.append(video_name)
        videos_by_class['real'] = random.sample(real_videos, int(self.vid_sample * len(real_videos)))
        videos_by_class['fake'] = random.sample(fake_videos, int(self.vid_sample * len(fake_videos)))
        return videos_by_class

    def load_image_name(self):
        image_names = []
        for video_name in tqdm(self.videos_by_class['real'] + self.videos_by_class['fake']):
            video_path = os.path.join(self.root_path, video_name)
            random.seed(2022)
            frame_names = os.listdir(video_path)
            if len(frame_names) > self.test_frame_nums:
                frame_names = random.sample(frame_names, self.test_frame_nums)
            for image_name in frame_names:
                image_names.append(os.path.join(video_name, image_name))
        return image_names

    # def load_landmarks(self, landmarks_file):
    #     """
    #
    #     :param landmarks_file: input landmarks json file name
    #     :return: all_landmarks: having the shape of 64x2 list. represent left eye,
    #                             right eye, noise, left lip, right lip
    #     """
    #     all_landmarks = OrderedDict()
    #     with open(landmarks_file, "r", encoding="utf-8") as file:
    #         line = file.readline()
    #         while line:
    #             line = json.loads(line)
    #             all_landmarks[line["image_name"]] = np.array(line["landmarks"])
    #             line = file.readline()
    #     return all_landmarks

    def pil_loader(self, image_path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(image_path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
        # img = Image.open(image_path)
        # return img

    def cv2_loader(self, image_path):
        img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        # img = Image.fromarray(img)
        return img

    def __getitem__(self, index):
        if self.phase == 'train':
            if index % 2 == 0:  # load real face image
                video_name = self.videos_by_class['real'][min(index // 2, self.num_real_class)]
                label = 0  # real label is 0
            else:  # load fake face image
                video_name = self.videos_by_class['fake'][min(index // 2, self.num_fake_class)]
                label = 1  # fake label is 1

            video_path = os.path.join(self.root_path, video_name)
            image_names = os.listdir(video_path)
            image_name = random.sample(image_names, 1)[0]
            image_path = os.path.join(video_path, image_name)
            if self.is_pillow:
                image = self.pil_loader(image_path)
                if self.aug:
                    # The augmentation will be applied to the image with probability 0.5.
                    image = np.array(image)
                    image = self.augmenter(image=image)
                    image = Image.fromarray(image)

                # for _, t in enumerate(self.transform.transforms):
                #     image = t(image)

                image = self.transform(image)
            else:
                image = self.cv2_loader(image_path)
                if self.aug:
                    image = self.augmenter(image=image)
                # image = self.transform(image=image)['image']
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                image = Image.fromarray(image)
                image = self.transform(image)
            # print(f"{image_name.replace('.png', '.pt')}")
            # Face Embed Load
            face_emb = torch.load(os.path.join(video_path.replace('/data2/linkaiqing/code/Reprogramming/dataset',
                                                                  '/data2/linkaiqing/code/Reprogramming/dataset/Face_Embedding/Transface'),
                                               image_name.replace('.png', '.pt')), map_location='cpu')
            # print(f"{image_name.replace('.png', '.pt')}")

            return image, label, face_emb, image_path

        else:
            image_path = os.path.join(self.root_path, self.image_name[index])
            if self.is_pillow:
                image = self.pil_loader(image_path)
                image = self.transform(image)
            else:
                image = self.cv2_loader(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                image = Image.fromarray(image)
                image = self.transform(image)
                # image = self.transform(image=image)['image']
            if self.image_name[index].find('youtube') != -1:
                label = 0  # real label is 0
            else:
                label = 1  # fake label is 1

            # Face Embed Load
            face_emb = torch.load(((image_path.replace('/data2/linkaiqing/code/Reprogramming/dataset',
                                                       '/data2/linkaiqing/code/Reprogramming/dataset/Face_Embedding/Transface').replace(
                '.png', '.pt'))), map_location='cpu')
            return image, label, face_emb, image_path

    def __len__(self):
        if self.phase == 'train':  # load Rebalanced image data
            return self.num_real_class + self.num_fake_class
        else:  # load all image
            return len(self.image_name)


class CDFDataset(Dataset):
    def __init__(self, root_path="", video_names=[], phase='train', test_frame_nums=10, vid_sample=1, is_pillow=True,
                 num_class=2, transform=None):
        assert phase in ['train', 'valid', 'test']
        self.is_pillow = is_pillow
        self.vid_sample = vid_sample
        self.root_path = root_path
        self.video_names = video_names
        self.phase = phase
        self.num_classes = num_class
        self.transform = transform
        self.test_frame_nums = test_frame_nums
        self.videos_by_class = self.load_video_name()  # load all video name
        if phase != 'train':
            image_name = self.load_image_name()
            self.image_name = []
            for name in image_name:
                new_name = name.replace('/data2/linkaiqing/code/Reprogramming/dataset',
                                        '/data2/linkaiqing/code/Reprogramming/dataset/Face_Embedding/Transface')
                if os.path.exists(new_name.replace('.png', '.pt')):
                    self.image_name.append(name)
            print(1)
        else:
            if len(self.videos_by_class['real']) < len(self.videos_by_class['fake']):
                self.smallest_class = 'real'
                self.largest_class = 'fake'
            else:
                self.smallest_class = 'fake'
                self.largest_class = 'real'
            print('The number of real videos is : %d' % len(self.videos_by_class['real']))
            print('The number of fake videos is : %d' % len(self.videos_by_class['fake']))
            self.num_smallest_class = len(self.videos_by_class[self.smallest_class])
            self.num_largest_class = len(self.videos_by_class[self.largest_class])

    def load_video_name(self):
        videos_by_class = {}

        real_videos = []
        fake_videos = []
        for video_name in tqdm(self.video_names):
            all_image_name = os.listdir(video_name)
            if len(all_image_name) < 1:
                continue

            if video_name.find('mp4') == -1:
                continue
            if video_name.find('Celeb-real') != -1:  # video is from Celeb-real
                real_videos.append(video_name)

            elif video_name.find('YouTube-real') != -1:  # video is from YouTube-real
                real_videos.append(video_name)

            else:  # video is fake
                fake_videos.append(video_name)

        videos_by_class['real'] = random.sample(real_videos, int(self.vid_sample * len(real_videos)))
        videos_by_class['fake'] = random.sample(fake_videos, int(self.vid_sample * len(fake_videos)))
        return videos_by_class

    def load_image_name(self):
        image_names = []
        for video_name in tqdm(self.videos_by_class['real'] + self.videos_by_class['fake']):
            video_path = os.path.join(self.root_path, video_name)
            random.seed(2021)
            frame_names = os.listdir(video_path)
            if len(frame_names) > self.test_frame_nums:
                frame_names = random.sample(frame_names, self.test_frame_nums)
            for image_name in frame_names:
                image_names.append(os.path.join(video_name, image_name))
        return image_names

    # def load_landmarks(self, landmarks_file):
    #     """
    #
    #     :param landmarks_file: input landmarks json file name
    #     :return: all_landmarks: having the shape of 64x2 list. represent left eye,
    #                             right eye, noise, left lip, right lip
    #     """
    #     all_landmarks = OrderedDict()
    #     with open(landmarks_file, "r", encoding="utf-8") as file:
    #         line = file.readline()
    #         while line:
    #             line = json.loads(line)
    #             all_landmarks[line["image_name"]] = np.array(line["landmarks"])
    #             line = file.readline()
    #     return all_landmarks

    def pil_loader(self, image_path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(image_path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def cv2_loader(self, image_path):
        img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        # img = Image.fromarray(img)
        return img

    def __getitem__(self, index):
        if self.phase == 'train':
            if index % 2 == 0:  # load smallest_class face image
                video_name = self.videos_by_class[self.smallest_class][min(index // 2, self.num_smallest_class)]
                video_path = os.path.join(self.root_path, video_name)
                # load a real video
                image_name = random.sample(os.listdir(video_path), 1)[0]
                image_path = os.path.join(video_path, image_name)
                label = 0  # real label is 0
            else:  # load largest_class face image
                video_index_from = math.ceil((index // 2) / self.num_smallest_class * self.num_largest_class)
                # Small epsilon to make sure whole numbers round down (so math.ceil != math.floor)
                video_index_to = math.floor(
                    ((index // 2) + 1) / self.num_smallest_class * self.num_largest_class - 0.0001)
                video_index_to = max(video_index_from, video_index_to)
                video_index_to = min(video_index_to, self.num_largest_class)
                video_index = random.randint(video_index_from, video_index_to)

                video_name = self.videos_by_class[self.largest_class][video_index]
                video_path = os.path.join(self.root_path, video_name)
                image_name = random.sample(os.listdir(video_path), 1)[0]
                image_path = os.path.join(video_path, image_name)
                label = 1  # fake label is 1

            if self.is_pillow:
                image = self.pil_loader(image_path)
                image = self.transform(image)
            else:
                image = self.cv2_loader(image_path)
                # image = self.transform(image=image)['image']
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                image = Image.fromarray(image)
                image = self.transform(image)

            # Face Embed Load
            face_emb = torch.load(os.path.join(video_path.replace('/data2/linkaiqing/code/Reprogramming/dataset',
                                                                  '/data2/linkaiqing/code/Reprogramming/dataset/Face_Embedding/Transface'),
                                               image_name.replace('.png', '.pt')), map_location='cpu')
            return image, label, face_emb, image_path

        else:
            image_path = os.path.join(self.root_path, self.image_name[index])
            if self.is_pillow:
                image = self.pil_loader(image_path)
                image = self.transform(image)
            else:
                image = self.cv2_loader(image_path)
                # image = self.transform(image=image)['image']
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                image = Image.fromarray(image)
                image = self.transform(image)

            if self.image_name[index].find('real') != -1:
                label = 0  # real label is 0
            else:
                label = 1  # fake label is 1

            # Face Embed Load
            face_emb = torch.load(((image_path.replace('/data2/linkaiqing/code/Reprogramming/dataset',
                                                       '/data2/linkaiqing/code/Reprogramming/dataset/Face_Embedding/Transface').replace(
                '.png', '.pt'))), map_location='cpu')
            return image, label, face_emb, image_path

    def __len__(self):
        if self.phase == 'train':  # load Rebalanced image data
            return self.num_smallest_class * 2
        else:  # load all image
            return len(self.image_name)

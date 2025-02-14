import os.path
from data.base_dataset import BaseDataset, get_transform, get_transform_both
from data.image_folder import make_dataset
from PIL import Image
import random
import util.util as util
import numpy as np

class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_mask_A = ''
        self.is_COI = opt.is_COI
        if opt.is_COI:
            self.dir_A = '/data/nas_archive-research_local/999_project/015_COI_rnd_tf/a102du-black/sides/seg_patch_cropped_binary-labeled_for_cls/image'
            self.dir_B = '/data/nas_archive-research_local/999_project/015_COI_rnd_tf/a102du-white/sides/seg_patch_cropped_binary-labeled_for_cls/image'
            self.dir_mask_A = '/data/nas_archive-research_local/999_project/015_COI_rnd_tf/a102du-black/sides/seg_patch_cropped_binary-labeled_for_cls/mask/defect.3class'
        else:
            self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
            self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        if opt.phase == "test":
            if opt.is_COI:
                self.dir_A = '/data/nas_archive-research_local/999_project/015_COI_rnd_tf/a102du-black/sides/seg_patch_cropped_binary-labeled_for_cls/image'
                self.dir_B = '/data/nas_archive-research_local/999_project/015_COI_rnd_tf/a102du-white/sides/seg_patch_cropped_binary-labeled_for_cls/image'
            elif not os.path.exists(self.dir_A) and os.path.exists(os.path.join(opt.dataroot, "valA")):
                self.dir_A = os.path.join(opt.dataroot, "valA")
                self.dir_B = os.path.join(opt.dataroot, "valB")

        image_listA = make_dataset(self.dir_A, opt.max_dataset_size, opt.is_COI,self.dir_mask_A)
        self.A_paths = sorted(image_listA)   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size, opt.is_COI))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        
        if self.is_COI:
            A_mask = Image.open(A_path[1]) if len(A_path) > 1 else None
            A_img = Image.open(A_path[0]).convert('RGB')
            B_img = Image.open(B_path[0]).convert('RGB')
        else:
            A_mask = None
            A_img = Image.open(A_path).convert('RGB')
            B_img = Image.open(B_path).convert('RGB')
        # Apply image transformation
        # For FastCUT mode, if in finetuning phase (learning rate is decaying),
        # do not perform resize-crop data augmentation of CycleGAN.
#        print('current_epoch', self.current_epoch)
        is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)
        transform = get_transform(modified_opt)

        if A_mask is not None:
            A, A_mask = get_transform_both(modified_opt, A_img, A_mask)
        else:
            A = transform(A_img) # 256,256,3 >> 3,256,256
            A_mask = []
        B = transform(B_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path, 'A_mask' : A_mask} 

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)

from torch.utils.data import Dataset
from deep_video_extraction.utils.utils import crawl_directory


class VideoDataset(Dataset):
    def __init__(self, video_dir, transform=None) -> None:
        super().__init__()
        self.video_dir = video_dir
        self.videos = crawl_directory(video_dir)
        self.transform = transform

    def __str__(self):
        print(f'Video DataLoader')

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        frame_path = self.videos[index]
        frame = np.array(Image.open(frame_path).convert('RGB'))

        if self.transform is not None:
            augmentations = self.transform(frame=frame)
            frame = augmentations['frame']
        return frame

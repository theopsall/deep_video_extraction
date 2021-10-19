from torch.utils.data import Dataset
from deep_video_extraction.utils.utils import crawl_directory


class VideoDataset(Dataset):
    def __init__(self, video_dir) -> None:
        super().__init__()
        self.video_dir = video_dir
        self.videos = crawl_directory(video_dir)

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        video = self.videos[index]
        return video

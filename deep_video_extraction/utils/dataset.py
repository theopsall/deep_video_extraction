from torch.utils.data import Dataset
from config import MEAN, STD
from utils.utils import analyze_video
from torchvision import transforms


class VideoDataset(Dataset):
    def __init__(self, video_path) -> None:
        super().__init__()
        self.video_path = video_path
        self.frames, self.fps = analyze_video(video_path)
        self.toTensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=MEAN, std=STD)
        print(self.frames[0].shape)

    def __str__(self):
        return f'Video DataLoader'

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        frame = self.frames[index]
        frame = self.normalize(self.toTensor(frame))

        return frame

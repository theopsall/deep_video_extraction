from torch.utils.data import Dataset
from utils.utils import analyze_video


class VideoDataset(Dataset):
    def __init__(self, video_path, transform=None) -> None:
        super().__init__()
        self.video_path = video_path
        self.frames = analyze_video(video_path)
        self.transform = transform

    def __str__(self):
        print(f'Video DataLoader')

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        frame = self.frames[index]

        if self.transform is not None:
            frame = self.transform(frame)

        return frame

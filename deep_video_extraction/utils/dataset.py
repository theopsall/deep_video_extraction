from config import MEAN, STD
from utils.utils import analyze_video, analyze_spectrograms
from torch.utils.data import Dataset
from torchvision import transforms


class VideoDataset(Dataset):
    def __init__(self, video_path) -> None:
        super().__init__()
        self.video_path = video_path
        self.frames = analyze_video(video_path)
        self.toTensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=MEAN, std=STD)

    def __str__(self):
        return f"Video DataLoader"

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        frame = self.frames[index]
        frame = self.normalize(self.toTensor(frame))

        return frame


class OpticalFlowDataset(Dataset):
    def __init__(self, video_path) -> None:
        super().__init__()
        self.video_path = video_path
        self.frames = analyze_video(video_path)
        self.toTensor = transforms.ToTensor()

    def __str__(self):
        return f"Video DataLoader"

    def __len__(self):
        return len(self.frames)-1

    def __getitem__(self, index):
        return self.toTensor(self.frames[index]),self.toTensor(self.frames[index+1])


class SpectrogramDataset(Dataset):
    def __init__(self, audio_path) -> None:
        super().__init__()
        self.audio_path = audio_path
        self.spectrograms = analyze_spectrograms(audio_path)
        self.toTensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=MEAN, std=STD)

    def __str__(self):
        return f"Spectrogram Dataset"

    def __len__(self):
        return len(self.spectrograms)

    def __getitem__(self, index):
        spectrogram = self.spectrograms[index]
        spectrogram = self.normalize(self.toTensor(spectrogram))

        return spectrogram
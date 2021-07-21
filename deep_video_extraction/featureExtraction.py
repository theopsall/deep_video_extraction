from deep_video_extraction.utils import utils
from deep_video_extraction.Extractors import visualExtractor as vsE
from torch.utils.data import DataLoader
import torch




def main():
    tree = utils.crawl_directory('videos')
    frames = utils.analyze_video(tree[0], 10)
    vs = vsE.VisualExtractor()
    for i in frames:
        print(i)
        print(i.shape)
        print(i.shape)
        new_u = torch.from_numpy(i)
        print(new_u.shape)
        predict = vs(new_u)
        print(predict.shape)

if __name__ == "__main__":
    main()

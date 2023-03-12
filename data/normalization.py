import torch


class Normalization():
    """This class is for normalizing the spectrograms batch by batch. The normalization used is min-max, two modes 'framewise' and 'imagewise' can be selected. In this paper, we found that 'imagewise' normalization works better than 'framewise'"""

    def __init__(self, mode='framewise'):
        if mode == 'framewise':
            def normalize(x):
                size = x.shape
                # Finding max values for each frame
                x_max = x.max(1, keepdim=True)[0]
                x_min = x.min(1, keepdim=True)[0]
                # If there is a column with all zero, nan will occur
                output = (x-x_min)/(x_max-x_min)
                output[torch.isnan(output)] = 0  # Making nan to 0
                return output
        elif mode == 'imagewise':
            def normalize(x):
                size = x.shape
                x_max = x.view(size[0], size[1]*size[2]
                               ).max(1, keepdim=True)[0]
                x_min = x.view(size[0], size[1]*size[2]
                               ).min(1, keepdim=True)[0]
                x_max = x_max.unsqueeze(1)  # Make it broadcastable
                x_min = x_min.unsqueeze(1)  # Make it broadcastable
                output = (x-x_min)/(x_max-x_min)
                output[torch.isnan(output)] = 0
                return output
        else:
            print(f'please choose the correct mode')
        self.normalize = normalize

    def transform(self, x):
        return self.normalize(x)

import torch.multiprocessing as torch_mp
import multiprocessing as mp
import torch
import numpy as np

torch_mp.set_sharing_strategy('file_system')


def make_data(ind):
    return torch.rand(5, 1) * ind


def main():

    input_data = np.arange(100_000)
    input_data = torch.tensor(input_data)

    with mp.Pool(4) as pool:
        result = pool.map(make_data, input_data)

    print(len(result))
    print(result[0])
    print(result[1234])


if __name__ == "__main__":
    main()

import torch
import torch.multiprocessing as torch_mp
import multiprocessing as mp

torch_mp.set_sharing_strategy("file_system")  # Without this -> RuntimeError: received 0 items of ancdata

# torch_mp.set_start_method("spawn", force=True) # Does nothing
# mp.set_start_method("spawn", force=True)


def basic_worker(data):
    return [data, data + 1]


def basic():
    """Input list, return list, use multiprocessing."""
    input_data = range(100_000_000)
    # If I increase the range to 100_000_000, the memory usage grows
    # continously, but swap space handles it and it finishes normally.

    with mp.Pool() as pool:
        result = pool.map(basic_worker, input_data)

    print(result[1234])


def tensor_worker(data):
    return torch.tensor([data, data + 1])


def ilist_rtensor_mp():
    """Input list, return tensor, use multiprocessing."""
    input_data = range(100_000)

    with mp.Pool() as pool:
        result = pool.map(tensor_worker, input_data)
    # RuntimeError: unable to mmap 80 bytes from file </torch_48989_3226785460_2544>: Cannot allocate memory (12)

    print(result[1234])


def itensor_rtensor_mp():
    """Input tensor, return tensor, use multiprocessing."""
    input_data = torch.arange(0, 100_000)

    with mp.Pool() as pool:
        result = pool.map(tensor_worker, input_data)
    # RuntimeError: unable to mmap 80 bytes from file </torch_52183_2330234596_4043>: Cannot allocate memory (12)

    print(result[1234])


def ilist_rtensor_torchmp():
    """Input list, return tensor, use torch.multiprocessing."""
    input_data = range(100_000)

    with torch_mp.Pool() as pool:
        result = pool.map(tensor_worker, input_data)
    # RuntimeError: unable to mmap 80 bytes from file </torch_54397_1627901275_4099>: Cannot allocate memory (12)

    print(result[1234])


def itensor_rtensor_torchmp():
    """Input tensor, return tensor, use torch.multiprocessing."""
    input_data = torch.arange(0, 100_000)

    with torch_mp.Pool() as pool:
        result = pool.map(tensor_worker, input_data)
    # RuntimeError: unable to mmap 80 bytes from file </torch_54943_2537684984_4044>: Cannot allocate memory (12)

    print(result[1234])


if __name__ == "__main__":
    itensor_rtensor_mp()

# Full error
# ----------
# Exception in thread Thread-3 (_handle_results):
# Traceback (most recent call last):
#   File "/usr/lib/python3.10/threading.py", line 1016, in _bootstrap_inner
#     self.run()
#   File "/usr/lib/python3.10/threading.py", line 953, in run
#     self._target(*self._args, **self._kwargs)
#   File "/usr/lib/python3.10/multiprocessing/pool.py", line 579, in _handle_results
#     task = get()
#   File "/usr/lib/python3.10/multiprocessing/connection.py", line 251, in recv
#     return _ForkingPickler.loads(buf.getbuffer())
#   File "/home/.../lib/python3.10/site-packages/torch/multiprocessing/reductions.py", line 514, in rebuild_storage_filename
#     storage = torch.UntypedStorage._new_shared_filename_cpu(manager, handle, size)
# RuntimeError: unable to mmap 80 bytes from file </torch_54943_2537684984_4044>: Cannot allocate memory (12)

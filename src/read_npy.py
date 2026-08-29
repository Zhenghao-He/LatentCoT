import argparse
from pathlib import Path

import numpy as np


# DEFAULT_DIR = Path(
#     "/p/realai/zhenghao/Hint_And_Reason/results/meta-llama/"
#     "Meta-Llama-3.1-8B-Instruct/features/gsm8k/layers.19"
# ) # array([26886., 30461., 63569.,  8629., 56208., 13288.,  4200., 25950., 30612., 21725.])

# array([26886., 30461., 63569.,  8629., 56208., 13288.,  4200., 25950.,
#        30612., 21725., 28011., 11646., 35967., 49408., 12888., 20732.,
#        51375., 45757.,  6128., 33814., 61758., 47464., 55652., 34754.,
#        17101.])

# DEFAULT_DIR = Path(
#     "/p/realai/zhenghao/Hint_And_Reason/results/meta-llama/"
#     "Meta-Llama-3.1-8B-Instruct/features/gpqa/layers.19"
# ) # array([41105., 59058., 12158.,  9340., 13288.,  9688., 21797., 16918., 32077., 50305.])
# array([41105., 59058., 12158.,  9340., 13288.,  9688., 21797., 16918.,
#        32077., 50305., 38767., 30461.,  3710., 17101., 10982., 18511.,
#        25815., 22271., 25950., 35277., 64760., 40570.,  6128.,  7333.,
#        41863.])


DEFAULT_DIR = Path(
    "/p/realai/zhenghao/Hint_And_Reason/results/meta-llama/"
    "Meta-Llama-3.1-8B-Instruct/features/bbh/layers.19"
)# array([12158.,  9340., 21797., 32077.,  3710., 17101., 22271., 30461., 25815., 40570.])

# array([12158.,  9340., 21797., 32077.,  3710., 17101., 22271., 30461.,
#        25815., 40570., 32694., 14170., 52264., 56208., 26629., 51817.,
#        26489., 42729.,  6128., 60680., 57030., 31550., 40797., 13011.,
#         8493.])
def summarize_array(array: np.ndarray, preview: int) -> None:
    print(f"  shape: {array.shape}")
    print(f"  dtype: {array.dtype}")

    if array.size == 0:
        print("  array is empty")
        return

    if np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_):
        print(f"  min: {array.min()}")
        print(f"  max: {array.max()}")
        print(f"  mean: {array.mean()}")

    flat = array.reshape(-1)
    shown = flat[:preview]
    print(f"  first {len(shown)} values: {shown}")


def load_one_file(file_path: Path, preview: int, allow_pickle: bool) -> None:
    print(f"\n[file] {file_path}")
    array = np.load(file_path, allow_pickle=allow_pickle)
    import pdb; pdb.set_trace()
    summarize_array(array, preview)


def main() -> None:
    parser = argparse.ArgumentParser(description="Read .npy files and print basic info.")
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_DIR,
        help="A .npy file or a directory containing .npy files.",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=10,
        help="How many flattened values to print for each array.",
    )
    parser.add_argument(
        "--allow-pickle",
        action="store_true",
        help="Enable loading pickled object arrays if needed.",
    )
    args = parser.parse_args()

    target = args.path
    if not target.exists():
        raise FileNotFoundError(f"Path does not exist: {target}")

    if target.is_file():
        if target.suffix != ".npy":
            raise ValueError(f"Expected a .npy file, got: {target}")
        load_one_file(target, args.preview, args.allow_pickle)
        return

    npy_files = sorted(target.glob("*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"No .npy files found in directory: {target}")

    print(f"Found {len(npy_files)} .npy files in {target}")
    for file_path in npy_files:
        load_one_file(file_path, args.preview, args.allow_pickle)


if __name__ == "__main__":
    main()

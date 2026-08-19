import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src" / "threecad_segmentation"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from segformer_b0 import SegFormerB0Binary
from train_common import common_parser, config_from_args, run_train


def main():
    parser = common_parser(default_batch_size=2, default_grad_accum=2)
    args = parser.parse_args()
    cfg = config_from_args(args)
    run_train(SegFormerB0Binary(pretrained=True), cfg, "segformer_b0")


if __name__ == "__main__":
    main()

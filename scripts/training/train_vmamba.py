import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src" / "threecad_segmentation"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from vmamba_t import build_vmamba_t_binary
from train_common import common_parser, config_from_args, run_train


def main():
    # Batch 1 + accumulation 4 keeps the effective batch size at 4 while
    # remaining safe on the user's tested T4 setup.
    parser = common_parser(default_batch_size=1, default_grad_accum=4)
    args = parser.parse_args()
    cfg = config_from_args(args)
    model = build_vmamba_t_binary(pretrained=True)
    run_train(model, cfg, "vmamba_t_s2l5")


if __name__ == "__main__":
    main()

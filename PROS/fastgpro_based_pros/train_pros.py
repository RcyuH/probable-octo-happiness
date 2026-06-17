"""CLI entry point for the FastGRPO-based PROS trainer."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    package_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(package_root))
    from fastgpro_based_pros.pros_config import ProsConfig
else:
    from .pros_config import ProsConfig


def main(argv: list[str] | None = None) -> None:
    cfg = ProsConfig.from_args(argv)
    if cfg.dry_run:
        print(cfg.to_dict())
        return
    if __package__ is None or __package__ == "":
        from fastgpro_based_pros.pros_trainer import ProsTrainer
    else:
        from .pros_trainer import ProsTrainer

    trainer = ProsTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()

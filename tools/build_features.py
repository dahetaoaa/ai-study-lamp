#!/usr/bin/env python3
"""为已有模板JSON中的每一页构建AKAZE特征；用 backend 的 uv 环境运行。"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.vision.matcher import TemplateLibrary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--templates", type=Path, default=ROOT / "data/templates")
    parser.add_argument("--fixture", type=Path)
    args = parser.parse_args()
    library = TemplateLibrary(args.templates, args.fixture)
    library.load()
    print(f"特征已就绪：{len(library.papers)}份试卷，{len(library.pages)}页")


if __name__ == "__main__":
    main()

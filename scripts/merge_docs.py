#!/usr/bin/env python3
"""
将 docs 目录下的所有 markdown 文件合并成一个文件
"""

import os
from pathlib import Path

def merge_docs():
    docs_dir = Path(__file__).parent.parent / "docs"
    output_file = Path(__file__).parent.parent / "all_docs_merged.md"

    # 收集所有 md/mdx 文件
    md_files = []
    for ext in ["*.md", "*.mdx"]:
        md_files.extend(docs_dir.rglob(ext))

    # 按路径排序
    md_files.sort()

    print(f"找到 {len(md_files)} 个文档")

    with open(output_file, "w", encoding="utf-8") as out:
        out.write("# Yiwen Blog 文档合集\n\n")
        out.write(f"共 {len(md_files)} 篇文档\n\n")
        out.write("---\n\n")

        for i, md_file in enumerate(md_files, 1):
            relative_path = md_file.relative_to(docs_dir)
            print(f"[{i}/{len(md_files)}] 处理: {relative_path}")

            try:
                content = md_file.read_text(encoding="utf-8")

                # 写入分隔符和文件路径
                out.write(f"## 📄 {relative_path}\n\n")
                out.write(content)
                out.write("\n\n---\n\n")

            except Exception as e:
                print(f"  ⚠️ 读取失败: {e}")

    # 统计文件大小
    size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"\n✅ 完成! 输出文件: {output_file}")
    print(f"   文件大小: {size_mb:.2f} MB")

if __name__ == "__main__":
    merge_docs()

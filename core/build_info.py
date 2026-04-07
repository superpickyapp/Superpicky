# -*- coding: utf-8 -*-
"""
构建信息
此文件在发布构建时由 build_release.sh 自动修改，以注入 git commit hash
"""

# 在打包时会被替换为实际的 commit hash
COMMIT_HASH = "be2f41a3"

# 发布渠道：CI 打包时自动注入（"nightly" = RC 预发布，"official" = 正式版）
# 本地开发默认 "dev"，不触发更新检查
RELEASE_CHANNEL = "dev"




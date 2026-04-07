#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SuperPicky - 更新检测器
检查 GitHub Releases 获取最新版本，支持 Mac/Windows 分平台下载
"""

import sys
import platform
import urllib.request
import json
import re
from typing import Optional, Tuple, Dict
from packaging import version


# 当前版本号（从 constants.py 统一获取）
from constants import APP_VERSION
CURRENT_VERSION = APP_VERSION

# GitHub API 配置
GITHUB_REPO = "jamesphotography/SuperPicky"
GITHUB_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
GITHUB_RELEASES_URL = f"https://github.com/{GITHUB_REPO}/releases/latest"

# 平台+架构对应的 Asset 文件名模式
# 三层匹配策略：精确架构 > 通用版本 > 任意版本
PLATFORM_ARCH_PATTERNS = {
    'darwin': {
        'arm64': ['_arm64', '-arm64', '_apple_silicon', '-apple_silicon', '_m1', '-m1', '_m2', '-m2'],
        'x86_64': ['_x64', '-x64', '_x86_64', '-x86_64', '_intel', '-intel'],
        'universal': ['_universal', '-universal', '_mac', '-mac', 'macos', '.dmg']
    },
    'win32': {
        'AMD64': ['_x64', '-x64', '_win64', '-win64'],
        'x86': ['_x86', '-x86', '_win32', '-win32'],
        'universal': ['_win', '-win', 'windows', '.exe', '.msi', '-setup']
    }
}


class UpdateChecker:
    """更新检测器"""
    
    def __init__(self, current_version: str = CURRENT_VERSION):
        self.current_version = current_version
        self._latest_info: Optional[Dict] = None
    
    def check_for_updates(self, timeout: int = 10) -> Tuple[bool, Optional[Dict]]:
        """
        检查是否有更新
        
        Args:
            timeout: 请求超时时间（秒）
            
        Returns:
            (has_update, update_info) - update_info 包含:
                - version: 最新版本号
                - current_version: 当前版本号
                - download_url: 当前平台的下载链接
                - release_notes: 发布说明
                - release_url: GitHub Release 页面链接
        """
        try:
            # macOS SSL证书问题修复
            import ssl
            import urllib.request
            
            # 创建自定义的SSL上下文，禁用证书验证
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            # 请求 GitHub API
            req = urllib.request.Request(
                GITHUB_API_URL,
                headers={
                    'Accept': 'application/vnd.github.v3+json',
                    'User-Agent': f'SuperPicky/{self.current_version}'
                }
            )
            
            with urllib.request.urlopen(req, timeout=timeout, context=ssl_context) as response:
                data = json.loads(response.read().decode('utf-8'))
            
            self._latest_info = data
            
            # 解析版本号
            latest_version = data.get('tag_name', '').lstrip('vV')
            if not latest_version:
                # 无法获取版本，返回当前版本信息
                return False, {
                    'version': self.current_version,
                    'current_version': self.current_version,
                    'download_url': None,
                    'release_notes': '',
                    'release_url': GITHUB_RELEASES_URL,
                }
            
            # 比较版本
            try:
                has_update = version.parse(latest_version) > version.parse(self.current_version)
            except Exception:
                # 简单字符串比较作为回退
                has_update = latest_version != self.current_version
            
            # 获取当前平台的下载链接
            download_url = self._find_platform_download(data.get('assets', []))
            
            # 始终返回版本信息
            update_info = {
                'version': latest_version,
                'current_version': self.current_version,
                'download_url': download_url,
                'release_notes': data.get('body', ''),
                'release_url': data.get('html_url', GITHUB_RELEASES_URL),
                'published_at': data.get('published_at', ''),
            }
            
            return has_update, update_info
            
        except urllib.error.URLError as e:
            print(f"⚠️ 检查更新失败 (网络错误): {e}")
            return False, {'version': '检查失败', 'current_version': self.current_version, 'error': str(e)}
        except json.JSONDecodeError as e:
            print(f"⚠️ 检查更新失败 (解析错误): {e}")
            return False, {'version': '检查失败', 'current_version': self.current_version, 'error': str(e)}
        except Exception as e:
            print(f"⚠️ 检查更新失败: {e}")
            return False, {'version': '检查失败', 'current_version': self.current_version, 'error': str(e)}
    
    def _find_platform_download(self, assets: list) -> Optional[str]:
        """
        根据当前平台和架构查找对应的下载链接

        三层匹配策略：
        1. 精确架构匹配 - 优先查找 arm64/intel/x64 精确匹配
        2. 通用版本回退 - 查找 universal 版本
        3. 任意版本兜底 - 返回第一个平台相关的 DMG/EXE

        Args:
            assets: GitHub Release 的 assets 列表

        Returns:
            下载链接或 None
        """
        if not assets:
            return None

        # 确定当前平台和架构
        platform_key = 'darwin' if sys.platform == 'darwin' else 'win32'
        machine = platform.machine()  # arm64, x86_64, AMD64, x86 等

        arch_patterns = PLATFORM_ARCH_PATTERNS.get(platform_key, {})
        if not arch_patterns:
            return None

        # 获取当前架构的精确匹配模式
        exact_patterns = arch_patterns.get(machine, [])
        universal_patterns = arch_patterns.get('universal', [])

        # 第一层：精确架构匹配
        for asset in assets:
            name = asset.get('name', '').lower()
            download_url = asset.get('browser_download_url', '')

            for pattern in exact_patterns:
                if pattern.lower() in name:
                    return download_url

        # 第二层：通用版本回退
        for asset in assets:
            name = asset.get('name', '').lower()
            download_url = asset.get('browser_download_url', '')

            for pattern in universal_patterns:
                if pattern.lower() in name:
                    return download_url

        # 第三层：任意平台相关版本兜底
        # macOS: 返回第一个 .dmg 文件
        # Windows: 返回第一个 .exe 或 .msi 文件
        fallback_extensions = ['.dmg'] if platform_key == 'darwin' else ['.exe', '.msi']
        for asset in assets:
            name = asset.get('name', '').lower()
            download_url = asset.get('browser_download_url', '')

            for ext in fallback_extensions:
                if name.endswith(ext):
                    return download_url

        return None
    
    @staticmethod
    def get_platform_name() -> str:
        """获取当前平台名称（用于UI显示，包含架构信息）"""
        machine = platform.machine()

        if sys.platform == 'darwin':
            if machine == 'arm64':
                return 'macOS (Apple Silicon)'
            else:
                return 'macOS (Intel)'
        elif sys.platform.startswith('win'):
            if machine == 'AMD64':
                return 'Windows (64-bit)'
            else:
                return 'Windows (32-bit)'
        else:
            return f'Linux ({machine})'

    @staticmethod
    def get_platform_short_name() -> str:
        """获取平台简短标识（用于文件命名匹配）"""
        machine = platform.machine()

        if sys.platform == 'darwin':
            if machine == 'arm64':
                return 'mac_arm64'
            else:
                return 'mac_intel'
        elif sys.platform.startswith('win'):
            if machine == 'AMD64':
                return 'win64'
            else:
                return 'win32'
        else:
            return f'linux_{machine}'


def check_update_async(callback, current_version: str = CURRENT_VERSION):
    """
    异步检查更新（在后台线程执行）
    
    Args:
        callback: 回调函数，签名 callback(has_update: bool, update_info: Optional[Dict])
        current_version: 当前版本号
    """
    import threading
    
    def _check():
        checker = UpdateChecker(current_version)
        has_update, update_info = checker.check_for_updates()
        callback(has_update, update_info)
    
    thread = threading.Thread(target=_check, daemon=True)
    thread.start()


# 测试代码
if __name__ == "__main__":
    print("=== SuperPicky 更新检测器测试 ===\n")
    print(f"当前版本: {CURRENT_VERSION}")
    print(f"当前平台: {UpdateChecker.get_platform_name()}")
    print(f"平台标识: {UpdateChecker.get_platform_short_name()}")
    print(f"CPU 架构: {platform.machine()}\n")

    checker = UpdateChecker()
    has_update, info = checker.check_for_updates()

    if has_update:
        print(f"✅ 发现新版本: {info['version']}")
        print(f"📦 下载链接: {info['download_url']}")
        print(f"🔗 Release 页面: {info['release_url']}")
        print(f"\n📝 发布说明:\n{info['release_notes'][:500]}...")
    else:
        print("✅ 已是最新版本")

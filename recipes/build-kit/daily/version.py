from typing import Optional

__all__ = ['__version__', 'debug', 'cuda', 'git_version', 'hip', 'rocm', 'xpu']
__version__ = '2.13.0a0+gfx1013'
debug = False
cuda: Optional[str] = None
git_version = 'Unknown'
hip: Optional[str] = '7.17.0'
rocm: Optional[str] = '10.2.0'
xpu: Optional[str] = None

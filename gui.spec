# -*- mode: python -*-
a = Analysis(['gui.py'],
             pathex=['C:\\Users\\Lalit\\Documents\\CODE\\Python\\glee-py-gui'],
             hiddenimports=[],
             hookspath=None,
             runtime_hooks=None)
pyz = PYZ(a.pure)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          name='gui.exe',
          debug=False,
          strip=None,
          upx=True,
          console=False )

import re, glob
for f in glob.glob('/app/flash-attention/setup.py') + glob.glob('/app/flash-attention/csrc/flash_attn*/**/*.py', recursive=True):
    txt = open(f).read()
    txt2 = re.sub(r'(-Wno-(?:unused-result|vla-cxx-extension))', r'\1 -Wno-deprecated-attributes', txt)
    if txt2 != txt:
        open(f, 'w').write(txt2)
        print(f'[fa-patch] added -Wno-deprecated-attributes to {f}')

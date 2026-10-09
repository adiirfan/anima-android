"""argb2png.py <in.argb> <W> <H> <out.png> [view width]: phone ARGB dump -> PNG"""
import sys
import numpy as np
from PIL import Image
w, h = int(sys.argv[2]), int(sys.argv[3])
a = np.fromfile(sys.argv[1], dtype=np.uint32).reshape(h, w)
im = Image.fromarray(np.stack([(a >> 16) & 255, (a >> 8) & 255, a & 255], -1).astype(np.uint8))
if len(sys.argv) > 5:
    vw = int(sys.argv[5]); im = im.resize((vw, vw * h // w))
im.save(sys.argv[4])

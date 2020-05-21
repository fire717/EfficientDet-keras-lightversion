from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

compute_overlap_lines = [
"cimport cython",
"import numpy as np",
"cimport numpy as np",
"def compute_overlap(np.ndarray[double, ndim=2] boxes,np.ndarray[double, ndim=2] query_boxes):",
"    cdef unsigned int N = boxes.shape[0]",
"    cdef unsigned int K = query_boxes.shape[0]",
"    cdef np.ndarray[double, ndim=2] overlaps = np.zeros((N, K), dtype=np.float64)",
"    cdef double iw, ih, box_area",
"    cdef double ua",
"    cdef unsigned int k, n",
"    for k in range(K):",
"        box_area = ((query_boxes[k, 2] - query_boxes[k, 0] + 1) *(query_boxes[k, 3] - query_boxes[k, 1] + 1))",
"        for n in range(N):",
"            iw = (min(boxes[n, 2], query_boxes[k, 2]) -max(boxes[n, 0], query_boxes[k, 0]) + 1)",
"            if iw > 0:",
"                ih = (min(boxes[n, 3], query_boxes[k, 3]) -max(boxes[n, 1], query_boxes[k, 1]) + 1)",
"                if ih > 0:",
"                    ua = np.float64((boxes[n, 2] - boxes[n, 0] + 1) *(boxes[n, 3] - boxes[n, 1] + 1) +box_area - iw * ih)",
"                    overlaps[n, k] = iw * ih / ua",
"    return overlaps",
]
with open("compute_overlap.pyx", "w") as f:
    for line in compute_overlap_lines:
        f.write(line+"\n")

# setup(
#     ext_modules=cythonize("compute_overlap.pyx"),
#     include_dirs=[np.get_include()]
# )
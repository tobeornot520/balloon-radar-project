
def crop_roi(rd, center, size=16):
    r,v=center
    return rd[..., max(0,r-size//2):r+size//2,
              max(0,v-size//2):v+size//2]

def build_roi_feature(power_roi, polar_roi=None):
    return power_roi if polar_roi is None else polar_roi


from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment



class FPDistCalc:
    def __init__(self, fpcal):
        self.fpcalc = fpcal

    def compare(self, a, b):
        fps_a = self.fpcalc(a)
        fps_b = self.fpcalc(b)
        return self.compare_fps(fps_a, fps_b)

    def compare_fps(self, fps_a, fps_b):
        C = distance_matrix(fps_a, fps_b)
        r_ind, c_ind = linear_sum_assignment(C)
        fp_dist = C[r_ind, c_ind].sum()
        return fp_dist


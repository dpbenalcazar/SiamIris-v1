import math
import numpy as np

class biometrics(object):
    def __init__(self, mated, non_m):
        self.mated = np.array(mated)
        self.non_m = np.array(non_m)
        self.Nmated = len(mated)
        self.Nnon_m = len(non_m)
        self.gr = (math.sqrt(5) + 1) / 2

    def d_prime(self):
        avg_mated = np.mean(self.mated)
        std_mated = np.std( self.mated)
        avg_non_m = np.mean(self.non_m)
        std_non_m = np.std( self.non_m)
        return np.abs(avg_mated - avg_non_m)/np.sqrt(0.5*(std_mated**2 + std_non_m**2))

    def FAR(self, th):
        return 100*np.sum(self.non_m<th)/self.Nnon_m

    def FARb(self, th):
        return abs(100*np.sum(self.non_m<th)/self.Nnon_m - self.FARv)

    def FRR(self, th):
        return 100*np.sum(self.mated>th)/self.Nmated

    def err(self, th):
        return abs(self.FAR(th) - self.FRR(th))

    def golden_search(self, f, tol=1e-5):
        a = min(self.mated)
        b = max(self.non_m)
        c = b - (b - a) / self.gr
        d = a + (b - a) / self.gr

        while abs(b - a) > tol:
            if f(c) < f(d):
                b = d
            else:
                a = c

            c = b - (b - a) / self.gr
            d = a + (b - a) / self.gr

        return (b + a) / 2

    def get_EER(self):
        EER_th = self.golden_search(f=self.err)
        EER = self.FAR(EER_th)
        return EER, EER_th

    def get_FRR_at(self, FAR_val=0.1):
        self.FARv = FAR_val
        th = self.golden_search(f=self.FARb)
        return self.FAR(th), self.FRR(th), th

    def for_plots(self, d_th = 0.001):
        all_th = np.arange(min(self.mated),max(self.non_m), d_th)
        all_FAR = np.array([self.FAR(th) for th in all_th])
        all_FRR = np.array([self.FRR(th) for th in all_th])
        return all_FAR, all_FRR, all_th

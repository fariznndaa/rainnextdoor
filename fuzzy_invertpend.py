import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# 1. MODEL PLANT NON-LINIER (CONICAL TANK)
# =====================================================================
class ConicalTank:
    def __init__(self, R=1.5, H=4.0, Cv=0.35):
        self.R = R      # Radius atas tangki (m)
        self.H = H      # Tinggi total tangki (m)
        self.Cv = Cv    # Koefisien valve keluar

    def step(self, h, Qin, dt):
        """Level berikutnya dengan integrasi Euler"""
        h_safe = max(h, 0.01)                              # cegah pembagian nol
        A_h = np.pi * ((self.R / self.H) * h_safe) ** 2    # luas permukaan A(h)
        Qout = self.Cv * np.sqrt(h_safe)                   # Torricelli
        dhdt = (Qin - Qout) / A_h
        return max(h + dhdt * dt, 0.001)


# =====================================================================
# 2. FUZZY LOGIC CONTROLLER (INCREMENTAL / FUZZY-PI, SUGENO ORDER-0)
# =====================================================================
class FuzzyPIController:
    def __init__(self, ez=0.4):
        self.ez = ez    # setengah lebar MF e_Z: segitiga [-ez, 0, ez]

    @staticmethod
    def trimf(x, a, b, c):
        """Fungsi keanggotaan segitiga"""
        if x <= a or x >= c:
            return 0.0
        elif x <= b:
            return (x - a) / (b - a + 1e-9)
        else:
            return (c - x) / (c - b + 1e-9)

    def compute_delta_u(self, e, de):
        t = self.trimf

        # --- Fuzzifikasi error ---
        e_NB = t(e, -4.0, -2.0, -0.8)
        e_N  = t(e, -1.8, -0.8,  0.0)
        e_Z  = t(e, -self.ez, 0.0, self.ez)
        e_P  = t(e,  0.0,  0.8,  1.8)
        e_PB = t(e,  0.8,  2.0,  4.0)

        # --- Fuzzifikasi delta error ---
        de_N = t(de, -2.0, -0.5, 0.0)
        de_Z = t(de, -0.3,  0.0, 0.3)
        de_P = t(de,  0.0,  0.5, 2.0)

        # --- Rule base (bobot, singleton output) ---
        rules = [
            (min(e_PB, de_P),  0.15), (min(e_PB, de_Z),  0.10), (min(e_PB, de_N),  0.03),
            (min(e_P,  de_P),  0.08), (min(e_P,  de_Z),  0.04), (min(e_P,  de_N),  0.00),
            (min(e_Z,  de_P),  0.02), (min(e_Z,  de_Z),  0.00), (min(e_Z,  de_N), -0.02),
            (min(e_N,  de_P),  0.00), (min(e_N,  de_Z), -0.04), (min(e_N,  de_N), -0.08),
            (min(e_NB, de_P), -0.03), (min(e_NB, de_Z), -0.10), (min(e_NB, de_N), -0.15),
        ]

        # --- Defuzzifikasi: weighted average ---
        num = sum(w * du for w, du in rules)
        den = sum(w for w, du in rules)
        if den == 0:
            if e > 0.5:  return 0.08
            if e < -0.5: return -0.08
            return 0.0
        return num / den


# =====================================================================
# 3. LOOP SIMULASI
# =====================================================================
def run_simulation(Cv=0.35, ez=0.4, dt=0.1, t_end=240.0, Ku=4.0):
    steps = int(t_end / dt)
    tank = ConicalTank(Cv=Cv)
    flc = FuzzyPIController(ez=ez)

    h, u, prev_e = 0.5, 0.5, 0.0
    T, H, SP, U = [], [], [], []

    for i in range(steps):
        t = i * dt
        sp = 2.5 if t < 120.0 else 1.0

        e = sp - h
        de = (e - prev_e) / dt
        prev_e = e

        du = flc.compute_delta_u(e, de)
        u = np.clip(u + du * Ku * dt, 0.0, 3.0)   # akumulasi + saturasi pompa
        h = tank.step(h, u, dt)

        T.append(t); H.append(h); SP.append(sp); U.append(u)

    return np.array(T), np.array(H), np.array(SP), np.array(U)


# =====================================================================
# 4. PERHITUNGAN PARAMETER RESPONS TRANSIEN
# =====================================================================
def hitung_metrik(T, H, t0, t1, h0, sp):
    """
    tr  : rise time 10%-90% dari besar perubahan setpoint
    ts  : settling time pita 2% dari nilai SP
    OS  : overshoot (%) terhadap besar perubahan setpoint
    ess : steady state error (rata-rata 10 detik terakhir)
    """
    m = (T >= t0) & (T < t1)
    t = T[m] - t0
    h = H[m]
    d = sp - h0                       # besar langkah (negatif jika turun)
    frac = (h - h0) / d

    tr = t[np.argmax(frac >= 0.9)] - t[np.argmax(frac >= 0.1)]

    di_luar = np.where(np.abs(h - sp) > 0.02 * abs(sp))[0]
    if len(di_luar) == 0:
        ts = 0.0
    elif di_luar[-1] + 1 < len(t):
        ts = t[di_luar[-1] + 1]
    else:
        ts = t[-1]

    if d > 0:
        os_ = max(0.0, (h.max() - sp) / abs(d) * 100)
    else:
        os_ = max(0.0, (sp - h.min()) / abs(d) * 100)   # undershoot saat turun

    ess = sp - np.mean(h[-100:])
    return tr, ts, os_, ess


def laporkan(nama, T, H):
    h120 = H[int(120 / 0.1) - 1]      # level saat setpoint berganti
    r1 = hitung_metrik(T, H, 0, 120, 0.5, 2.5)
    r2 = hitung_metrik(T, H, 120, 240, h120, 1.0)
    print(f"\n=== {nama} ===")
    print(f"{'Parameter':<22}{'Region 1 (2.5 m)':>20}{'Region 2 (1.0 m)':>20}")
    print(f"{'Rise time (s)':<22}{r1[0]:>20.2f}{r2[0]:>20.2f}")
    print(f"{'Settling time 2% (s)':<22}{r1[1]:>20.2f}{r2[1]:>20.2f}")
    print(f"{'Overshoot (%)':<22}{r1[2]:>20.2f}{r2[2]:>20.2f}")
    print(f"{'Steady state err (m)':<22}{r1[3]:>20.4f}{r2[3]:>20.4f}")


# =====================================================================
# 5. PLOT
# =====================================================================
def plot_hasil(judul, T, H, SP, U):
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(T, SP, 'r--', label='Setpoint (m)', linewidth=2)
    plt.plot(T, H, 'b-', label='Level Tangki h(t) [FLC]', linewidth=2)
    plt.title(judul, fontsize=14)
    plt.ylabel('Ketinggian Air (meter)', fontsize=12)
    plt.grid(True)
    plt.legend(loc='upper right')

    plt.subplot(2, 1, 2)
    plt.plot(T, U, 'g-', label='Debit Pompa Qin u(t)', linewidth=1.5)
    plt.xlabel('Waktu (detik)', fontsize=12)
    plt.ylabel('Laju Alir u(t) (m³/s)', fontsize=12)
    plt.grid(True)
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()


# =====================================================================
# 6. MAIN: SKENARIO 1, 2, 3
# =====================================================================
if __name__ == "__main__":
    skenario = [
        ("Skenario 1: Baseline (e_Z = ±0.4, Cv = 0.35)",   dict(Cv=0.35, ez=0.4)),
        ("Skenario 2: e_Z dipersempit (±0.1, Cv = 0.35)",  dict(Cv=0.35, ez=0.1)),
        ("Skenario 3: Cv = 0.6 (e_Z = ±0.4)",              dict(Cv=0.60, ez=0.4)),
    ]

    for judul, param in skenario:
        T, H, SP, U = run_simulation(**param)
        laporkan(judul, T, H)
        plot_hasil("Simulasi FLC pada Conical Tank - " + judul, T, H, SP, U)
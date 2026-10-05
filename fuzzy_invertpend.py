"""
Praktikum FLC - Inverted Pendulum (Sugeno orde nol, plant nonlinier, RK4)

Cara pakai:
    python flc_inverted_pendulum.py            # jalankan semua eksperimen A-E + grafik
    python flc_inverted_pendulum.py nominal    # hanya respons nominal (Eksperimen B)
    python flc_inverted_pendulum.py A          # Eksperimen A saja
    python flc_inverted_pendulum.py C          # Eksperimen C saja
    python flc_inverted_pendulum.py D          # Eksperimen D saja
    python flc_inverted_pendulum.py E          # Eksperimen E saja
    python flc_inverted_pendulum.py mf         # grafik fungsi keanggotaan (mf.png)
    python flc_inverted_pendulum.py dt         # cek pengaruh DT
    python flc_inverted_pendulum.py surface    # peta permukaan FLC
    python flc_inverted_pendulum.py opsional   # noise sensor, dead-zone, AND min vs product

Semua grafik (PNG) dan tabel (CSV) disimpan di folder ./hasil
"""
import os
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")  # simpan ke berkas; hapus baris ini jika ingin plt.show()
import matplotlib.pyplot as plt

OUT_DIR = "hasil"
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# 1. Parameter plant dan simulasi
# theta = 0 rad berarti pendulum tegak ke atas
# ============================================================
P = {
    "M": 0.70,      # massa kereta (kg)
    "m": 0.20,      # massa pendulum (kg)
    "l": 0.30,      # jarak pivot ke pusat massa (m)
    "b": 0.08,      # gesekan viskos kereta (N.s/m)
    "g": 9.81,      # gravitasi (m/s^2)
    "Fmax": 15.0,   # batas gaya aktuator (N)
}
DT = 0.005
T_END = 10.0

# ============================================================
# 2. Fuzzy Logic Controller Sugeno orde nol
# ============================================================
LABELS = ["NB", "NS", "ZE", "PS", "PB"]
CENTERS = np.linspace(-1.0, 1.0, 5)

# Baris = e_theta [NB..PB], kolom = theta_dot [NB..PB]
RULE_DEFAULT = np.array([
    [-1.0, -1.0, -1.0, -0.5,  0.0],
    [-1.0, -1.0, -0.5,  0.0,  0.5],
    [-1.0, -0.5,  0.0,  0.5,  1.0],
    [-0.5,  0.0,  0.5,  1.0,  1.0],
    [ 0.0,  0.5,  1.0,  1.0,  1.0],
])

# Konfigurasi controller yang dapat diubah saat tuning
CFG_DEFAULT = {
    "E_SCALE_DEG": 12.0,     # |e_theta| = nilai ini -> |E| = 1
    "DE_SCALE_DEG": 100.0,   # |theta_dot| = nilai ini -> |DE| = 1
    "Fmax": 15.0,
    "and_op": "min",         # "min" atau "product"
    "rule": RULE_DEFAULT,
}


def make_cfg(**kw):
    cfg = dict(CFG_DEFAULT)
    cfg.update(kw)
    return cfg


def memberships(z):
    """Lima fungsi keanggotaan segitiga merata pada semesta [-1, 1]."""
    z = np.clip(z, -1.0, 1.0)
    return np.maximum(1.0 - np.abs(z - CENTERS) / 0.5, 0.0)


def fuzzy_force(e_theta, theta_dot, cfg=CFG_DEFAULT):
    """Inferensi AND (min/product) dan defuzzifikasi weighted average."""
    E = np.clip(e_theta / np.deg2rad(cfg["E_SCALE_DEG"]), -1.0, 1.0)
    DE = np.clip(theta_dot / np.deg2rad(cfg["DE_SCALE_DEG"]), -1.0, 1.0)
    mu_e = memberships(E)
    mu_de = memberships(DE)
    if cfg["and_op"] == "product":
        w = mu_e[:, None] * mu_de[None, :]
    else:
        w = np.minimum(mu_e[:, None], mu_de[None, :])
    y = np.sum(w * cfg["rule"]) / np.sum(w)
    return cfg["Fmax"] * y


# ============================================================
# 3. Plant nonlinier dan integrator RK4
# Keadaan s = [x, x_dot, theta, theta_dot]
# ============================================================
def plant(s, force, p=P):
    x, x_dot, theta, theta_dot = s
    M, m, l, b, g = p["M"], p["m"], p["l"], p["b"], p["g"]
    sn, cs = np.sin(theta), np.cos(theta)
    den = M + m - m * cs**2
    x_ddot = (force - b * x_dot + m * l * theta_dot**2 * sn
              - m * g * sn * cs) / den
    theta_ddot = (g * sn - x_ddot * cs) / l
    return np.array([x_dot, x_ddot, theta_dot, theta_ddot])


def rk4_step(s, force, dt, p=P):
    k1 = plant(s, force, p)
    k2 = plant(s + 0.5 * dt * k1, force, p)
    k3 = plant(s + 0.5 * dt * k2, force, p)
    k4 = plant(s + dt * k3, force, p)
    return s + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0


def simulate(closed_loop=True, theta0_deg=8.0, mass=0.20,
             push_N=6.0, push_start=5.0, push_duration=0.10,
             cfg=None, dt=DT, noise_std_deg=0.0, deadzone_N=0.0, seed=0):
    cfg = cfg or CFG_DEFAULT
    rng = np.random.default_rng(seed)
    p = P.copy()
    p["m"] = mass
    p["Fmax"] = cfg["Fmax"]
    t = np.arange(0.0, T_END + dt, dt)
    s = np.zeros((len(t), 4))
    u = np.zeros(len(t))
    theta_ref = np.zeros(len(t))
    disturbance = np.zeros(len(t))
    s[0, 2] = np.deg2rad(theta0_deg)

    for k in range(len(t) - 1):
        x, x_dot, theta, theta_dot = s[k]

        if closed_loop:
            # Loop posisi luar: menghasilkan referensi sudut kecil.
            theta_ref[k] = np.clip(-0.16 * x - 0.25 * x_dot,
                                   -np.deg2rad(6), np.deg2rad(6))
            theta_meas = theta
            if noise_std_deg > 0:
                theta_meas += np.deg2rad(rng.normal(0.0, noise_std_deg))
            e_theta = theta_meas - theta_ref[k]
            u[k] = fuzzy_force(e_theta, theta_dot, cfg)
            if abs(u[k]) < deadzone_N:
                u[k] = 0.0
        else:
            u[k] = 0.0

        if push_start <= t[k] < push_start + push_duration:
            disturbance[k] = push_N

        total_force = u[k] + disturbance[k]
        s[k + 1] = rk4_step(s[k], total_force, dt, p)

        # Hentikan simulasi jika pendulum jatuh melewati 80 derajat.
        if abs(s[k + 1, 2]) > np.deg2rad(80):
            s[k + 2:] = np.nan
            break

    theta_ref[-1] = theta_ref[-2]
    u[-1] = u[-2]
    return t, s, u, theta_ref, disturbance


def metrics(t, s, u, disturbance, band=1.0):
    theta_deg = np.rad2deg(s[:, 2])
    valid = np.isfinite(theta_deg)
    push_idx = np.where(disturbance > 0)[0]
    after_push = push_idx[-1] + 1 if len(push_idx) else 0

    def settling_time(start):
        for k in range(start, len(t)):
            if np.all(np.abs(theta_deg[k:][valid[k:]]) <= band):
                return t[k] - t[start]
        return np.nan

    fell = (not np.all(valid)) or np.nanmax(np.abs(theta_deg)) >= 20.0
    return {
        "peak_angle_deg": float(np.nanmax(np.abs(theta_deg))),
        "max_position_m": float(np.nanmax(np.abs(s[:, 0]))),
        "max_force_N": float(np.nanmax(np.abs(u))),
        "recovery_s": float(settling_time(after_push)),
        "stabil": "Tidak" if fell else "Ya",
    }


# ============================================================
# 4. Utilitas plot dan penyimpanan tabel
# ============================================================
def plot_result(title, result, fname):
    t, s, u, theta_ref, disturbance = result
    fig, ax = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    ax[0].plot(t, np.rad2deg(s[:, 2]), label="theta")
    ax[0].plot(t, np.rad2deg(theta_ref), "--", label="theta_ref")
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].set_ylabel("Sudut (deg)")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)
    ax[1].plot(t, s[:, 0], color="tab:green")
    ax[1].set_ylabel("Posisi x (m)")
    ax[1].grid(True, alpha=0.3)
    ax[2].plot(t, u, label="gaya kendali")
    ax[2].plot(t, disturbance, "--", label="gangguan")
    ax[2].set_ylabel("Gaya (N)")
    ax[2].set_xlabel("Waktu (s)")
    ax[2].legend()
    ax[2].grid(True, alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, fname)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("  grafik disimpan:", path)


def save_table(fname, header, rows):
    path = os.path.join(OUT_DIR, fname)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print("  tabel disimpan:", path)


def print_table(header, rows):
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows))
              for i, h in enumerate(header)]
    line = " | ".join(str(h).ljust(w) for h, w in zip(header, widths))
    print(line)
    print("-" * len(line))
    for r in rows:
        print(" | ".join(str(c).ljust(w) for c, w in zip(r, widths)))


def fmt(v, nd=3):
    return "-" if (v is None or (isinstance(v, float) and np.isnan(v))) else round(v, nd)


# ============================================================
# 5. Eksperimen
# ============================================================
def exp_A():
    """Eksperimen A: verifikasi open-loop."""
    print("\n=== Eksperimen A: open-loop ===")
    rows = []
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for th in (2, 4, 8):
        t, s, u, ref, d = simulate(closed_loop=False, theta0_deg=th, push_N=0.0)
        a = np.abs(np.rad2deg(s[:, 2]))
        a_valid = np.where(np.isfinite(a), a, np.inf)

        def first(v):
            idx = np.where(a_valid > v)[0]
            return round(float(t[idx[0]]), 3) if len(idx) else None

        rows.append([f"{th} deg", first(30), first(60),
                     "Jatuh (tidak stabil)" if first(30) else "Tidak jatuh"])
        ax.plot(t, np.rad2deg(s[:, 2]), label=f"theta0 = {th} deg")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-100, 100)
    ax.set_xlabel("Waktu (s)")
    ax.set_ylabel("Sudut (deg)")
    ax.set_title("Open-loop: pendulum jatuh dari sudut awal kecil")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "A_open_loop.png"), dpi=150)
    plt.close(fig)
    header = ["theta awal", "t |theta|>30 (s)", "t |theta|>60 (s)", "Kesimpulan"]
    print_table(header, rows)
    save_table("A_open_loop.csv", header, rows)


def exp_B():
    """Eksperimen B: respons nominal closed-loop."""
    print("\n=== Eksperimen B: respons nominal closed-loop ===")
    res = simulate()
    mt = metrics(res[0], res[1], res[2], res[4])
    targets = [
        ("Maksimum |theta| (deg)", mt["peak_angle_deg"], "< 20", mt["peak_angle_deg"] < 20),
        ("Waktu pemulihan (s)", mt["recovery_s"], "<= 2", mt["recovery_s"] <= 2),
        ("Maksimum |x| (m)", mt["max_position_m"], "<= 0.25", mt["max_position_m"] <= 0.25),
        ("Maksimum |u| (N)", mt["max_force_N"], "<= 15", mt["max_force_N"] <= 15),
    ]
    rows = [[n, fmt(v), tg, "Lulus" if ok else "Tidak lulus"] for n, v, tg, ok in targets]
    header = ["Metrik", "Hasil", "Target", "Status"]
    print_table(header, rows)
    save_table("B_nominal.csv", header, rows)
    plot_result("FLC closed-loop (nominal)", res, "B_nominal.png")

    # grafik open-loop untuk pembanding
    ol = simulate(closed_loop=False, push_N=0.0)
    plot_result("Open-loop (theta0 = 8 deg)", ol, "B_open_loop.png")


def exp_C():
    """Eksperimen C: ketahanan terhadap variasi satu faktor."""
    print("\n=== Eksperimen C: ketahanan ===")
    cases = [(4, 0.20, 6), (8, 0.20, 6), (12, 0.20, 6),
             (8, 0.15, 6), (8, 0.30, 6), (8, 0.20, 10)]
    rows = []
    fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for i, (th, m, push) in enumerate(cases, 1):
        res = simulate(theta0_deg=th, mass=m, push_N=push)
        mt = metrics(res[0], res[1], res[2], res[4])
        rows.append([i, f"{th} deg", f"{m:.2f} kg", f"{push} N",
                     fmt(mt["peak_angle_deg"], 2), fmt(mt["recovery_s"]),
                     fmt(mt["max_position_m"]), mt["stabil"]])
        lbl = f"#{i}: th0={th}, m={m}, d={push}"
        ax[0].plot(res[0], np.rad2deg(res[1][:, 2]), label=lbl)
        ax[1].plot(res[0], res[1][:, 0])
        ax[2].plot(res[0], res[2])
    ax[0].set_ylabel("Sudut (deg)")
    ax[0].legend(fontsize=7)
    ax[1].set_ylabel("Posisi x (m)")
    ax[2].set_ylabel("Gaya u (N)")
    ax[2].set_xlabel("Waktu (s)")
    for a in ax:
        a.grid(True, alpha=0.3)
    fig.suptitle("Eksperimen C: variasi sudut awal, massa, dan gangguan")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "C_ketahanan.png"), dpi=150)
    plt.close(fig)
    header = ["No", "theta0", "m", "Gangguan", "Peak |theta| (deg)",
              "Recovery (s)", "Max |x| (m)", "Stabil"]
    print_table(header, rows)
    save_table("C_ketahanan.csv", header, rows)


def exp_D():
    """Eksperimen D: skala input dan gaya maksimum."""
    print("\n=== Eksperimen D: skala input dan Fmax ===")
    configs = {
        "Dasar": (12, 100, 15),
        "A": (8, 80, 15),
        "B": (16, 120, 15),
        "C": (12, 100, 10),
        "D": (12, 100, 20),
    }
    rows = []
    fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for name, (es, des, fm) in configs.items():
        cfg = make_cfg(E_SCALE_DEG=es, DE_SCALE_DEG=des, Fmax=fm)
        res = simulate(cfg=cfg)
        mt = metrics(res[0], res[1], res[2], res[4])
        rows.append([name, f"{es} deg", f"{des} deg/s", f"{fm} N",
                     fmt(mt["peak_angle_deg"], 2), fmt(mt["recovery_s"]),
                     fmt(mt["max_position_m"]), fmt(mt["max_force_N"], 2)])
        ax[0].plot(res[0], np.rad2deg(res[1][:, 2]), label=name)
        ax[1].plot(res[0], res[1][:, 0])
        ax[2].plot(res[0], res[2])
    ax[0].set_ylabel("Sudut (deg)")
    ax[0].legend()
    ax[1].set_ylabel("Posisi x (m)")
    ax[2].set_ylabel("Gaya u (N)")
    ax[2].set_xlabel("Waktu (s)")
    for a in ax:
        a.grid(True, alpha=0.3)
    fig.suptitle("Eksperimen D: pengaruh E_SCALE, DE_SCALE, dan Fmax")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "D_tuning.png"), dpi=150)
    plt.close(fig)
    header = ["Konfigurasi", "E_SCALE", "DE_SCALE", "Fmax", "Peak |theta| (deg)",
              "Recovery (s)", "Max |x| (m)", "Max |u| (N)"]
    print_table(header, rows)
    save_table("D_tuning.csv", header, rows)


def exp_E():
    """Eksperimen E: modifikasi basis aturan (ubah 2-4 sel, satu per satu).

    EDIT daftar `changes` di bawah. Tulis hipotesis fisik sebelum menjalankan.
    Indeks baris/kolom: 0=NB, 1=NS, 2=ZE, 3=PS, 4=PB.
    Setiap perubahan dijalankan TERPISAH dari aturan dasar agar efeknya dapat ditelusuri.
    """
    print("\n=== Eksperimen E: modifikasi basis aturan ===")
    changes = [
        # (baris=E, kolom=DE, nilai_baru, hipotesis)
        (3, 2, 1.0, "E=PS, DE=ZE: gaya diperbesar 0.5 -> 1.0, koreksi diharapkan lebih cepat"),
        (1, 3, -0.5, "E=NS, DE=PS: pendulum sudah kembali ke tegak; gaya negatif tambahan akan menghambat"),
        (2, 3, 1.0, "E=ZE, DE=PS: redaman lebih kuat, efek kecil karena sel hanya aktif dekat tegak"),
    ]
    base = simulate()
    mb = metrics(base[0], base[1], base[2], base[4])
    rows = []
    results = []
    for (r, c, val, hip) in changes:
        rule = RULE_DEFAULT.copy()
        awal = rule[r, c]
        rule[r, c] = val
        res = simulate(cfg=make_cfg(rule=rule))
        mt = metrics(res[0], res[1], res[2], res[4])
        results.append((f"({LABELS[r]},{LABELS[c]}): {awal} -> {val}", res))
        hasil = (f"rec={fmt(mt['recovery_s'])} s (dasar {fmt(mb['recovery_s'])} s), "
                 f"max|x|={mt['max_position_m']:.3f} m, max|u|={mt['max_force_N']:.2f} N")
        rows.append([f"({LABELS[r]},{LABELS[c]})", awal, val, hip, hasil])
    header = ["Sel aturan", "Nilai awal", "Nilai baru", "Hipotesis", "Hasil"]
    print_table(header, rows)
    save_table("E_aturan.csv", header, rows)

    fig, ax = plt.subplots(2, 1, figsize=(9, 7))
    for a_, (x0, x1) in zip(ax, [(0, 10), (4.9, 7.0)]):
        a_.plot(base[0], np.rad2deg(base[1][:, 2]), "k", lw=2, label="aturan dasar")
        for nama, res in results:
            a_.plot(res[0], np.rad2deg(res[1][:, 2]), label=nama)
        a_.axhline(0, color="gray", lw=0.6)
        a_.axhline(1, color="gray", lw=0.6, ls=":")
        a_.axhline(-1, color="gray", lw=0.6, ls=":")
        a_.set_xlim(x0, x1)
        a_.set_ylabel("Sudut (deg)")
        a_.grid(True, alpha=0.3)
    ax[0].legend(fontsize=8)
    ax[1].set_xlabel("Waktu (s)")
    ax[0].set_title("Eksperimen E: dampak perubahan satu sel aturan (atas: penuh, bawah: sekitar gangguan)")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "E_aturan.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("  grafik disimpan:", path)


def exp_mf():
    """Grafik fungsi keanggotaan segitiga (disimpan sebagai hasil/mf.png)."""
    print("\n=== Fungsi keanggotaan ===")
    z = np.linspace(-1, 1, 401)
    mu = np.array([memberships(v) for v in z])
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for i, lbl in enumerate(LABELS):
        ax.plot(z, mu[:, i], label=lbl)
    ax.set_xlabel("Variabel ternormalisasi (E atau DE)")
    ax.set_ylabel("Derajat keanggotaan")
    ax.set_title("Fungsi keanggotaan segitiga")
    ax.legend(ncol=5, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "mf.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("  grafik disimpan:", path)


def exp_dt():
    """Cek galat integrasi: bandingkan DT 0.005, 0.01, 0.02 s."""
    print("\n=== Cek pengaruh DT ===")
    rows = []
    for dt in (0.005, 0.01, 0.02):
        res = simulate(dt=dt)
        mt = metrics(res[0], res[1], res[2], res[4])
        rows.append([dt, fmt(mt["peak_angle_deg"], 3), fmt(mt["recovery_s"]),
                     fmt(mt["max_position_m"], 4), fmt(mt["max_force_N"], 3)])
    header = ["DT (s)", "Peak |theta| (deg)", "Recovery (s)", "Max |x| (m)", "Max |u| (N)"]
    print_table(header, rows)
    save_table("dt_check.csv", header, rows)


def exp_surface():
    """Peta permukaan keluaran FLC terhadap E dan DE."""
    print("\n=== Permukaan keluaran FLC ===")
    e = np.linspace(-1, 1, 61)
    de = np.linspace(-1, 1, 61)
    Z = np.zeros((len(e), len(de)))
    for i, ev in enumerate(e):
        for j, dv in enumerate(de):
            Z[i, j] = fuzzy_force(ev * np.deg2rad(12.0), dv * np.deg2rad(100.0))
    EE, DD = np.meshgrid(de, e)
    fig = plt.figure(figsize=(7, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(EE, DD, Z, cmap="viridis")
    ax.set_xlabel("DE (ternormalisasi)")
    ax.set_ylabel("E (ternormalisasi)")
    ax.set_zlabel("Gaya u (N)")
    ax.set_title("Permukaan keluaran FLC")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "surface.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("  grafik disimpan:", path)


def exp_opsional():
    """Tantangan opsional: noise sensor, dead-zone, AND min vs product."""
    print("\n=== Tantangan opsional ===")
    skenario = [
        ("Nominal", dict()),
        ("Noise sensor 0.1 deg", dict(noise_std_deg=0.1)),
        ("Dead-zone 0.5 N", dict(deadzone_N=0.5)),
        ("AND product", dict(cfg=make_cfg(and_op="product"))),
    ]
    rows = []
    for nama, kw in skenario:
        res = simulate(**kw)
        mt = metrics(res[0], res[1], res[2], res[4])
        rows.append([nama, fmt(mt["peak_angle_deg"], 2), fmt(mt["recovery_s"]),
                     fmt(mt["max_position_m"]), fmt(mt["max_force_N"], 2), mt["stabil"]])
    header = ["Skenario", "Peak |theta| (deg)", "Recovery (s)", "Max |x| (m)",
              "Max |u| (N)", "Stabil"]
    print_table(header, rows)
    save_table("opsional.csv", header, rows)


# ============================================================
# 6. Main
# ============================================================
if __name__ == "__main__":
    pilihan = sys.argv[1].lower() if len(sys.argv) > 1 else "semua"
    peta = {
        "a": exp_A, "nominal": exp_B, "b": exp_B, "c": exp_C, "d": exp_D,
        "e": exp_E, "mf": exp_mf, "dt": exp_dt, "surface": exp_surface, "opsional": exp_opsional,
    }
    if pilihan == "semua":
        for fn in (exp_A, exp_B, exp_C, exp_D, exp_E, exp_mf, exp_dt, exp_surface, exp_opsional):
            fn()
    elif pilihan in peta:
        peta[pilihan]()
    else:
        print(__doc__)
    print("\nSelesai. Hasil ada di folder:", os.path.abspath(OUT_DIR))
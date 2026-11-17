# =============================================================
#  Pacejka Magic Formula tyre app (Pacejka 2004 book + MF6.1 pressure)
#  Formulas copied one-for-one from the Excel sheet "Sheet1".
#  Run with:   streamlit run tyre_app.py
# =============================================================
import re
import numpy as np

import plotly.graph_objects as go
import streamlit as st

# Numpy versions of the Excel functions, so formulas read like the sheet
SIN, COS, ATAN, EXP, SQRT, ABS, SIGN = np.sin, np.cos, np.arctan, np.exp, np.sqrt, np.abs, np.sign
PI = np.pi
MIN = np.minimum


# -------------------------------------------------------------
# 1. Read the .tir file  ->  dictionary {"PCX1": 2.307, ...}
# -------------------------------------------------------------
def read_tir(text):
    P = {}
    for line in text.splitlines():
        m = re.match(r"\s*([A-Za-z_0-9]+)\s*=\s*([-+0-9.eE]+)", line)
        if m:
            try:
                P[m.group(1).upper()] = float(m.group(2))
            except ValueError:
                pass
    return P


# -------------------------------------------------------------
# 2. The tyre model  (cell references from Sheet1 in comments)
# -------------------------------------------------------------
def tyre_model(P, Fz, alpha_deg, camber_deg, kappa, p_kpa, L, zeta=1.0):
    g = lambda k: P.get(k, 0.0)            # missing coefficient -> 0

    # ---- inputs (C2:C7, D4, D5, G4:G6) ----
    alpha = alpha_deg * PI / 180            # D4
    gam = SIN(camber_deg * PI / 180)        # D5  (sin of camber)
    Fz0 = g("FNOMIN")                       # C3
    Fz0p = 1.0 * Fz0                        # G4  Fz0' (scale G3 = 1)
    dfz = (Fz - Fz0p) / Fz0p                # G5
    dpi = (p_kpa * 1000 - g("NOMPRES")) / g("NOMPRES")   # G6
    R0 = g("UNLOADED_RADIUS")               # P3
    Vx = g("LONGVL")                        # M4

    # ---- pure longitudinal Fx0 (C9:C18) ----
    Shx = (g("PHX1") + g("PHX2") * dfz) * L["LHX"]                                   # C17
    Svx = Fz * (g("PVX1") + g("PVX2") * dfz) * (Vx / (Vx + 0.1)) * L["LVX"] * L["LMU"]  # C18
    kx = kappa + Shx                                                                  # C10
    Cx = g("PCX1") * L["LC"]                                                          # C11
    mux = (g("PDX1") + g("PDX2") * dfz) * (1 + g("PPX3") * dpi + g("PPX4") * dpi**2) * L["LMU"]  # C13
    Dx = mux * Fz                                                                     # C12
    Ex = (g("PEX1") + g("PEX2") * dfz + g("PEX3") * dfz * dfz) * (1 - g("PEX4") * SIGN(kx)) * L["LE"]  # C14
    Kxk = Fz * L["LKX"] * (g("PKX1") + g("PKX2") * dfz) * EXP(g("PKX3") * dfz) * (1 + g("PPX1") * dpi + g("PPX2") * dpi**2)  # C15
    Bx = Kxk / (Cx * Dx + 0.1)                                                        # C16
    Fx0 = Dx * SIN(Cx * ATAN(Bx * kx - Ex * (Bx * kx - ATAN(Bx * kx)))) + Svx         # C9

    # ---- pure lateral Fy0 (C21:C32) ----
    Cy = g("PCY1") * L["LC"]                                                          # C23
    muy = ((g("PDY1") + g("PDY2") * dfz) * (1 + g("PPY3") * dpi + g("PPY4") * dpi**2) / (1 + g("PDY3") * gam * gam)) * L["LMU"]  # C25
    Dy = muy * Fz                                                                     # C24
    Ey = (g("PEY1") + g("PEY2") * dfz) * (1 + g("PEY5") * gam * gam - (g("PEY3") + g("PEY4") * gam) * SIGN(alpha)) * L["LE"]  # C26
    Kya = (g("PKY1") * Fz0p * (1 + g("PPY1") * dpi)
           * SIN(g("PKY4") * ATAN(Fz / ((g("PKY2") + g("PKY5") * gam * gam) * (1 + g("PPY2") * dpi) * Fz0p)))
           / (1 + g("PKY3") * gam * gam) * L["LKY"])                                  # C27
    By = Kya / (Cy * Dy + 0.1)                                                        # C28
    Kyg0 = Fz * (g("PKY6") + g("PKY7") * dfz) * (1 + g("PPY5") * dpi) * L["LKYC"]     # C32
    Svyg = Fz * (g("PVY3") + g("PVY4") * dfz) * gam * L["LKYC"] * L["LMU"]            # C30
    Svy = Fz * (g("PVY1") + g("PVY2") * dfz) * L["LVY"] * L["LMU"] + Svyg             # C31
    Shy = (g("PHY1") + g("PHY2") * dfz) * L["LHY"] + (Kyg0 * gam - Svyg) / Kya        # C29
    ay = alpha + Shy                                                                  # C22
    Fy0 = Dy * SIN(Cy * ATAN(By * ay - Ey * (By * ay - ATAN(By * ay)))) + Svy         # C21

    # ---- pure aligning torque Mz0 (C35:C51) ----
    SHt = g("QHZ1") + g("QHZ2") * dfz + (g("QHZ3") + g("QHZ4") * dfz) * gam           # C35
    at = alpha + SHt                                                                  # C36
    Kyap = Kya                                                                        # C37
    SHf = Shy + Svy / Kyap                                                            # C38
    ar = alpha + SHf                                                                  # C39
    Bt = ((g("QBZ1") + g("QBZ2") * dfz + g("QBZ3") * dfz**2)
          * (1 + g("QBZ4") * gam + g("QBZ5") * ABS(gam) + g("QBZ6") * gam**2) * L["LKY"] / L["LMU"])  # C40
    Ct = g("QCZ1")                                                                    # C41
    Dt0 = Fz * (R0 / Fz0p) * (g("QDZ1") + g("QDZ2") * dfz) * (1 - g("PPZ1") * dpi) * L["LTR"] * SIGN(Vx)  # C42
    Dt = Dt0 * (1 + g("QDZ3") * ABS(gam) + g("QDZ4") * gam**2)                        # C43
    Et = MIN((g("QEZ1") + g("QEZ2") * dfz + g("QEZ3") * dfz**2)
             * (1 + (g("QEZ4") + g("QEZ5") * gam) * (2 / PI) * ATAN(Bt * Ct * at)), 1)  # C44
    t0 = Dt * COS(Ct * ATAN(Bt * at - Et * (Bt * at - ATAN(Bt * at)))) * COS(alpha)   # C45
    Mzp0 = -t0 * Fy0                                                                  # C46
    Br = g("QBZ9") * L["LKY"] / L["LMU"] + g("QBZ10") * By * Cy                       # C47
    Cr = zeta                                                                         # C48
    Dr = (Fz * R0 * ((g("QDZ6") + g("QDZ7") * dfz) * L["LRES"]
                     + (g("QDZ8") + g("QDZ9") * dfz) * (1 + g("PPZ2") * dpi) * gam * L["LKZC"]
                     + (g("QDZ10") + g("QDZ11") * dfz) * gam * ABS(gam))
          * COS(alpha) * L["LMU"] * SIGN(Vx))                                         # C49
    Mzr0 = Dr * COS(Cr * ATAN(Br * ar))                                               # C50
    Mz0 = Mzp0 + Mzr0                                                                 # C51

    # ---- combined longitudinal Fx (C54:C61) ----
    SHxa = g("RHX1")                                                                  # C54
    a_s = alpha + SHxa                                                                # C55
    Bxa = (g("RBX1") + g("RBX3") * gam**2) * COS(ATAN(g("RBX2") * kappa)) * L["LXAL"]  # C56
    Cxa = g("RCX1")                                                                   # C57
    Exa = MIN(g("REX1") + g("REX2") * dfz, 1)                                         # C58
    Gxa0 = COS(Cxa * ATAN(Bxa * SHxa - Exa * (Bxa * SHxa - ATAN(Bxa * SHxa))))        # C59
    Gxa = COS(Cxa * ATAN(Bxa * a_s - Exa * (Bxa * a_s - ATAN(Bxa * a_s)))) / Gxa0     # C60
    Fx = Gxa * Fx0                                                                    # C61

    # ---- combined lateral Fy (C64:C73) ----
    SHyk = g("RHY1") + g("RHY2") * dfz                                                # C64
    ks = kappa + SHyk                                                                 # C65
    Byk = (g("RBY1") + g("RBY4") * gam**2) * COS(ATAN(g("RBY2") * (alpha - g("RBY3")))) * L["LYKA"]  # C66
    Cyk = g("RCY1")                                                                   # C67
    Eyk = MIN(g("REY1") + g("REY2") * dfz, 1)                                         # C68
    DVyk = muy * Fz * (g("RVY1") + g("RVY2") * dfz + g("RVY3") * gam) * COS(ATAN(g("RVY4") * alpha))  # C69
    SVyk = DVyk * SIN(g("RVY5") * ATAN(g("RVY6") * kappa)) * L["LVYKA"]               # C70
    Gyk0 = COS(Cyk * ATAN(Byk * SHyk - Eyk * (Byk * SHyk - ATAN(Byk * SHyk))))        # C71
    Gyk = COS(Cyk * ATAN(Byk * ks - Eyk * (Byk * ks - ATAN(Byk * ks)))) / Gyk0        # C72
    Fy = Gyk * Fy0 + SVyk                                                             # C73

    # ---- combined aligning torque Mz (C76:C84) ----
    ratio = Kxk / Kyap                                                                # C76
    at_eq = SQRT(at**2 + (ratio * kappa)**2) * SIGN(at)                               # C77
    ar_eq = SQRT(ar**2 + (ratio * kappa)**2) * SIGN(ar)                               # C78
    t = Dt * COS(Ct * ATAN(Bt * at_eq - Et * (Bt * at_eq - ATAN(Bt * at_eq)))) * COS(alpha)  # C79
    Fyp = Fy - SVyk                                                                   # C80
    Mzp = -t * Fyp                                                                    # C81
    Mzr = Dr * COS(Cr * ATAN(Br * ar_eq))                                             # C82
    s = R0 * (g("SSZ1") + g("SSZ2") * (Fy / Fz0p) + (g("SSZ3") + g("SSZ4") * dfz) * gam) * L["LS"]  # C83
    Mz = Mzp + Mzr + s * Fx                                                           # C84

    # ---- overturning moment Mx (C87:C88) ----
    Mx0 = Fz * R0 * (g("QSX1") - g("QSX2") * gam * (1 + g("PPMX1") * dpi) + g("QSX3") * Fy0 / Fz0p) * L["LMX"]  # C87
    Mx = Fz * R0 * (g("QSX1") - g("QSX2") * gam * (1 + g("PPMX1") * dpi) + g("QSX3") * Fy / Fz0p) * L["LMX"]    # C88

    return dict(Fx0=Fx0, Fy0=Fy0, Mz0=Mz0, Fx=Fx, Fy=Fy, Mz=Mz, Mx0=Mx0, Mx=Mx,
                kx=kx, Cx=Cx, Dx=Dx, mux=mux, Ex=Ex, Kxk=Kxk, Bx=Bx, Shx=Shx, Svx=Svx,
                ay=ay, Cy=Cy, Dy=Dy, muy=muy, Ey=Ey, Kya=Kya, By=By, Shy=Shy, Svyg=Svyg, Svy=Svy, Kyg0=Kyg0,
                SHt=SHt, Bt=Bt, Ct=Ct, Dt=Dt, Et=Et, t0=t0, Br=Br, Dr=Dr, Mzr0=Mzr0,
                Bxa=Bxa, Exa=Exa, Gxa=Gxa, Byk=Byk, Eyk=Eyk, SVyk=SVyk, Gyk=Gyk,
                at_eq=at_eq, ar_eq=ar_eq, t=t, Mzr=Mzr, s=s)


# -------------------------------------------------------------
# 3. The app screen
# -------------------------------------------------------------
st.set_page_config(page_title="Pacejka Tyre Model", layout="wide")
st.title("Pacejka Magic Formula Tyre Model")

tir_file = st.sidebar.file_uploader("Load tyre file (.tir or .txt)", type=["tir", "txt"])
if tir_file is None:
    st.info("Load your .tir file in the left panel to start.")
    st.stop()
P = read_tir(tir_file.read().decode("utf-8", errors="ignore"))
st.sidebar.success(f"Loaded {len(P)} parameters  |  FNOMIN = {P.get('FNOMIN', 0):.0f} N")

st.sidebar.header("Operating point")
Fz = st.sidebar.number_input("Vertical load Fz (N)", 50.0, 5000.0, float(P.get("FNOMIN", 1000)), 10.0)
alpha_deg = st.sidebar.slider("Slip angle (deg)", -15.0, 15.0, 5.0, 0.1)
kappa = st.sidebar.slider("Slip ratio", -0.30, 0.30, 0.05, 0.005)
camber_deg = st.sidebar.slider("Camber (deg)", -6.0, 6.0, 0.0, 0.1)
p_kpa = st.sidebar.number_input("Pressure (kPa)", 30.0, 200.0, P.get("INFLPRES", P.get("NOMPRES", 83000)) / 1000, 1.0)

st.sidebar.header("Scaling factors (λ)")
with st.sidebar.expander("Edit scaling factors"):
    L = {
        "LMU":   st.number_input("λμ  peak friction", value=P.get("LMUY", 1.0), step=0.05),
        "LKX":   st.number_input("λKxκ  slip stiffness", value=P.get("LKX", 1.0), step=0.05),
        "LKY":   st.number_input("λKyα  cornering stiffness", value=P.get("LKY", 1.0), step=0.05),
        "LC":    st.number_input("λC  shape factor", value=P.get("LCY", 1.0), step=0.05),
        "LE":    st.number_input("λE  curvature factor", value=P.get("LEY", 1.0), step=0.05),
        "LHX":   st.number_input("λHx  horizontal shift x", value=P.get("LHX", 1.0), step=0.05),
        "LHY":   st.number_input("λHy  horizontal shift y", value=P.get("LHY", 1.0), step=0.05),
        "LVX":   st.number_input("λVx  vertical shift x", value=P.get("LVX", 1.0), step=0.05),
        "LVY":   st.number_input("λVy  vertical shift y", value=P.get("LVY", 1.0), step=0.05),
        "LKYC":  st.number_input("λKyγ  camber force stiffness", value=P.get("LKYC", 1.0), step=0.05),
        "LKZC":  st.number_input("λKzγ  camber torque stiffness", value=P.get("LKZC", 1.0), step=0.05),
        "LTR":   st.number_input("λt  pneumatic trail", value=P.get("LTR", 1.0), step=0.05),
        "LRES":  st.number_input("λMr  residual torque", value=P.get("LRES", 1.0), step=0.05),
        "LXAL":  st.number_input("λxα  α influence on Fx", value=P.get("LXAL", 1.0), step=0.05),
        "LYKA":  st.number_input("λyκ  κ influence on Fy", value=P.get("LYKA", 1.0), step=0.05),
        "LVYKA": st.number_input("λVyκ  κ-induced ply-steer", value=P.get("LVYKA", 1.0), step=0.05),
        "LS":    st.number_input("λs  Mz moment arm of Fx", value=P.get("LS", 1.0), step=0.05),
        "LMX":   st.number_input("λMx  overturning couple", value=P.get("LMX", 1.0), step=0.05),
    }

# ---- results at the operating point ----
r = tyre_model(P, Fz, alpha_deg, camber_deg, kappa, p_kpa, L)

st.subheader("Results at this operating point")
c = st.columns(4)
c[0].metric("Pure Fx0 (N)", f"{r['Fx0']:.1f}")
c[1].metric("Pure Fy0 (N)", f"{r['Fy0']:.1f}")
c[2].metric("Pure Mz0 (Nm)", f"{r['Mz0']:.2f}")
c[3].metric("Mx0 (Nm)", f"{r['Mx0']:.2f}")
c = st.columns(4)
c[0].metric("Combined Fx (N)", f"{r['Fx']:.1f}")
c[1].metric("Combined Fy (N)", f"{r['Fy']:.1f}")
c[2].metric("Combined Mz (Nm)", f"{r['Mz']:.2f}")
c[3].metric("Combined Mx (Nm)", f"{r['Mx']:.2f}")

with st.expander("Show all intermediate values (compare with Excel)"): st.markdown("\n".join(["| Name | Value |", "|---|---|"] + [f"| {k} | {float(v):.6g} |" for k, v in r.items()]))

# ---- sweep charts ----
st.subheader("Sweeps")
loads_txt = st.text_input("Normal loads to plot (N, comma separated)", "250, 500, 750, 1000, 1250, 1500")
load_list = [float(x) for x in loads_txt.split(",") if x.strip()]
alphas_txt = st.text_input("Slip angles to plot (deg, comma separated)", "0, 2, 4, 6, 8")
alpha_list = [float(a) for a in alphas_txt.split(",") if a.strip()]
mode = st.radio("Combined-slip charts: one curve per", ["Normal load (at sidebar slip angle)", "Slip angle (at sidebar load)"], horizontal=True)
k_sweep = np.linspace(-0.2, 0.2, 201)
a_sweep = np.linspace(-12, 12, 241)

def line_chart(title, xlab, ylab):
    f = go.Figure()
    f.update_layout(title=title, xaxis_title=xlab, yaxis_title=ylab, height=420,
                    margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.2))
    return f

# combined-slip charts
if mode.startswith("Normal load"):
    cases = [(fz_i, alpha_deg, f"Fz = {fz_i:g} N") for fz_i in load_list]
    note = f"α = {alpha_deg:g}°"
else:
    cases = [(Fz, a, f"α = {a:g}°") for a in alpha_list]
    note = f"Fz = {Fz:g} N"
f_fx = line_chart(f"Combined Fx vs slip ratio ({note})", "Slip ratio κ", "Fx (N)")
f_fy = line_chart(f"Combined Fy vs slip ratio ({note})", "Slip ratio κ", "Fy (N)")
f_fc = line_chart(f"Friction circle ({note})", "Fx (N)", "Fy (N)")
for fz_i, a_i, name in cases:
    rs = tyre_model(P, fz_i, a_i, camber_deg, k_sweep, p_kpa, L)
    f_fx.add_trace(go.Scatter(x=k_sweep, y=rs["Fx"], name=name))
    f_fy.add_trace(go.Scatter(x=k_sweep, y=rs["Fy"], name=name))
    f_fc.add_trace(go.Scatter(x=rs["Fx"], y=rs["Fy"], name=name))

# pure-slip charts: one curve per normal load
f_fya = line_chart("Pure Fy0 vs slip angle (κ = 0)", "Slip angle (deg)", "Fy0 (N)")
f_mz = line_chart("Pure Mz0 vs slip angle (κ = 0)", "Slip angle (deg)", "Mz0 (Nm)")
f_fxk = line_chart("Pure Fx0 vs slip ratio (α = 0)", "Slip ratio κ", "Fx0 (N)")
f_mx = line_chart("Mx0 vs slip angle (κ = 0)", "Slip angle (deg)", "Mx0 (Nm)")
for fz_i in load_list:
    ra = tyre_model(P, fz_i, a_sweep, camber_deg, 0.0, p_kpa, L)
    rk = tyre_model(P, fz_i, 0.0, camber_deg, k_sweep, p_kpa, L)
    f_fya.add_trace(go.Scatter(x=a_sweep, y=ra["Fy0"], name=f"Fz = {fz_i:g} N"))
    f_mz.add_trace(go.Scatter(x=a_sweep, y=ra["Mz0"], name=f"Fz = {fz_i:g} N"))
    f_mx.add_trace(go.Scatter(x=a_sweep, y=ra["Mx0"], name=f"Fz = {fz_i:g} N"))
    f_fxk.add_trace(go.Scatter(x=k_sweep, y=rk["Fx0"], name=f"Fz = {fz_i:g} N"))

col1, col2 = st.columns(2)
col1.plotly_chart(f_fx, use_container_width=True)
col2.plotly_chart(f_fy, use_container_width=True)
col1.plotly_chart(f_fc, use_container_width=True)
col2.plotly_chart(f_fya, use_container_width=True)
col1.plotly_chart(f_fxk, use_container_width=True)
col2.plotly_chart(f_mz, use_container_width=True)
col1.plotly_chart(f_mx, use_container_width=True)
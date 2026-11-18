"""
Quantum Teleportation — Streamlit app
=====================================

Run:
    pip install streamlit numpy plotly
    streamlit run app.py
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ------------------------------------------------------------------ page & colours
st.set_page_config(page_title="Quantum Teleportation", page_icon="🌀", layout="wide")

BG = "#F6F3EC"
PANEL = "#EAF0EE"
INK = "#1F2B33"
MUTED = "#4A5560"
TEAL = "#2F7A78"
ORANGE = "#C9733F"
GREY = "#B9B4A8"

st.markdown(f"""
<style>
.stApp {{ background: {BG}; }}
section[data-testid="stSidebar"] {{ background: {PANEL}; }}
h1, h2, h3 {{ color: {INK}; }}
.step-box {{ background: {PANEL}; border-left: 6px solid {ORANGE}; border-radius: 12px;
            padding: 16px 20px; font-size: 1.05rem; color: {INK}; line-height: 1.6; }}
.small-note {{ color: {MUTED}; font-size: 0.92rem; }}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------ quantum core
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
ket0 = np.array([1, 0], dtype=complex)
ket1 = np.array([0, 1], dtype=complex)
BELL00 = (np.kron(ket0, ket0) + np.kron(ket1, ket1)) / np.sqrt(2)


def apply_1q(state, n, q, U):
    M = np.array([[1]], dtype=complex)
    for k in range(n):
        M = np.kron(M, U if k == q else I2)
    return M @ state


def apply_cnot(state, n, control, target):
    new = state.copy()
    for i in range(2 ** n):
        bits = [(i >> (n - 1 - k)) & 1 for k in range(n)]
        if bits[control]:
            bits[target] ^= 1
            new[int("".join(map(str, bits)), 2)] = state[i]
    return new


def reduced(state, n, q):
    t = np.moveaxis(state.reshape([2] * n), q, 0).reshape(2, -1)
    return t @ t.conj().T


def bloch_vector(rho):
    return np.array([np.real(np.trace(rho @ s)) for s in (X, Y, Z)])


def measure(state, n, qubits, rng):
    probs = np.abs(state) ** 2
    out = {}
    for i, p in enumerate(probs):
        key = tuple((i >> (n - 1 - q)) & 1 for q in qubits)
        out[key] = out.get(key, 0) + p
    keys = list(out)
    w = np.array([out[k] for k in keys])
    pick = keys[rng.choice(len(keys), p=w / w.sum())]
    new = state.copy()
    for i in range(2 ** n):
        if tuple((i >> (n - 1 - q)) & 1 for q in qubits) != pick:
            new[i] = 0
    return pick, new / np.linalg.norm(new)


def bob_qubit(state, m0, m1):
    base = (m0 << 2) | (m1 << 1)
    v = np.array([state[base], state[base | 1]])
    return v / np.linalg.norm(v)


def fidelity(a, b):
    return abs(np.vdot(a, b)) ** 2


FIX = {(0, 0): ("I", I2, "do nothing"),
       (0, 1): ("X", X, "swap 0 and 1"),
       (1, 0): ("Z", Z, "flip the sign of |1⟩"),
       (1, 1): ("iY", 1j * Y, "swap, then fix the sign")}


def c_tex(z):
    z = complex(np.round(z, 2))
    if abs(z.imag) < 0.005:
        return f"{z.real:.2f}"
    if abs(z.real) < 0.005:
        return f"{z.imag:.2f}i"
    return f"({z.real:.2f} {'+' if z.imag >= 0 else '-'} {abs(z.imag):.2f}i)"


def qubit_tex(v):
    return rf"{c_tex(v[0])}\,|0\rangle + {c_tex(v[1])}\,|1\rangle"


# ------------------------------------------------------------------ plots
def bloch_fig(vec, title, colour):
    u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
    fig = go.Figure()
    fig.add_surface(x=np.cos(u) * np.sin(v), y=np.sin(u) * np.sin(v), z=np.cos(v),
                    opacity=0.12, showscale=False, colorscale=[[0, TEAL], [1, TEAL]], hoverinfo="skip")
    for a in ((-1, 1, 0, 0, 0, 0), (0, 0, -1, 1, 0, 0), (0, 0, 0, 0, -1, 1)):
        fig.add_scatter3d(x=a[0:2], y=a[2:4], z=a[4:6], mode="lines", line=dict(color=GREY, width=2),
                          hoverinfo="skip", showlegend=False)
    L = float(np.linalg.norm(vec))
    if L > 0.02:
        fig.add_scatter3d(x=[0, vec[0]], y=[0, vec[1]], z=[0, vec[2]], mode="lines",
                          line=dict(color=colour, width=9), showlegend=False, hoverinfo="skip")
        fig.add_cone(x=[vec[0]], y=[vec[1]], z=[vec[2]], u=[vec[0] * .3], v=[vec[1] * .3], w=[vec[2] * .3],
                     colorscale=[[0, colour], [1, colour]], showscale=False, sizemode="absolute", sizeref=0.25,
                     anchor="tip", hoverinfo="skip")
    fig.add_scatter3d(x=[0, 0], y=[0, 0], z=[1.25, -1.3], mode="text", text=["|0⟩", "|1⟩"],
                      textfont=dict(size=16, color=INK), showlegend=False, hoverinfo="skip")
    state = "definite state" if L > 0.98 else ("completely random" if L < 0.02 else "partly random")
    fig.update_layout(
        title=dict(text=f"<b>{title}</b><br><span style='font-size:13px;color:{MUTED}'>arrow length {L:.2f} — {state}</span>",
                   x=0.5, xanchor="center", font=dict(size=15, color=INK)),
        height=330, margin=dict(l=0, r=0, t=60, b=0), paper_bgcolor=BG,
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
                   aspectmode="cube", camera=dict(eye=dict(x=1.3, y=1.3, z=0.9)), bgcolor=BG))
    return fig


def bars_fig(labels, probs, colours, title, xtitle):
    fig = go.Figure(go.Bar(x=labels, y=probs, marker_color=colours,
                           text=[f"{p*100:.0f}%" if p > 0.005 else "" for p in probs], textposition="outside"))
    fig.update_layout(title=dict(text=title, font=dict(size=15, color=INK)), height=330,
                      yaxis=dict(range=[0, 1.12], title="chance", tickformat=".0%", gridcolor="#E2DDD2"),
                      xaxis=dict(title=xtitle, type="category"), paper_bgcolor=BG, plot_bgcolor=BG,
                      margin=dict(l=10, r=10, t=50, b=10), font=dict(color=INK))
    return fig


# ------------------------------------------------------------------ session state
def psi_from(theta_deg, phi_deg):
    th, ph = np.radians(theta_deg), np.radians(phi_deg)
    return np.array([np.cos(th / 2), np.exp(1j * ph) * np.sin(th / 2)])


ss = st.session_state
ss.setdefault("theta", 60.0)
ss.setdefault("phi", 45.0)
ss.setdefault("rng", np.random.default_rng())


def reset():
    ss.psi0 = psi_from(ss.theta, ss.phi)
    ss.state = np.kron(ss.psi0, BELL00)
    ss.step = 0
    ss.bits = None
    ss.bob_before = None


def randomise():
    ss.theta = float(np.degrees(np.arccos(ss.rng.uniform(-1, 1))))
    ss.phi = float(ss.rng.uniform(0, 360))
    reset()


def next_step():
    if ss.step == 0:
        ss.state = apply_cnot(ss.state, 3, 0, 1)
    elif ss.step == 1:
        ss.state = apply_1q(ss.state, 3, 0, H)
    elif ss.step == 2:
        ss.bits, ss.state = measure(ss.state, 3, [0, 1], ss.rng)
        ss.bob_before = bob_qubit(ss.state, *ss.bits)
    elif ss.step == 3:
        ss.state = apply_1q(ss.state, 3, 2, FIX[ss.bits][1])
    else:
        reset()
        return
    ss.step += 1


if "step" not in ss:
    reset()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Your qubit ψ")
    st.slider("θ — how much |1⟩ is blended in", 0.0, 180.0, key="theta", step=1.0, on_change=reset, format="%.0f°")
    st.slider("φ — phase", 0.0, 360.0, key="phi", step=1.0, on_change=reset, format="%.0f°")
    st.button("🎲 Random ψ", on_click=randomise, width="stretch")
    psi = psi_from(ss.theta, ss.phi)
    st.latex(r"|\psi\rangle = \cos\frac{\theta}{2}\,|0\rangle + e^{i\varphi}\sin\frac{\theta}{2}\,|1\rangle")
    st.latex(r"\lambda = " + c_tex(psi[0]))
    st.latex(r"\mu = " + c_tex(psi[1]))
    st.markdown(f"<p class='small-note'>Chance of 0: <b>{abs(psi[0])**2*100:.0f}%</b> · "
                f"chance of 1: <b>{abs(psi[1])**2*100:.0f}%</b></p>", unsafe_allow_html=True)
    st.divider()
    st.markdown("<p class='small-note'>Short arrow on a sphere = that qubit, on its own, looks random.<br>"
                "Full-length arrow = a definite state.</p>", unsafe_allow_html=True)

st.title("🌀 Quantum Teleportation — try it yourself")
tab1, tab2, tab3 = st.tabs(["Teleport step by step", "Run it 1000 times", "Superdense coding"])

# ------------------------------------------------------------------ tab 1
STEP_INFO = [
    ("Step 0 — Start",
     "Alice holds ψ (qubit 1) and her half of an entangled pair (qubit 2). Bob holds the other half (qubit 3). "
     "Look at Bob's sphere: the arrow has length 0 — on its own, his qubit is pure randomness."),
    ("Step 1 — CNOT",
     "Alice flips qubit 2 whenever qubit 1 is |1⟩. Now ψ is tangled up with the pair, "
     "so even Alice's own qubit 1 starts to look random."),
    ("Step 2 — Hadamard",
     "Alice applies H to qubit 1. The state splits into four equal parts — one for each result "
     "Alice could measure. Bob's qubit still looks completely random."),
    ("Step 3 — Alice measures",
     "Alice measures qubits 1 and 2 and gets two ordinary bits. The original ψ is gone. "
     "She sends the two bits to Bob by phone or internet — no faster than light."),
    ("Step 4 — Bob's fix",
     "Bob uses the two bits to pick I, X, Z or iY. His qubit is now exactly the original ψ. Teleportation complete! 🎉"),
]
BTN = ["Apply CNOT ▶", "Apply Hadamard ▶", "Alice measures ▶", "Bob applies the fix ▶", "Start again ↺"]

with tab1:
    st.progress(ss.step / 4, text=f"Step {ss.step} of 4")
    c1, c2, _ = st.columns([1.2, 0.8, 3])
    c1.button(BTN[ss.step], on_click=next_step, type="primary", width="stretch")
    c2.button("Reset", on_click=reset, width="stretch")

    title, text = STEP_INFO[ss.step]
    st.markdown(f"<div class='step-box'><b>{title}.</b> {text}</div>", unsafe_allow_html=True)

    names = ["Qubit 1 — Alice’s ψ", "Qubit 2 — Alice’s half", "Qubit 3 — Bob’s half"]
    cols = [ORANGE, TEAL, ORANGE if ss.step == 4 else TEAL]
    bc = st.columns(3)
    for k in range(3):
        bc[k].plotly_chart(bloch_fig(bloch_vector(reduced(ss.state, 3, k)), names[k], cols[k]),
                           width="stretch", config={"displayModeBar": False}, key=f"bloch{k}")

    left, right = st.columns([1.1, 1])
    probs = np.abs(ss.state) ** 2
    colours = [ORANGE if (ss.bits is not None and p > 1e-9) else (GREY if ss.bits is not None else TEAL) for p in probs]
    left.plotly_chart(bars_fig([format(i, "03b") for i in range(8)], probs, colours,
                               "Chance of each 3-digit result", "digits: qubit 1 · qubit 2 · qubit 3"),
                      width="stretch", config={"displayModeBar": False}, key="bars")

    with right:
        st.subheader("What’s happening")
        st.latex(r"\text{Original: } |\psi\rangle = " + qubit_tex(ss.psi0))
        if ss.step == 0:
            st.latex(r"|\Psi\rangle = |\psi\rangle \otimes \frac{1}{\sqrt{2}}\left(|00\rangle + |11\rangle\right)")
        if ss.step == 1:
            st.latex(r"\text{CNOT}|\Psi\rangle = \frac{\lambda}{\sqrt{2}}(|000\rangle + |011\rangle) + "
                     r"\frac{\mu}{\sqrt{2}}(|110\rangle + |101\rangle)")
        if ss.step == 2:
            st.latex(r"H\,\text{CNOT}|\Psi\rangle = \frac{1}{2}\Big[\,|00\rangle(\lambda|0\rangle+\mu|1\rangle)"
                     r" + |01\rangle(\mu|0\rangle+\lambda|1\rangle) \\ + |10\rangle(\lambda|0\rangle-\mu|1\rangle)"
                     r" + |11\rangle(-\mu|0\rangle+\lambda|1\rangle)\Big]")
        if ss.bits is not None:
            name, _, what = FIX[ss.bits]
            st.markdown(f"**Alice measured:** `{ss.bits[0]}{ss.bits[1]}` → Bob must apply **{name}** ({what})")
            st.latex(r"\text{Bob before the fix: } " + qubit_tex(ss.bob_before))
        if ss.step == 4:
            bob = bob_qubit(ss.state, *ss.bits)
            st.latex(r"\text{Bob after the fix: } " + qubit_tex(bob))
            st.success(f"Match with the original (fidelity): {fidelity(ss.psi0, bob)*100:.1f}%")

    with st.expander("Bob’s fix table"):
        st.markdown("""
| Alice measures | Chance | Bob’s qubit is now | Bob applies |
|---|---|---|---|
| 00 | 25% | λ\\|0⟩ + µ\\|1⟩ | I — do nothing |
| 01 | 25% | µ\\|0⟩ + λ\\|1⟩ | X — swap 0 and 1 |
| 10 | 25% | λ\\|0⟩ − µ\\|1⟩ | Z — flip the sign of \\|1⟩ |
| 11 | 25% | −µ\\|0⟩ + λ\\|1⟩ | iY — swap, then fix the sign |
""")

# ------------------------------------------------------------------ tab 2
with tab2:
    st.markdown("Teleport the **same ψ** many times. Alice’s results should be pure chance (≈25% each), "
                "and Bob should end up with ψ every single time.")
    n = st.select_slider("How many runs?", options=[100, 500, 1000, 5000], value=1000)
    if st.button("Run ▶", type="primary"):
        psi0 = psi_from(ss.theta, ss.phi)
        counts = {k: 0 for k in FIX}
        fids = []
        for _ in range(n):
            s = apply_1q(apply_cnot(np.kron(psi0, BELL00), 3, 0, 1), 3, 0, H)
            bits, s = measure(s, 3, [0, 1], ss.rng)
            s = apply_1q(s, 3, 2, FIX[bits][1])
            counts[bits] += 1
            fids.append(fidelity(psi0, bob_qubit(s, *bits)))
        ss.many = (n, counts, np.array(fids))
    if "many" in ss:
        n_, counts, fids = ss.many
        m1, m2, m3 = st.columns(3)
        m1.metric("Runs", n_)
        m2.metric("Bob got ψ exactly", f"{np.mean(fids > 0.9999)*100:.0f}%")
        m3.metric("Average fidelity", f"{fids.mean()*100:.2f}%")
        labs = ["00", "01", "10", "11"]
        vals = [counts[(int(l[0]), int(l[1]))] / n_ for l in labs]
        fig = bars_fig(labs, vals, [TEAL] * 4, "What Alice measured", "Alice’s two bits")
        fig.add_hline(y=0.25, line_dash="dash", line_color=ORANGE, annotation_text="expected 25%")
        fig.update_yaxes(range=[0, 0.5])
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key="many_bars")
        st.info("Alice’s bits are random, so on their own they reveal nothing about λ or µ. "
                "Only together with the entangled pair do they rebuild ψ at Bob’s side.")

# ------------------------------------------------------------------ tab 3
SD_OPS = {"00": ("I", I2), "01": ("Z", Z), "10": ("X", X), "11": ("iY", 1j * Y)}
BELL_TEX = {"00": r"|\beta_{00}\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)",
            "01": r"|\beta_{01}\rangle = \frac{1}{\sqrt{2}}(|00\rangle - |11\rangle)",
            "10": r"|\beta_{10}\rangle = \frac{1}{\sqrt{2}}(|10\rangle + |01\rangle)",
            "11": r"|\beta_{11}\rangle = \frac{1}{\sqrt{2}}(|01\rangle - |10\rangle)"}

with tab3:
    st.markdown("Alice and Bob share the pair |β₀₀⟩. Alice encodes **2 bits** by doing one of four things "
                "to her half — then sends Bob just **1 qubit**.")
    a, b = st.columns([1, 1.4])
    with a:
        msg = st.radio("Message Alice wants to send", list(SD_OPS), horizontal=True)
        name, U = SD_OPS[msg]
        s = apply_1q(BELL00.copy(), 2, 0, U)
        st.markdown(f"**1. Alice applies:** `{name}`")
        st.markdown("**2. The pair becomes:**")
        st.latex(BELL_TEX[msg])
        st.markdown("**3. Alice sends her ONE qubit to Bob.**")
        s = apply_1q(apply_cnot(s, 2, 0, 1), 2, 0, H)          # Bell measurement = CNOT + H, then measure
        bits, _ = measure(s, 2, [0, 1], ss.rng)
        read = f"{bits[1]}{bits[0]}"                           # same bit order as the β labels
        st.markdown(f"**4. Bob measures both qubits and reads:** `{read}`")
        (st.success if read == msg else st.error)("Correct — 2 bits arrived inside 1 qubit!" if read == msg else "Mismatch")
    with b:
        p = np.abs(s) ** 2
        probs = [p[0], p[2], p[1], p[3]]
        st.plotly_chart(bars_fig(["00", "01", "10", "11"], probs, [ORANGE if q > 0.5 else GREY for q in probs],
                                 "What Bob’s measurement can give", "Bob reads"),
                        width="stretch", config={"displayModeBar": False}, key="sd_bars")

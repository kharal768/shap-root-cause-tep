"""Second process: a non-isothermal jacketed CSTR (A -> B) with PI temperature and level control,
used as a second process whose faults have a disturbed variable known by construction (Section 4.7).
Reactor model after the CSTR example of Seborg et al. (2011), with a jacket energy balance added. One continuous run in which normal operation and 11 fault types alternate
in random order with random episode lengths, mirroring the structure of the TEP datasets.
Channels (11): F, Tf, CA, CB, T (measured), Tj, Tci, Fc (actual coolant flow), h (measured level), Fout, Fc_cmd (controller output)."""
import numpy as np, json, os
rng = np.random.default_rng(2026)
NAMES = ["F", "Tf", "CA", "CB", "T", "Tj", "Tci", "Fc", "h", "Fout", "Fc_cmd"]
# parameters (L, min, K, mol, J)
k0, ER, dH, rhoCp, UA0, A, Vj, rhoCpj = 7.2e10, 8750.0, -2.0e4, 239.0, 5.0e4, 1.0, 100.0, 239.0
F0, CAf0, Tf0, Tci0, Tsp, Vsp = 100.0, 1.0, 350.0, 280.0, 350.0, 100.0
DT, SUB = 0.025, 20            # integration step (min) and sub-steps per sample -> one sample every 0.5 min
GT = {1: ["CA"], 2: ["Tf"], 3: ["Tci"], 4: ["Tci"], 5: ["F"], 6: ["CA"], 7: ["Fc", "Fc_cmd"], 8: ["Tj", "Fc"],
      9: ["CA", "CB"], 10: ["T"], 11: ["h"]}
DIRECT = {2, 3, 4, 5, 7, 10, 11}   # disturbed variable itself measured; 1, 6, 8, 9: proxy channels
DESC = {1: "feed concentration CAf, step (unmeasured)", 2: "feed temperature Tf, step", 3: "coolant inlet temperature, step",
        4: "coolant inlet temperature, random variation", 5: "feed flow, step", 6: "feed concentration CAf, random variation (unmeasured)",
        7: "coolant valve sticking", 8: "jacket fouling, UA drift (unmeasured)", 9: "catalyst deactivation, k0 drift (unmeasured)",
        10: "reactor temperature sensor bias", 11: "level sensor drift"}

def ou(x, sig, tau, dt):  # Ornstein-Uhlenbeck increment
    return x - x / tau * dt + sig * np.sqrt(2 * dt / tau) * rng.standard_normal()

def simulate():
    # episode schedule: normal and fault episodes alternate; each fault 6 times
    faults = [f for f in GT for _ in range(6)]; rng.shuffle(faults)
    sched = []
    for f in faults:
        sched.append((0, int(rng.integers(200, 600)))); sched.append((f, int(rng.integers(200, 700))))
    sched.append((0, 400))
    V, CA, CB, T, Tj = Vsp, 0.5, 0.5, Tsp, 330.0
    Fc, Fc_act, IT, IL, Fout = 83.7, 83.7, 0.0, 0.0, F0
    dCAf = dTf = dTci = dF = rCAf = rTci = 0.0
    X, Y = [], []
    for f, n in sched:
        t_in, hdrift, stick = 0.0, 0.0, Fc_act
        for _ in range(n):
            for _ in range(SUB):
                t_in += DT
                dCAf = ou(dCAf, 0.01, 30, DT); dTf = ou(dTf, 0.5, 30, DT); dTci = ou(dTci, 0.5, 30, DT); dF = ou(dF, 1.0, 20, DT)
                rCAf = ou(rCAf, 0.08 if f == 6 else 0.0, 10, DT); rTci = ou(rTci, 4.0 if f == 4 else 0.0, 10, DT)
                CAf = CAf0 + dCAf + (0.1 if f == 1 else 0) + rCAf
                Tf = Tf0 + dTf + (4.0 if f == 2 else 0)
                Tci = Tci0 + dTci + (4.0 if f == 3 else 0) + rTci
                F = F0 * (1 + dF / 100) * (1.08 if f == 5 else 1)
                UA = UA0 * (1 - 0.4 * min(t_in / 300, 1) if f == 8 else 1)
                kk = k0 * (1 - 0.3 * min(t_in / 300, 1) if f == 9 else 1) * np.exp(-ER / T)
                # measured values seen by controllers
                Tm = T + (1.5 if f == 10 else 0)
                if f == 11: hdrift = min(hdrift + 0.0005 * DT, 0.15)
                hm = V / A / 100 + hdrift
                eT = Tm - Tsp; IT += eT * DT; Fc = np.clip(83.7 + 25 * eT + 1 * IT, 0, 600)
                eL = hm - Vsp / A / 100; IL += eL * DT; Fout = np.clip(F0 + 300 * eL + 30 * IL, 0, 400)
                if f == 7:   # sticking: valve moves only when the command differs by more than 8 L/min
                    if abs(Fc - stick) > 8: stick = Fc
                    Fc_act = stick
                else: Fc_act = Fc
                r = kk * CA
                V += (F - Fout) * DT
                CA += (F / V * (CAf - CA) - r) * DT
                CB += (-F / V * CB + r) * DT
                T += (F / V * (Tf - T) + (-dH) / rhoCp * r - UA / (rhoCp * V) * (T - Tj)) * DT
                Tj += (Fc_act / Vj * (Tci - Tj) + UA / (rhoCpj * Vj) * (T - Tj)) * DT
            meas = np.array([F, Tf, CA, CB, Tm, Tj, Tci, Fc_act, hm, Fout, Fc])
            noise = np.array([0.5, 0.1, 0.005, 0.005, 0.1, 0.1, 0.1, 0.5, 0.002, 0.5, 0.5])
            X.append(meas + noise * rng.standard_normal(11)); Y.append(f)
    return np.array(X), np.array(Y)

if __name__ == "__main__":
    X, y = simulate()
    print(X.shape, np.bincount(y), np.isfinite(X).all(), X.mean(0).round(2), X.std(0).round(3))
    os.makedirs("../data", exist_ok=True); np.savez("../data/reactor.npz", X=X, y=y, names=np.array(NAMES))
    json.dump({"names": NAMES, "gt": GT, "direct": sorted(DIRECT), "desc": DESC}, open("../data/meta.json", "w"), indent=1)

"""Tennessee Eastman Process channel names, evaluated faults and documented root-cause channels
(Table 2 of the paper). Imported by the analysis scripts."""
# XMEAS(1..41) then XMV(1..11), 0-indexed to match the data array's column order
XMEAS_NAMES = [f"XMEAS({i})" for i in range(1, 42)]
XMV_NAMES = [f"XMV({i})" for i in range(1, 12)]
ALL_CHANNEL_NAMES = XMEAS_NAMES + XMV_NAMES  # 52 total, index 0-51

CHANNEL_DESCRIPTIONS = {
    "XMEAS(1)": "A feed (stream 1)",
    "XMEAS(2)": "D feed (stream 2)",
    "XMEAS(3)": "E feed (stream 3)",
    "XMEAS(4)": "A and C feed (stream 4)",
    "XMEAS(7)": "Reactor pressure",
    "XMEAS(8)": "Reactor level",
    "XMEAS(9)": "Reactor temperature",
    "XMEAS(21)": "Reactor cooling water outlet temperature",
    "XMEAS(22)": "Condenser/separator cooling water outlet temperature",
    "XMV(3)": "A feed flow valve (stream 1)",
    "XMV(4)": "A and C feed flow valve (stream 4)",
    "XMV(10)": "Reactor cooling water flow valve",
    "XMV(11)": "Condenser cooling water flow valve",
}

# Fault type/description table (all 20 faults, for the classification task and reporting)
FAULT_INFO = {
    0:  {"desc": "Normal operation", "type": "N/A"},
    1:  {"desc": "A/C feed ratio, B composition constant (stream 4)", "type": "Step"},
    2:  {"desc": "B composition, A/C ratio constant (stream 4)", "type": "Step"},
    3:  {"desc": "D feed temperature (stream 2)", "type": "Step"},
    4:  {"desc": "Reactor cooling water inlet temperature", "type": "Step"},
    5:  {"desc": "Condenser cooling water inlet temperature", "type": "Step"},
    6:  {"desc": "A feed loss (stream 1)", "type": "Step"},
    7:  {"desc": "C header pressure loss, reduced availability (stream 4)", "type": "Step"},
    8:  {"desc": "A, B, C feed composition (stream 4)", "type": "Random variation"},
    9:  {"desc": "D feed temperature (stream 2)", "type": "Random variation"},
    10: {"desc": "C feed temperature (stream 4)", "type": "Random variation"},
    11: {"desc": "Reactor cooling water inlet temperature", "type": "Random variation"},
    12: {"desc": "Condenser cooling water inlet temperature", "type": "Random variation"},
    13: {"desc": "Reaction kinetics", "type": "Slow drift"},
    14: {"desc": "Reactor cooling water valve", "type": "Sticking"},
    15: {"desc": "Condenser cooling water valve", "type": "Sticking"},
    16: {"desc": "Unknown", "type": "Unknown"},
    17: {"desc": "Unknown", "type": "Unknown"},
    18: {"desc": "Unknown", "type": "Unknown"},
    19: {"desc": "Unknown", "type": "Unknown"},
    20: {"desc": "Unknown", "type": "Unknown"},
}

# Faults excluded from physical root-cause validation, and why
EXCLUDED_UNOBSERVABLE = [3, 9, 15]   # documented as producing no measurable signature
EXCLUDED_UNKNOWN = [16, 17, 18, 19, 20]  # no consensus root-cause variable exists

# Ground-truth implicated channel(s) for the 12 faults WITH a documented,
# consensus root cause. Listed as a set since some faults implicate more
# than one directly-related channel (e.g. the measured variable AND the
# controller's compensating manipulated variable).
GROUND_TRUTH_CHANNELS = {
    1:  {"XMEAS(4)", "XMV(4)"},
    2:  {"XMEAS(4)", "XMV(4)"},
    4:  {"XMEAS(21)", "XMV(10)"},
    5:  {"XMEAS(22)", "XMV(11)"},
    6:  {"XMEAS(1)", "XMV(3)"},
    7:  {"XMEAS(4)", "XMV(4)"},
    8:  {"XMEAS(4)", "XMV(4)"},
    10: {"XMEAS(4)", "XMV(4)"},
    11: {"XMEAS(21)", "XMV(10)"},
    12: {"XMEAS(22)", "XMV(11)"},
    13: {"XMEAS(7)", "XMEAS(8)", "XMEAS(9)"},  # diffuse: internal kinetics, no single channel
    14: {"XMV(10)"},  # cleanest case: the sticking valve's own command channel
}

# Faults sharing overlapping ground-truth channels with at least one other
# fault (limits channel-level discriminability between these specific pairs/groups)
OVERLAPPING_GROUPS = [
    {1, 2, 7, 8, 10},  # all implicate stream-4 region
    {4, 11},           # same channel, step vs. random variation
    {5, 12},           # same channel, step vs. random variation
]

VALIDATABLE_FAULTS = sorted(GROUND_TRUTH_CHANNELS.keys())

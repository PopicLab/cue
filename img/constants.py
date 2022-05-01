from enum import Enum

# BAM types
BAM_TYPE = Enum("BAM_TYPE", 'LONG '
                            'LINKED '
                            'SHORT ')

# Signal types
class SVSignals(str, Enum):
    RD = "RD"
    RD_LOW = "RD_LOW"
    RD_CLIPPED = "RD_CLIPPED"
    SR_RP = "SR_RP"
    LLRR = "LLRR"
    RL = "RL"
    SM = "SM"
    LLRR_VS_LR = "LLRR_VS_LR"

class RefSignals(str, Enum):
    RD = "REF_RD"
    RD_LOW = "REF_RD_LOW"
    RD_CLIPPED = "REF_RD_CLIPPED"
    SR_RP = "REF_SR_RP"
    LLRR = "REF_LLRR"
    RL = "REF_RL"
    SM = "REF_SM"


SV_SIGNAL_SCALAR = {SVSignals.RD, SVSignals.RD_LOW, SVSignals.RD_CLIPPED,
                    RefSignals.RD, RefSignals.RD_LOW, RefSignals.RD_CLIPPED}
SV_SIGNAL_PAIRED = {SVSignals.LLRR, SVSignals.RL}

def to_ref_signal(signal):
    mapping = {SVSignals.RD: RefSignals.RD,
               SVSignals.RD_LOW: RefSignals.RD_LOW,
               SVSignals.RD_CLIPPED: RefSignals.RD_CLIPPED,
               SVSignals.SR_RP: RefSignals.SR_RP,
               SVSignals.LLRR: RefSignals.LLRR,
               SVSignals.RL: RefSignals.RL,
               SVSignals.SM: RefSignals.SM}
    return mapping[signal]


SV_SIGNAL_SET = Enum("SV_SIGNAL_SET", 'SHORT '
                                      'SHORT3 '
                                      'LONG '
                                      'LINKED '
                                      'EMPTY ')

SV_SIGNALS_BY_TYPE = {
    SV_SIGNAL_SET.SHORT: [SVSignals.RD, SVSignals.RD_LOW, SVSignals.SR_RP, SVSignals.LLRR, SVSignals.RL,
                          SVSignals.LLRR_VS_LR],
    SV_SIGNAL_SET.SHORT3: [SVSignals.RD, SVSignals.RD_LOW, SVSignals.SR_RP],
    SV_SIGNAL_SET.LINKED: [SVSignals.SM, SVSignals.RD_LOW, SVSignals.SR_RP, SVSignals.LLRR, SVSignals.RL],
    SV_SIGNAL_SET.LONG: [SVSignals.SM, SVSignals.RD_LOW, SVSignals.SR_RP, SVSignals.RD_CLIPPED],
    SV_SIGNAL_SET.EMPTY: []
    }

SV_SIGNAL_SET_CHANNEL_IDX = {
    SV_SIGNAL_SET.SHORT: {SV_SIGNAL_SET.SHORT: range(len(SV_SIGNALS_BY_TYPE[SV_SIGNAL_SET.SHORT])),
                          SV_SIGNAL_SET.SHORT3: [0, 1, 2]},
    SV_SIGNAL_SET.SHORT3: {SV_SIGNAL_SET.SHORT3: range(len(SV_SIGNALS_BY_TYPE[SV_SIGNAL_SET.SHORT3]))},
    SV_SIGNAL_SET.LONG: {SV_SIGNAL_SET.LONG: range(len(SV_SIGNALS_BY_TYPE[SV_SIGNAL_SET.LONG]))},
    SV_SIGNAL_SET.LINKED: {SV_SIGNAL_SET.LINKED: range(len(SV_SIGNALS_BY_TYPE[SV_SIGNAL_SET.LINKED]))}
    }

SV_SIGNAL_RP_TYPE = Enum("SV_SIGNAL_RP_TYPE", 'LLRR '
                                              'RL '
                                              'LR ')

# Image classes
SV_CLASS_SET = Enum("SV_CLASS_SET", 'BASIC4 '
                                    'BASIC4ZYG '
                                    'BASIC5 '
                                    'BASIC5ZYG '
                                    'BINARY')

SV_CLASSES = {SV_CLASS_SET.BASIC4: ["NEG", "DEL", "INV", "DUP"],
              SV_CLASS_SET.BASIC5: ["NEG", "DEL", "INV", "DUP", "IDUP"],
              SV_CLASS_SET.BASIC4ZYG: ["NEG", "DEL-HOM", "INV-HOM", "DUP-HOM", "DEL-HET", "INV-HET", "DUP-HET"],
              SV_CLASS_SET.BASIC5ZYG: ["NEG", "DEL-HOM", "INV-HOM", "DUP-HOM", "DEL-HET", "INV-HET", "DUP-HET",
                                       "IDUP-HOM", "IDUP-HET"],
              SV_CLASS_SET.BINARY: ["NEG", "POS"]}

CLASS_BACKGROUND = "NEG"
CLASS_SV = "POS"

SV_ZYGOSITY_SETS = {SV_CLASS_SET.BASIC4ZYG, SV_CLASS_SET.BASIC5ZYG}

SV_LABELS = {"NEG": 0, "POS": 1, "DEL": 1, "INV": 2, "DUP": 3,
             "DEL-HOM": 1, "INV-HOM": 2, "DUP-HOM": 3,
             "DEL-HET": 4, "INV-HET": 5, "DUP-HET": 6,
             "IDUP": 7, "IDUP-HOM": 7, "IDUP-HET": 8}

LABEL_BACKGROUND = 0
LABEL_SV = 1
LABEL_LANDMARK_DEFAULT = 0
KP_VISIBLE = 1
KP_FILTERED = -1

class TargetType(str, Enum):
    boxes = "boxes"
    keypoints = "keypoints"
    labels = "labels"
    classes = "classes"
    image_id = "image_id"
    area = "area"
    heatmaps = "heatmaps"
    weight = "weight"
    scores = "scores"
    gloc = "gloc"
    dataset_id = "dataset_id"

class ZYGOSITY(str, Enum):
    HET = "HET"
    HOM = "HOM"
    UNK = "UNK"
    HOMREF = "HOMREF"


ZYGOSITY_ENCODING_SIM = {"homAB": ZYGOSITY.HOM, "hetA": ZYGOSITY.HET, "hetB": ZYGOSITY.HET, "UNK": ZYGOSITY.UNK}
ZYGOSITY_ENCODING = {(0, 1): ZYGOSITY.HET, (1, 1): ZYGOSITY.HOM, (1, 0): ZYGOSITY.HET, (0, 0): ZYGOSITY.HOMREF,
                     (None, None): ZYGOSITY.UNK}
ZYGOSITY_ENCODING_BED = {"0/1": ZYGOSITY.HET, "1/1": ZYGOSITY.HOM, "1/0": ZYGOSITY.HET, "./.": ZYGOSITY.UNK}
ZYGOSITY_GT_BED = {ZYGOSITY.HOM: "1/1", ZYGOSITY.HET: "0/1", ZYGOSITY.UNK: "./."}
ZYGOSITY_GT_VCF = {ZYGOSITY.HOMREF: (0, 0), ZYGOSITY.HOM: (1, 1), ZYGOSITY.HET: (0, 1), ZYGOSITY.UNK: (None, None)}

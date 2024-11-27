from dataclasses import dataclass


@dataclass
class filesystem:
    AUX_ML2 = "/net/expertise2/HsWindSea_study/aux_ml2_test/S1A_AUX_ML2_V20140406T133000_G20240429T091835.SAFE"
    OUTPUT_FOLDER = ".tmp"


@dataclass
class INFERENCE:
    VERBOSITY = 0
    BATCH_SIZE = 40
    CLIP_MIN = 0.0

class CONSTANTS :
    IMACS_NAMES = ['imacs_15_20',
                   'imacs_20_27',
                   'imacs_27_38',
                   'imacs_38_52',
                   'imacs_52_74',
                   'imacs_74_106',
                   'imacs_106_163',
                   'imacs_163_290']
keypoints_list = [
    'RWRL',
    'RWRM',
    'RELL',
    'RELM',
    'RSHF',
    'RSHB',
    'LWRL',
    'LWRM',
    'LELL',
    'LELM',
    'LSHF',
    'LSHB',
    'RANL',
    'RANM',
    'RKNL',
    'RKNM',
    'RTRO',
    'LANL',
    'LANM',
    'LKNL',
    'LKNM',
    'LTRO',
    'REAR',
    'LEAR'
]

keypoints_connections = {
    "FRONT_SHOULDER": [10, 4],           # LSHF, RSHF
    "BACK_SHOULDER": [11, 5],            # LSHB, RSHB
    "RIGHT_SHOULDER": [4, 5],            # RSHF, RSHB
    "LEFT_SHOULDER": [10, 11],           # LSHF, LSHB
    "LEFT_UPPER_ARM_OUT": [11, 8],       # LSHB, LELL
    "LEFT_UPPER_ARM_IN": [10, 9],        # LSHF, LELM
    "RIGHT_UPPER_ARM_OUT": [5, 2],       # RSHB, RELL
    "RIGHT_UPPER_ARM_IN": [4, 3],        # RSHF, RELM
    "LEFT_FOREARM_IN": [9, 7],           # LELM, LWRM
    "LEFT_FOREARM_OUT": [8, 6],          # LELL, LWRL
    "RIGHT_WRIST": [1, 0],               # RWRM, RWRL
    "LEFT_WRIST": [7, 6],                # LWRM, LWRL
    "RIGHT_FOREARM_IN": [3, 1],          # RELM, RWRM
    "RIGHT_FOREARM_OUT": [2, 0],         # RELL, RWRL
    "RIGHT_ELBOW": [3, 2],               # RELM, RELL
    "LEFT_ELBOW": [9, 8],                # LELM, LELL
    "LEFT_HIP": [23, 21],                # LEAR, LTRO
    "FRONT_HIP": [21, 16],               # LTRO, RTRO
    "BACK_HIP": [23, 22],                # LEAR, REAR
    "RIGHT_HIP": [22, 16],               # REAR, RTRO
    "RIGHT_KNEE_OUT": [22, 14],          # REAR, RKNL
    "RIGHT_KNEE_IN": [22, 15],           # REAR, RKNM
    "LEFT_KNEE_OUT": [23, 19],           # LEAR, LKNL
    "LEFT_KNEE_IN": [23, 20],            # LEAR, LKNM
    "LEFT_BODY_FRONT": [10, 21],         # LSHF, LTRO
    "RIGHT_BODY_FRONT": [4, 16],         # RSHF, RTRO
    "LEFT_BODY_BACK": [11, 23],          # LSHB, LEAR
    "RIGHT_BODY_BACK": [5, 22],          # RSHB, REAR
    "LEFT_KNEE": [20, 19],               # LKNM, LKNL
    "RIGHT_KNEE": [15, 14],              # RKNM, RKNL
    "LEFT_THIGH_OUT": [21, 19],          # LTRO, LKNL
    "RIGHT_THIGH_OUT": [16, 14],         # RTRO, RKNL
    "LEFT_THIGH_IN": [21, 20],           # LTRO, LKNM
    "RIGHT_THIGH_IN": [16, 15],          # RTRO, RKNM
    "RIGHT_LOWER_LEG_OUT": [14, 12],     # RKNL, RANL
    "LEFT_LOWER_LEG_OUT": [19, 17],      # LKNL, LANL
    "LEFT_LOWER_LEG_IN": [20, 18],       # LKNM, LANM
    "RIGHT_LOWER_LEG_IN": [15, 13],      # RKNM, RANM
    "RIGHT_ANKLE": [13, 12],             # RANM, RANL
    "LEFT_ANKLE": [18, 17]               # LANM, LANL
}

exp_keypoints_list = [
    "NOSE",
    "LEFT_EYE",
    "RIGHT_EYE",
    "LEFT_EAR",
    "RIGHT_EAR",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_ELBOW",
    "RIGHT_ELBOW",
    "LEFT_WRIST",
    "RIGHT_WRIST",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE"
]

exp_keypoints_connections = {
    "NOSE_LEFT_EYE": [0, 1],
    "NOSE_RIGHT_EYE": [0, 2],
    "LEFT_EYE_EAR": [1, 3],
    "RIGHT_EYE_EAR": [2, 4],
    "SHOULDER": [5, 6],
    "LEFT_UPPER_ARM": [5, 7],
    "RIGHT_UPPER_ARM": [6, 8],
    "LEFT_FORE_ARM": [7, 9],
    "RIGHT_FORE_ARM": [8, 10],
    "LEFT_TORSO": [5, 11],
    "RIGHT_TORSO": [6, 12],
    "GROIN": [11, 12],
    "LEFT_THIGH": [11, 13],
    "RIGHT_THIGH": [12, 14],
    "LEFT_LOWER_LEG": [13, 15],
    "RIGHT_LOWER_LEG": [14, 16],
}

keypoint_pairs_for_calc = {
    "LEFT_SHOULDER": ["LSHF", "LSHB"],
    "RIGHT_SHOULDER": ["RSHF", "RSHB"],
    "LEFT_ELBOW": ["LELM", "LELL"],
    "RIGHT_ELBOW": ["RELM", "RELL"],
    "LEFT_WRIST": ["LWRM", "LWRL"],
    "RIGHT_WRIST": ["RWRM", "RWRL"],
    "LEFT_HIP": ["LEAR", "LTRO"],
    "RIGHT_HIP": ["REAR", "RTRO"],
    "LEFT_KNEE": ["LKNM", "LKNL"],
    "RIGHT_KNEE": ["RKNM", "RKNL"],
    "LEFT_ANKLE": ["LANM", "LANL"],
    "RIGHT_ANKLE": ["RANM", "RANL"]
}

body_keypoints_list = [
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_ELBOW",
    "RIGHT_ELBOW",
    "LEFT_WRIST",
    "RIGHT_WRIST",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE"
]

body_keypoints_connections = {
    "SHOULDER": [5, 6],
    "LEFT_UPPER_ARM": [5, 7],
    "RIGHT_UPPER_ARM": [6, 8],
    "LEFT_FORE_ARM": [7, 9],
    "RIGHT_FORE_ARM": [8, 10],
    "LEFT_TORSO": [5, 11],
    "RIGHT_TORSO": [6, 12],
    "GROIN": [11, 12],
    "LEFT_THIGH": [11, 13],
    "RIGHT_THIGH": [12, 14],
    "LEFT_LOWER_LEG": [13, 15],
    "RIGHT_LOWER_LEG": [14, 16],
}

compare_keypoints_list = [
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_ELBOW",
    "RIGHT_ELBOW",
    "LEFT_WRIST",
    "RIGHT_WRIST",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE"
]

compare_keypoints_connections = {
    "SHOULDER": [0, 1],
    "LEFT_UPPER_ARM": [0, 2],
    "RIGHT_UPPER_ARM": [1, 3],
    "LEFT_FORE_ARM": [2, 4],
    "RIGHT_FORE_ARM": [3, 5],
    "LEFT_TORSO": [0, 6],
    "RIGHT_TORSO": [1, 7],
    "HIP": [6, 7],
    "LEFT_THIGH": [6, 8],
    "RIGHT_THIGH": [7, 9],
    "LEFT_LOWER_LEG": [8, 10],
    "RIGHT_LOWER_LEG": [9, 11],
}
import collections
import music21
from music21 import harmony
from reward.scales import PITCH_NAME_2_NUM, PITCH_NUM_2_NAME, SCALE_INTERVALS, get_scale_pitches

UNKNOWN_KEY = "UNK"
C2S_RULES = {
    # format - rule_name: (key_shift, candidate_scales)
    # candidate_scales format - [(relative_scale_root, scale_name)]
    "maj7_tonic":      (0, 
                        [(0,"Ionian"), (0,"blues_major"), (0,"pent_major")]),
    "maj7_subdom":     (7, 
                        [(0,"Lydian"), (0,"pent_major"), (7,"blues_major")]),
    "maj7_unk":        (UNKNOWN_KEY, 
                        [(0,"Ionian"), (0,"pent_major"), (0,"blues_major")]),
    "maj7_#11":        (UNKNOWN_KEY, 
                        [(0,"Lydian")]),
    "min9_tonic":      (3, 
                        [(0,"Aeolian"), (0,"blues_minor"), (0,"pent_minor")]),
    "min_subdom":      (-2%12, 
                        [(0,"Dorian"), (-1%12,"blues_major"), (7,"blues_minor")]),
    "min9_unk":        (UNKNOWN_KEY, 
                        [(0,"pent_minor"), (0,"Aeolian")]),
    "min69_tonic":     (3, 
                        [(0,"melodic_minor"), (0,"blues_minor"), (0,"pent_minor")]),
    "min69_unk":       (UNKNOWN_KEY, 
                        [(0,"melodic_minor")]),
    "min7_phri":       (-4%12, 
                        [(0,"Phrigian"), (0,"blues_minor")]),
    "min6_unk":        (UNKNOWN_KEY, 
                        [(0,"melodic_minor"), (0,"pent_minor")]),
    "minmaj":          (UNKNOWN_KEY, 
                        [(0,"melodic_minor"), (0,"harmonic_minor")]),
    "min_tonic":       (3, 
                        [(0,"Aeolian"), (0,"melodic_minor"), (0,"blues_minor"), (0,"pent_minor")]),
    "min_unk":         (UNKNOWN_KEY, 
                        [(0,"Aeolian"), (0,"melodic_minor"), (0,"pent_minor")]),
    "dom9":            (UNKNOWN_KEY, 
                        [(0,"Mixolydian"), (0,"blues_major")]),
    "dom7b13":         (UNKNOWN_KEY, 
                        [(0,"melodic_major"), (0,"alt"), (0,"blues_major")]),
    "dom7#11_maj_dom": (-7%12, 
                        [(0,"Lydian_dom"), (0,"blues_major")]),
    "dom7#11_min_dom": (-4%12, 
                        [(0,"Lydian_dom"), (0,"alt")]),
    "dom7#11_unk":     (UNKNOWN_KEY, 
                        [(0,"Lydian_dom")]),
    "dom_alt":         (UNKNOWN_KEY, 
                        [(0,"alt")]),
    "dom_sus":         (UNKNOWN_KEY, 
                        [(0,"Phrigidorian"), (0,"Phrigian")]),
    "dom7_trans":      (0, 
                        [(0,"Mixolydian"), (0,"blues_major"), (-7%12,"blues_major")]),
    "dom7_maj_dom":    (-7%12, 
                        [(0,"Mixolydian"), (0,"Lydian_dom"), (0,"blues_major")]),
    "dom7_min_dom":    (-4%12, 
                        [(0,"Phrigian#3"), (0,"alt")]),
    "dom7_dd":         (3, 
                        [(0,"alt")]),
    "dom7_unk":        (UNKNOWN_KEY, 
                        [(0,"Mixolydian"), (0,"alt")]),
    "maj_phri":        (-4%12, 
                        [(0,"Phrigian#3")]),
    "half_dim":        (UNKNOWN_KEY, 
                        [(0,"half_dim")]),
    "dim7_dom":        (1, 
                        [(0,"dim_1"), (0,"dim_2"), (5,"alt")]),
    "dim7_unk":        (UNKNOWN_KEY, 
                        [(0,"dim_1"), (0,"dim_2")]),
    "dim3_dom":        (1, 
                        [(0,"dim_1"), (0,"dim_2"), (0,"Locrian"), (-4%12,"Mixolydian"), (5,"alt")]),
    "aug_subdom":      (7, 
                        [(0,"Lydian_aug"), (0,"whole_tone")]),
    "aug_unk":         (UNKNOWN_KEY, 
                        [(0,"whole_tone")]),
}

def c2s_rule_dict(list_of_rules):
    rule_dict = {}
    for (key, scale_list) in list_of_rules:
        rule_dict[key] = scale_list
    return rule_dict

# CM
MAJ_C2S_RULES = c2s_rule_dict([
    C2S_RULES["maj7_tonic"],
    C2S_RULES["maj7_subdom"],
    C2S_RULES["dom7_maj_dom"],
    C2S_RULES["maj_phri"],
    C2S_RULES["maj7_unk"],
])
MAJ_CHORD_TYPE = ["major"]

# CM7, CM9, Cadd2, C6, C69, etc.
MAJ7_FAMILY_C2S_RULES = c2s_rule_dict([
    C2S_RULES["maj7_tonic"],
    C2S_RULES["maj7_subdom"],
    C2S_RULES["maj7_unk"],
])
MAJ7_FAMILY_CHORD_TYPE = ["major-seventh", "major-sixth", "major-11th", "major-13th", "suspended-second"]

# CM7#11, can include 9,13
MAJ7LYDIAN_C2S_RULES = c2s_rule_dict([
    C2S_RULES["maj7_#11"],
])
MAJ7LYDIAN_CHORD_TYPE = ["major-ninth"] # music21 does not have #11 chord options

# Cm
MIN_C2S_RULES = c2s_rule_dict([
    C2S_RULES["min_tonic"],
    C2S_RULES["min_subdom"],
    C2S_RULES["min7_phri"],
    C2S_RULES["min_unk"],
])
MIN_CHORD_TYPE = ["minor"]

# Cm9, Cm11, Cm13, etc.
MIN9_FAMILY_C2S_RULES = c2s_rule_dict([
    C2S_RULES["min9_tonic"],
    C2S_RULES["min_subdom"],
    C2S_RULES["min9_unk"],
])
MIN9_FAMILY_CHORD_TYPE = ["minor-ninth", "minor-11th", "minor-13th"]

# Cm69, etc. (with melodic minor third extensions higher or equal than 7th)
MIN69_FAMILY_C2S_RULES = c2s_rule_dict([
    C2S_RULES["min69_tonic"],
    C2S_RULES["min69_unk"],
])
MIN69_FAMILY_CHORD_TYPE = ["minor-sixth"] # music21 does not have m69 chord options

# Cm7
MIN7_C2S_RULES = c2s_rule_dict([
    C2S_RULES["min9_tonic"],
    C2S_RULES["min_subdom"],
    C2S_RULES["min7_phri"],
    C2S_RULES["min_unk"],
])
MIN7_CHORD_TYPE = ["minor-seventh"]

# Cm6
MIN6_C2S_RULES = c2s_rule_dict([
    C2S_RULES["min69_tonic"],
    C2S_RULES["min_subdom"],
    C2S_RULES["min7_phri"],
    C2S_RULES["min6_unk"],
])
MIN6_CHORD_TYPE = ["minor-sixth"]

# CmM7 (9,11,13)
MINMAJ_FAMILY_C2S_RULES = c2s_rule_dict([
    C2S_RULES["minmaj"],
])
MINMAJ_FAMILY_CHORD_TYPE = ["minor-major-seventh", "minor-major-ninth", "minor-major-11th", "minor-major-13th"]

# C7
DOM7_C2S_RULES = c2s_rule_dict([
    C2S_RULES["dom7_trans"],
    C2S_RULES["dom7_maj_dom"],
    C2S_RULES["dom7_min_dom"],
    C2S_RULES["dom7_dd"],
    C2S_RULES["dom7_unk"],
])
DOM7_CHORD_TYPE = ["dominant-seventh"]

# F/C, Csus9, C9, C11, C13
DOM9_FAMILY_C2S_RULES = c2s_rule_dict([
    C2S_RULES["dom9"],
])
DOM9_FAMILY_CHORD_TYPE = ["dominant-ninth", "dominant-11th", "dominant-13th"]

# C7b13
DOM7b13_FAMILY_C2S_RULES = c2s_rule_dict([
    C2S_RULES["dom7b13"],
])
DOM7b13_FAMILY_CHORD_TYPE = ["dominant-ninth", "dominant-11th"] # music21 does not have domb13 chord options

# C7#11
DOM7LYDIAN_C2S_RULES = c2s_rule_dict([
    C2S_RULES["dom7#11_maj_dom"],
    C2S_RULES["dom7#11_min_dom"],
    C2S_RULES["dom7#11_unk"],
])
DOM7LYDIAN_CHORD_TYPE = ["dominant-ninth"] # music21 does not have dom#11 chord options

# C7b9, Calt
DOM7ALT_FAMILY_C2S_RULES = c2s_rule_dict([
    C2S_RULES["dom_alt"],
])
DOM7ALT_FAMILY_CHORD_TYPE = ["dominant-seventh"] # music21 does not have domb9, alt chord options

# C7susb9, C7sus#9, C7sus13
DOM7SUS_FAMILY_C2S_RULES = c2s_rule_dict([
    C2S_RULES["dom_sus"],
])
DOM7SUS_FAMILY_CHORD_TYPE = ["suspended-fourth", "suspended-fourth-seventh"]

# Cm7b5 (9, 11, b13)
HALFDIM_FAMILY_C2S_RULES = c2s_rule_dict([
    C2S_RULES["half_dim"],
])
HALFDIM_FAMILY_CHORD_TYPE = ["half-diminished-seventh", "half-diminished-ninth", "half-diminished-minor-ninth",
                             "half-diminished-11th", "half-diminished-13th"]

# Cdim7
DIM7_FAMILY_C2S_RULES = c2s_rule_dict([
    C2S_RULES["dim7_dom"],
    C2S_RULES["dim7_unk"],
])
DIM7_FAMILY_CHORD_TYPE = ["diminished-seventh", "diminished-ninth", "diminished-minor-ninth", "diminished-11th"]

# Cdim
DIM3_C2S_RULES = c2s_rule_dict([
    C2S_RULES["dim3_dom"],
    C2S_RULES["dim7_unk"],
])
DIM3_CHORD_TYPE = ["diminished"]

# C+, C+7 (9, #11)
AUG_FAMILY_C2S_RULES = c2s_rule_dict([
    C2S_RULES["aug_subdom"],
    C2S_RULES["aug_unk"],
])
AUG_FAMILY_CHORD_TYPE = ["augmented", "augmented-major-seventh", "augmented-seventh", "augmented-major-ninth", 
                         "augmented-dominant-ninth", "augmented-major-11th", "augmented-11th", 
                         "augmented-major-13th", "augmented-dominant-13th"]

CHORD_TYPE_FAMILIES = [
    (MAJ_CHORD_TYPE,            MAJ_C2S_RULES),
    (MIN_CHORD_TYPE,            MIN_C2S_RULES),
    (DOM7_CHORD_TYPE,           DOM7_C2S_RULES),
    (DIM3_CHORD_TYPE,           DIM3_C2S_RULES),
    
    (MAJ7_FAMILY_CHORD_TYPE,    MAJ7_FAMILY_C2S_RULES),
    (MAJ7LYDIAN_CHORD_TYPE,     MAJ7LYDIAN_C2S_RULES),
    
    (MIN9_FAMILY_CHORD_TYPE,    MIN9_FAMILY_C2S_RULES),
    (MIN69_FAMILY_CHORD_TYPE,   MIN69_FAMILY_C2S_RULES),
    (MIN7_CHORD_TYPE,           MIN7_C2S_RULES),
    (MIN6_CHORD_TYPE,           MIN6_C2S_RULES),
    (MINMAJ_FAMILY_CHORD_TYPE,  MINMAJ_FAMILY_C2S_RULES),
    
    (DOM9_FAMILY_CHORD_TYPE,    DOM9_FAMILY_C2S_RULES),
    (DOM7b13_FAMILY_CHORD_TYPE, DOM7b13_FAMILY_C2S_RULES),
    (DOM7LYDIAN_CHORD_TYPE,     DOM7LYDIAN_C2S_RULES),
    (DOM7ALT_FAMILY_CHORD_TYPE, DOM7ALT_FAMILY_C2S_RULES),
    (DOM7SUS_FAMILY_CHORD_TYPE, DOM7SUS_FAMILY_C2S_RULES),
    
    (HALFDIM_FAMILY_CHORD_TYPE, HALFDIM_FAMILY_C2S_RULES),
    (DIM7_FAMILY_CHORD_TYPE,    DIM7_FAMILY_C2S_RULES),
    
    (AUG_FAMILY_CHORD_TYPE,     AUG_FAMILY_C2S_RULES),
]

'''
pitch_list should be ordered
key should be an integer (0-11) indicating the Major key (for minor key, convert it into its parallel Major key)
'''
def get_scale_name_suggestions(root, chord_type, key):
    if type(key) == str and key != UNKNOWN_KEY:
        key = PITCH_NAME_2_NUM[key]

    root_midi_num = root
    if type(key) == int:
        key_difference = (key - root_midi_num)%12
    else:
        key_difference = UNKNOWN_KEY
    
    done = False
    for chord_type_list, c2s_rules in CHORD_TYPE_FAMILIES:
        if chord_type in chord_type_list:
            if key_difference in c2s_rules:
                suggested_scale_names = c2s_rules[key_difference][:]
            else:
                suggested_scale_names = c2s_rules[UNKNOWN_KEY][:]
            done = True
            break
    
    if not done:
        suggested_scale_names = [(root_midi_num,"chromatic")]
    else:
        for i in range(len(suggested_scale_names)):
            root_shift, scale_name = suggested_scale_names[i]
            suggested_scale_names[i] = ((root_shift + root_midi_num)%12, scale_name)
            
    return suggested_scale_names

def get_scale_suggestions(pitch_list, key, root_midi_num = None, chord_type = None):
    pitch_multi_hot = [0]*12
    for p in pitch_list:
        pitch_multi_hot[p%12] = 1
    
    if not root_midi_num or not chord_type:
        c = music21.chord.Chord(pitch_list)
        _, chord_type = harmony.chordSymbolFigureFromChord(
            c, includeChordType=True
        )
        root_midi_num = c.root().midi % 12
    
    if type(key) == int:
        key_difference = (key - root_midi_num)%12
    else:
        key_difference = UNKNOWN_KEY
        
    suggested_scale_names = get_scale_name_suggestions(root_midi_num, chord_type, key)
    
    suggested_scale_weights = []
    suggested_scale_pitch_nums = []
    for root, scale_name in suggested_scale_names:
        scale_multihot, scale_pitch_nums = get_scale_pitches(root, scale_name)
        suggested_scale_weights.append(list(map(sum, zip(scale_multihot, pitch_multi_hot))))
        suggested_scale_pitch_nums.append(scale_pitch_nums)
    
    return suggested_scale_names, suggested_scale_weights, suggested_scale_pitch_nums


from __future__ import annotations


STRUCTURE_TO_CONTROL = {
    "PTV56": "PTV56_coverage",
    "PTV63": "PTV63_coverage",
    "PTV70": "PTV70_coverage",
    "Brainstem": "Brainstem_sparing",
    "SpinalCord": "SpinalCord_sparing",
    "RightParotid": "RightParotid_sparing",
    "LeftParotid": "LeftParotid_sparing",
    "Esophagus": "Esophagus_sparing",
    "Larynx": "Larynx_sparing",
    "Mandible": "Mandible_sparing",
    "lim_post_neck": "posterior_neck_sparing",
}


def objective_name_to_control(objective_name: str) -> str | None:
    pass

    structure_name = objective_name.split(" mean dose", 1)[0]
    structure_name = structure_name.split(" max dose", 1)[0]
    structure_name = structure_name.split(" c-var ", 1)[0]
    if "mm_ring" in structure_name:
        return "ring_falloff"
    return STRUCTURE_TO_CONTROL.get(structure_name)

import awkward as ak
import correctionlib
import numpy as np

def calc_jec_variation(
        pt, eta, phi, energy,
        jer_factor, jec_unc, orig_idx,
        variation_orig_idx, variation_jer_factor
        ):
    """
    Applies a JEC variation (up or down) on a 4-vector.

    Note there are 3 'ordering levels':
    - "Final": the final ordering of jets after centrally applied corrections
    - "Original": the ordering of jets _before_ any corrections
    - "Variation": the final ordering of jets after the applying the correction of the
        _variation_

    The algorithm below first creates a map to reorder "Final" to "Variation", then
    applies the correction after ordering everything in "Variation" ordering.

    Args:
        pt (ak.Array): jet pt
        eta (ak.Array): jet eta
        phi (ak.Array): jet phi
        energy (ak.Array): jet energy
        jer_factor (ak.Array): the JER factor that was applied centrally to obtain the
            final jet
        jec_unc (ak.Array): the JEC uncertainty
        orig_idx (ak.Array): mapping of final corrected jet ordering back to 'original'
            ordering
        variation_orig_idx (ak.Array): mapping of variation ordering back to 'original'
            ordering
        variation_jer_factor (ak.Array): the variation's JER factor

    Returns:
        (ak.Array, ak.Array, ak.Array, ak.Array, ak.Array) : pt, eta, phi, energy after
            applying the variation and reordered by the pT after variation; permutation
            to propagate the reordering to the whole collection
    """

    # Create a map to reorder final corrected jets to the ordering of the variation
    map_orig_idx_to_var_idx = ak.argsort(variation_orig_idx, axis=-1)
    map_final_idx_to_orig_idx = orig_idx
    reorder_final_to_var = map_final_idx_to_orig_idx[map_orig_idx_to_var_idx]

    # Reorder everything that is in "Final" order to "Variation" order
    pt = pt[reorder_final_to_var]
    eta = eta[reorder_final_to_var]
    phi = phi[reorder_final_to_var]
    energy = energy[reorder_final_to_var]
    jer_factor = jer_factor[reorder_final_to_var]
    jec_unc = jec_unc[reorder_final_to_var]

    corr = 1. / jer_factor * (1.+jec_unc) * variation_jer_factor
    return pt*corr, eta, phi, energy*corr, reorder_final_to_var


def calc_jer_variation(
        pt, eta, phi, energy,
        jer_factor, orig_idx,
        variation_orig_idx, variation_jer_factor
        ):
    """
    Applies a JER variation (up or down) on a 4-vector.

    Note there are 3 'ordering levels':
    - "Final": the final ordering of jets after centrally applied corrections
    - "Original": the ordering of jets _before_ any corrections
    - "Variation": the final ordering of jets after the applying the correction of the
        _variation_

    The algorithm below first creates a map to reorder "Final" to "Variation", then
    applies the correction after ordering everything in "Variation" ordering.

    Args:
        pt (ak.Array): jet pt
        eta (ak.Array): jet eta
        phi (ak.Array): jet phi
        energy (ak.Array): jet energy
        jer_factor (ak.Array): the JER factor that was applied centrally to obtain the
            final jet
        orig_idx (ak.Array): mapping of final corrected jet ordering back to 'original'
            ordering
        variation_orig_idx (ak.Array): mapping of variation ordering back to 'original'
            ordering
        variation_jer_factor (ak.Array): the variation's JER factor

    Returns:
        (ak.Array, ak.Array, ak.Array, ak.Array) : pt, eta, phi, and energy after
            applying the variation and reordered by the pT after variation.
    """

    # Create a map to reorder final corrected jets to the ordering of the variation
    map_orig_idx_to_var_idx = ak.argsort(variation_orig_idx, axis=-1)
    map_final_idx_to_orig_idx = orig_idx
    reorder_final_to_var = map_final_idx_to_orig_idx[map_orig_idx_to_var_idx]

    # Reorder everything that is in "Final" order to "Variation" order
    pt = pt[reorder_final_to_var]
    eta = eta[reorder_final_to_var]
    phi = phi[reorder_final_to_var]
    energy = energy[reorder_final_to_var]
    jer_factor = jer_factor[reorder_final_to_var]

    corr = 1. / jer_factor * variation_jer_factor
    return pt*corr, eta, phi, energy*corr, reorder_final_to_var


###############################
####### PFNano section ########
###############################


def calc_jec_variation_PFNano(
    events: ak.Array,
    # jets: ak.Array,
    year: str,
    run: str,
    radius: str,
    source: str = "Nominal",
    direction: str = "",
    JEC_level: str = "L1L2L3Res",
    ) -> ak.Array:

    pt = events.FatJet_pt
    eta = events.FatJet_eta
    phi = events.FatJet_phi
    mass = events.FatJet_mass
    area = events.FatJet_area
    rawFactor = events.FatJet_rawFactor

    pt = pt * (1.0 - rawFactor)

    # calculate all variables needed as inputs
    rhos = events.fixedGridRhoFastjetAll
    variables = {
        "count": ak.num(pt),
        "JetPt": ak.flatten(pt),
        "JetEta": ak.flatten(eta),
        "JetPhi": ak.flatten(phi),
        "JetA": ak.flatten(area),
        "Rho": ak.flatten(ak.broadcast_arrays(rhos[:, np.newaxis], pt)[0]),
    }

    evaluator = get_evaluator(year, radius)
    jec_version = get_jec_version(year=year, radius=radius, JEC_level=JEC_level)

    if jec_version in list(evaluator.compound.keys()):
        jec = evaluate_jec(corrector=evaluator.compound[jec_version], variables=variables)
    else:
        jec = evaluate_jec(corrector=evaluator[jec_version], variables=variables)

    if source != "Nominal" and run == "MC":
        # TODO need to check "year" for Run3 conventions
        allowed_sources = [
            "Absolute",
            f"Absolute_{year}",
            "BBEC1",
            f"BBEC1_{year}",
            "EC2",
            f"EC2_{year}",
            "RelativeBal",
            f"RelativeSample_{year}",
            "HF",
            f"HF_{year}",
            "FlavorQCD",
            "Total",
        ]
        allowed_directions = ["up", "down"]
        if not source in allowed_sources:
            raise ValueError(f"JEC direction is {direction}, but it can be only {allowed_sources}.")
        if not direction in allowed_directions:
            raise ValueError(f"JEC direction is {direction}, but it can be only {allowed_directions}.")
        jec_version = get_jec_version(year=year, radius=radius, JEC_level=f"{source}")
        unc = evaluate_jec(corrector=evaluator[jec_version], variables=variables)
        unc *= -1 if direction == "down" else 1
        jec += unc

    jec = ak.unflatten(jec, counts=variables["count"])
    pt = pt * jec
    rawFactor = 1.0 - (1.0 / jec)

    return pt, eta, phi, mass


def get_evaluator(year, radius):
    corr_path = "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME"
    corr_path += f"/{year}"
    corr_path += "/jet_jerc.json.gz" if int(radius) == 4 else "/fatJet_jerc.json.gz"

    return correctionlib.CorrectionSet.from_file(corr_path)


def evaluate_jec(corrector: correctionlib.CorrectionSet, variables: dict[str, ak.Array]) -> ak.Array:
    inputs = [variables[input.name] for input in corrector.inputs]
    return corrector.evaluate(*inputs)


def get_jec_version(year: str, radius: str, JEC_level: str = "L1L2L3Res") -> str:

    if year == "2018_UL": version = "Summer19UL18_V5_MC"
    elif year == "2017_UL": version = "Summer19UL17_V5_MC"
    elif year == "2016_UL": version = "Summer19UL16_V7_MC" #TODO: add 2016 APV if needed
    else: raise ValueError("Supported years are 2016_UL, 2017_UL, and 2018_UL")

    collection = "AK4PFchs" if int(radius) == 4 else "AK8PFPuppi"
    
    return f"{version}_{JEC_level}_{collection}"
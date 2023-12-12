import awkward as ak
import uproot

from utils.Logger import *


def __is_rootcompat(a):
    """Is it a flat or 1-d jagged array?"""

    t = ak.type(a)
    if isinstance(t, ak._ext.ArrayType):
        if isinstance(t.type, ak._ext.PrimitiveType):
            return True
        if isinstance(t.type, ak._ext.ListType) and isinstance(
            t.type.type, ak._ext.PrimitiveType
        ):
            return True
    return False


def __zip_composite(ak_array):
    """Zip together branches belonging to the same collection.
    
    Args:
        ak_array (ak.Array): 
    
    Returns:
        ak.Array
    """

    # Additional naming scheme to allow composite object readback
    _rename_lookup = {
        "pt": "/.fPt",
        "eta": "/.fEta",
        "phi": "/.fPhi",
        "energy": "/.fE",
        "x": "/.fX",
        "y": "/.fY",
        "z": "/.fZ",
        "j": "jerFactor",
        "o": "origIndex",
    }

    return ak.zip(
        {
            _rename_lookup.get(n, n): ak.packed(ak.without_parameters(ak_array[n]))
            for n in ak_array.fields
            if __is_rootcompat(ak_array[n])
        }
    )


def __make_tree_maker_event_tree(events):
    """Prepare dict with simple branches and collections that uproot can write.
    
    Args:
        events (ak.Array): Events opened with the NTreeMaker schema.

    Returns:
        dict[str, ak.Array]
    """

    out = {}
    for bname in events.fields:
        if events[bname].fields:
            sub_collection = [  # Handling sub collection first
                x.replace("Counts", "")
                for x in events[bname].fields
                if x.endswith("Counts")
            ]
            if sub_collection:
                for subname in sub_collection:
                    if events[bname][subname].fields:
                        out[f"{bname}_{subname}"] = __zip_composite(
                            ak.flatten(events[bname][subname], axis=-1)
                        )
                    else:
                        out[f"{bname}_{subname}"] = ak.flatten(
                            events[bname][subname], axis=-1
                        )
            out[bname] = __zip_composite(events[bname])
        else:
            out[bname] = ak.packed(ak.without_parameters(events[bname]))
    return out


def write_tree_maker_root_file(output_file_name, events, trees={}, mode="recreate"):
    """Write events opened with the TreeMaker or NTreeMaker schema to a ROOT file.
    
    Args:
        output_file_name (str)
        events (ak.Array)
        trees (dict[str, any]): Other trees with no collections
            Keys are tree name
            Values are tree content
        mode (str): "recreate" or "update"
    """

    log.blank_line()
    log.info("Writing down output ROOT file %s" % output_file_name)
    with getattr(uproot, mode)(output_file_name) as output_file:
        output_file["Events"] = __make_tree_maker_event_tree(events)
        for tree_name, tree in trees.items():
            output_file[tree_name] = ak.Array(tree)
        
    log.info("TTree %s saved to output file" % tree_name)


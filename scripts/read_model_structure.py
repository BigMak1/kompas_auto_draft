"""Read the structure of 3D models (.m3d/.a3d) already open in the running KOMPAS.

Read-only: attaches with GetKompas(), never opens, rebuilds, recalculates or saves
anything. Prints one JSON report per model: part properties, mass properties,
overall size, and the operation tree with each feature's variables.

Usage (x64 Python on the KOMPAS machine):
    python read_model_structure.py [name-substring ...]
Without arguments every open part/assembly is reported.
"""

import json
import sys

KOMPAS_BIN = r"C:\Program Files\ASCON\KOMPAS-3D v25 Study\Bin"

sys.path.insert(0, KOMPAS_BIN)
import ksapi  # noqa: E402  (loads ksAPICLink.dll from KOMPAS_BIN on import)
from constants import constants as K  # noqa: E402
from constants3d import constants as K3  # noqa: E402

OBJ3D_TYPES = {v: k[len("o3d_") :] for k, v in vars(K3).items() if k.startswith("o3d_")}
MODEL_DOC_TYPES = {K.ksDocumentPart: "part", K.ksDocumentAssembly: "assembly"}
MAX_DEPTH = 6


def safe(fn, *args, default=None):
    """Call a KsAPI getter; turn failures into a short marker instead of aborting the report."""
    try:
        return fn(*args)
    except Exception as exc:  # noqa: BLE001  (report must survive any single failing getter)
        return default if default is not None else f"<error: {type(exc).__name__}: {exc}>"


def read_variables(feature):
    out = []
    for var in safe(feature.GetVariables, False, False, default=[]):
        out.append(
            {
                "name": safe(var.GetName),
                "value": safe(var.GetValue),
                "expression": safe(var.GetExpression),
                "note": safe(var.GetNote) or safe(var.GetParameterNote),
                "external": safe(var.IsExternal),
                "information": safe(var.IsInformation),
            }
        )
    return out


def read_feature(feature, depth=0):
    ftype = safe(feature.GetFeatureType)
    node = {
        "name": safe(feature.GetName),
        "type": OBJ3D_TYPES.get(ftype, ftype),
        "excluded": safe(feature.IsExcluded),
        "valid": safe(feature.IsValid),
        "variables": read_variables(feature),
    }
    if depth < MAX_DEPTH:
        children = safe(feature.GetSubFeatures, K.ksOperTree, False, False, default=[])
        if children:
            node["children"] = [read_feature(child, depth + 1) for child in children if child]
    return node


def read_model(doc):
    doc3d = ksapi.IKompasDocument3D(doc)
    part = doc3d.GetTopPart()
    mass = ksapi.IMassInertiaParam(part)
    gabarit = safe(part.GetGabarit, True, False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    size = None
    if isinstance(gabarit, tuple) and gabarit[0]:
        x1, y1, z1, x2, y2, z2 = gabarit[1:]
        size = {"min": [x1, y1, z1], "max": [x2, y2, z2], "size": [x2 - x1, y2 - y1, z2 - z1]}
    report = {
        "name": safe(doc.GetName),
        "full_path": safe(doc.GetFullPath),
        "doc_type": MODEL_DOC_TYPES.get(safe(doc.GetDocumentType)),
        "read_only": safe(doc.IsReadOnly),
        "changed": safe(doc.IsChanged),
        "part": {
            "name": safe(part.GetName),
            "marking": safe(part.GetMarking),
            "material": safe(part.GetMaterial),
            "density": safe(part.GetDensity),
            "mass": safe(part.GetMass),
            "is_detail": safe(part.IsDetail),
        },
        "mass_properties": {
            "actual": safe(mass.IsActual),
            "mass": safe(mass.GetMass),
            "volume": safe(mass.GetVolume),
            "area": safe(mass.GetArea),
            "density": safe(mass.GetDensity),
            "material": safe(mass.GetMaterial),
            "mass_units": safe(mass.GetMassUnits),
            "length_units": safe(mass.GetLengthUnits),
        }
        if mass
        else None,
        "gabarit": size,
        "tree": read_feature(ksapi.IFeature(part)),
    }
    return report


def main():
    filters = [a.lower() for a in sys.argv[1:]]
    app = ksapi.GetKompas()
    if app is None:
        print("GetKompas: no running KOMPAS visible from this session", file=sys.stderr)
        return 2
    previous = app.GetHideMessage()
    app.SetHideMessage(K.ksHideMessageNo)  # no modal dialog may block an unattended read
    try:
        for obj in app.GetDocuments().GetObjects():
            doc = ksapi.IKompasDocument(obj)
            if safe(doc.GetDocumentType) not in MODEL_DOC_TYPES:
                continue
            name = str(safe(doc.GetName))
            if filters and not any(f in name.lower() for f in filters):
                continue
            print(json.dumps(read_model(doc), ensure_ascii=False, indent=1, default=str))
    finally:
        app.SetHideMessage(previous)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

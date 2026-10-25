"""Add a through hole at the centre of a planar face of a part, in the running KOMPAS.

Opens the given .m3d visibly (or reuses it if already open), surveys its planar
faces and picks the largest one whose centre leaves room for the hole: the centre
and a ring of points at radius + margin must all lie on the face, so the hole
does not touch an edge or an existing hole.

Dry run by default: prints the survey and the chosen face, changes nothing.
With --apply it builds a sketch on that face with one circle, cuts it through
the whole part in both directions, rebuilds and saves the document.

Refuses to touch a file whose name does not contain "копия" (work on copies only).

Usage (x64 Python on the KOMPAS machine):
    python add_center_hole.py <path-to-copy.m3d> [--diameter 4] [--apply]
"""

import argparse
import json
import math
import sys

KOMPAS_BIN = r"C:\Program Files\ASCON\KOMPAS-3D v25 Study\Bin"

sys.path.insert(0, KOMPAS_BIN)
import ksapi  # noqa: E402  (loads ksAPICLink.dll from KOMPAS_BIN on import)
from constants import constants as K  # noqa: E402
from constants3d import constants as K3  # noqa: E402

MARGIN_MM = 1.0
RING_POINTS = 24
ON_FACE_TOL = 1e-6


def unit(v):
    n = math.sqrt(sum(c * c for c in v))
    return [c / n for c in v]


def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def in_plane_basis(normal):
    helper = [1.0, 0.0, 0.0] if abs(normal[0]) < 0.9 else [0.0, 1.0, 0.0]
    e1 = unit(cross(normal, helper))
    return e1, cross(normal, e1)


def survey_face(face, radius):
    """Centre, normal, area and whether a hole of `radius` fits at the centre."""
    u = (face.GetParamUMin() + face.GetParamUMax()) / 2
    v = (face.GetParamVMin() + face.GetParamVMax()) / 2
    ok_p, cx, cy, cz = face.GetPoint(u, v, 0.0, 0.0, 0.0)
    ok_n, nx, ny, nz = face.GetNormal(u, v, 0.0, 0.0, 0.0)
    if not (ok_p and ok_n):
        return None
    centre, normal = [cx, cy, cz], unit([nx, ny, nz])
    e1, e2 = in_plane_basis(normal)
    ring = radius + MARGIN_MM
    probes = [centre] + [
        [centre[k] + ring * (math.cos(a) * e1[k] + math.sin(a) * e2[k]) for k in range(3)]
        for a in (2 * math.pi * i / RING_POINTS for i in range(RING_POINTS))
    ]
    worst = max(face.GetDistanceToPoint(*p) for p in probes)
    return {
        "area_mm2": face.GetArea(K.ksLUnMM),
        "centre": centre,
        "normal": normal,
        "max_probe_distance": worst,
        "fits": worst < ON_FACE_TOL,
    }


def open_document(app, path):
    docs = app.GetDocuments()
    existing = docs.GetItemByFilePath(path)
    doc = existing if existing else docs.Open(path, True, False)
    return ksapi.IKompasDocument3D(doc) if doc else None


def cut_hole(part, face, centre, radius):
    container = ksapi.IModelContainer(part)
    sketch = container.GetSketchs().Add()
    sketch.SetPlane(face)
    sketch.SetAngle(0)
    if not sketch.Update():
        raise RuntimeError("sketch on the face was not created")
    ok, sx, sy = sketch.GetPointProjectionToXY(centre[0], centre[1], centre[2], 0.0, 0.0)
    if not ok:
        raise RuntimeError("could not project the face centre into the sketch")
    fragment = sketch.BeginEdit(False)
    if fragment is None:
        raise RuntimeError("could not enter sketch edit mode")
    try:
        view = fragment.GetViewsAndLayersManager().GetViews().GetActiveView()
        circle = ksapi.IDrawingContainer(view).GetCircles().Add()
        circle.SetXc(sx)
        circle.SetYc(sy)
        circle.SetRadius(radius)
        if not circle.Update():
            raise RuntimeError("circle was not created in the sketch")
    finally:
        sketch.EndEdit()

    extrusion = container.GetExtrusions().Add(K3.o3d_cutExtrusion)
    extrusion.SetProfiles([sketch])
    extrusion.SetDirection(K3.dtBoth)
    extrusion.SetExtrusionType(True, K3.etThroughAll)
    extrusion.SetExtrusionType(False, K3.etThroughAll)
    cut = ksapi.ICutExtrusion(extrusion)
    if cut:
        cut.SetCut(True)
    if not extrusion.Update():
        raise RuntimeError("cut extrusion failed to build")
    return sketch, extrusion, (sx, sy)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--diameter", type=float, default=4.0)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if "копия" not in args.path.lower():
        print("refusing: work on copies only (file name must contain 'копия')", file=sys.stderr)
        return 3
    radius = args.diameter / 2

    app = ksapi.GetKompas()
    if app is None:
        print("GetKompas: no running KOMPAS visible from this session", file=sys.stderr)
        return 2
    previous = app.GetHideMessage()
    app.SetHideMessage(K.ksHideMessageNo)  # no modal dialog may block an unattended run
    try:
        doc3d = open_document(app, args.path)
        if doc3d is None:
            print("could not open the document", file=sys.stderr)
            return 4
        part = doc3d.GetTopPart()
        mass_before = ksapi.IMassInertiaParam(part).GetMass()

        candidates = []
        for obj in ksapi.IModelContainer(part).GetObjects(K3.o3d_face):
            face = ksapi.IFace(obj)
            if face and face.IsPlanar():
                info = survey_face(face, radius)
                if info:
                    candidates.append((info, face))
        candidates.sort(key=lambda c: c[0]["area_mm2"], reverse=True)
        report = {
            "document": doc3d.GetFullPath(),
            "diameter": args.diameter,
            "planar_faces": [c[0] for c in candidates],
        }
        chosen = next((c for c in candidates if c[0]["fits"]), None)
        report["chosen"] = chosen[0] if chosen else None
        if chosen is None or not args.apply:
            report["applied"] = False
            print(json.dumps(report, ensure_ascii=False, indent=1))
            return 0 if chosen else 5

        sketch, extrusion, sketch_xy = cut_hole(part, chosen[1], chosen[0]["centre"], radius)
        part.RebuildModel(True)
        mass = ksapi.IMassInertiaParam(part)
        mass.Calculate()
        report.update({
            "applied": True,
            "sketch": sketch.GetName(),
            "sketch_xy": sketch_xy,
            "operation": extrusion.GetName(),
            "mass_before_g": mass_before,
            "mass_after_g": mass.GetMass(),
            "saved": doc3d.Save(),
        })
        print(json.dumps(report, ensure_ascii=False, indent=1))
        return 0
    finally:
        app.SetHideMessage(previous)


if __name__ == "__main__":
    raise SystemExit(main())

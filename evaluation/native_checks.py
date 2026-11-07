"""Нативные контрольные кубы и проверка цилиндра отверстия на копиях."""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from evaluation.native_export import atomic_json  # noqa: E402


def main() -> None:
    directory, bin_dir = map(Path, sys.argv[1:3])
    sys.path.insert(0, str(REPO / "vendor" / "kompas_lib"))
    import kompas_lib as lib

    from harness.native_worker import adapt_feature_diagnostics

    app = lib.Kompas.connect(start=False, bin_dir=str(bin_dir))
    ks = app.ks
    adapt_feature_diagnostics(ks.IFeature)
    original = app.active_document
    facts = {}
    try:
        for label in ("prediction", "reference"):
            model = directory / "inputs" / f"{label}.m3d"
            if app.find_open(model) is not None:
                raise RuntimeError("Snapshot already open")
            doc = app.open(model, read_only=True, visible=False)
            try:
                faces = ks.IFeature(doc.part).GetModelObjects([lib.by_name("o3d_face")]) or []
                cylinders = []
                for i, f in enumerate(faces[:48]):
                    surface = ks.IMathSurface3D(f.GetMathObject())
                    if surface.IsCylinder():
                        radius = surface.GetRadius()
                        if 10 < radius < 20:
                            placement = surface.GetPlacement()
                            origin = placement.GetOrigin(0, 0, 0)
                            axis = placement.GetVector(lib.by_name("o3d_axisOZ"), 0, 0, 0)
                            cylinders.append(
                                {"face_index": i, "radius_mm": radius, "axis_origin": origin, "axis": axis}
                            )
                            if origin[0] and abs(origin[1]) < 1e-6 and abs(origin[2]) < 1e-6:
                                break
                facts[label] = {"central_cylinders": cylinders, "faces_scanned_limit": 48}
                del faces
            finally:
                doc.close(save=False)
        atomic_json(directory / "local-native-features.json", facts)
        check = directory / "sanity-overlap-cubes"
        (check / "inputs").mkdir(parents=True, exist_ok=True)
        (check / "native-v2").mkdir(exist_ok=True)
        for label, x in [("prediction", 0), ("reference", 5)]:
            doc = app.new_part(visible=False, tag="evaluation:control:cube")
            try:
                builder = lib.PartBuilder(doc)
                with doc.edit():
                    sketch = builder.sketch("XY", lib.rect(x, 0, x + 10, 10))
                    builder.extrude(sketch, 10)
                doc.save_as(check / "inputs" / f"{label}.m3d")
                mi = ks.IMassInertiaParam(doc.part)
                mi.SetLengthUnits(lib.by_name("ksLUnMM"))
                mi.Calculate()
                atomic_json(check / "native-v2" / f"{label}.json", {"volume_mm3": mi.GetVolume()})
            finally:
                doc.close(save=False)
        atomic_json(
            check / "alignment.json",
            {"matrix_prediction_to_reference": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]},
        )
        atomic_json(check / "surface-metrics.json", {"prediction": {"bbox_mm": [[0, 0, 0], [10, 10, 10]]}})
        print(json.dumps(facts), flush=True)
    finally:
        if original is not None:
            original.activate()


if __name__ == "__main__":
    main()

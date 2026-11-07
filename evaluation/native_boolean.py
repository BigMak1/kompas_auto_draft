"""Нативное пересечение B-rep в отдельной временной детали КОМПАС.

Запуск: python native_boolean.py <comparison-directory> <KOMPAS-Bin>.
Только оценщик: копирование эталона запрещено в MCP построителя.
"""

import json
import math
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from evaluation.native_export import atomic_json  # noqa: E402


def main() -> None:
    directory, bin_dir = map(Path, sys.argv[1:3])
    sys.path.insert(0, str(REPO / "vendor" / "kompas_lib"))
    import kompas_lib as lib

    matrix = json.loads((directory / "alignment.json").read_text())["matrix_prediction_to_reference"]
    native = {
        label: json.loads((directory / "native-v2" / f"{label}.json").read_text(encoding="utf-8"))
        for label in ("prediction", "reference")
    }
    app = lib.Kompas.connect(start=False, bin_dir=str(bin_dir))
    ks = app.ks
    original = app.active_document
    opened = []
    result = {"status": "started", "engine": "KOMPAS-3D B-rep IBoolean", "stages": [], "transform": matrix}
    output = directory / "native-boolean.json"

    def stage(name: str, **data) -> None:
        result["stages"].append({"stage": name, **data})
        atomic_json(output, result)
        print(name, json.dumps(data), flush=True)

    def volume(part) -> float:
        ks.IKompasDocument3D(scratch.raw).RebuildDocument()
        mi = ks.IMassInertiaParam(part)
        mi.SetLengthUnits(lib.by_name("ksLUnMM"))
        mi.SetSourceData(False)
        mi.SetCopySourceData(False)
        mi.SetUseMassFromSource(False)
        mi.Calculate()
        return float(mi.GetVolume())

    try:
        sources = {}
        for label in ("prediction", "reference"):
            path = directory / "inputs" / f"{label}.m3d"
            if app.find_open(path) is not None:
                raise RuntimeError(f"Snapshot already open: {path}")
            sources[label] = app.open(path, read_only=True, visible=False)
            opened.append(sources[label])
        scratch = app.new_part(visible=False, tag="evaluation:kf0011:intersection")
        opened.append(scratch)
        container = ks.IModelContainer(scratch.part)
        copied = {}
        scratch.activate()
        for label, source in sources.items():
            before_ids = {ks.IBody(b).GetBodyId() for b in ks.IFeature(scratch.part).GetResultBodies() or []}
            bodies = ks.IFeature(source.part).GetResultBodies() or []
            if len(bodies) != 1:
                raise ValueError("This evaluator currently requires one body per model")
            copy = container.GetCopiesGeometry().Add()
            copy.SetContextObjects(False)
            copy.SetAutoUpdate(False)
            copy.SetWatchForSourceChange(False)
            if not copy.AddInitialObjectsFromExternalDocument(ks.IKompasDocument3D(source.raw), None, bodies):
                raise RuntimeError("Adding external body failed")
            updated = copy.Update()
            found = [
                b
                for b in ks.IFeature(scratch.part).GetResultBodies() or []
                if ks.IBody(b).GetBodyId() not in before_ids
            ]
            stage(
                "copy_" + label,
                updated=updated,
                bodies=len(found),
                error=ks.IFeature(copy).GetObjectError(),
                total_volume_mm3=volume(scratch.part),
            )
            if not updated or len(found) != 1 or ks.IFeature(copy).GetObjectError():
                raise RuntimeError("Copy geometry failed")
            copied[label] = found[0]
        total = volume(scratch.part)
        expected = sum(m["volume_mm3"] for m in native.values())
        if not math.isclose(total, expected, rel_tol=1e-7, abs_tol=1e-4):
            raise RuntimeError(f"Copied body volumes differ: {total} vs {expected}")
        move = container.GetBodyRepositions().Add()
        move.SetRepositionBody(copied["prediction"])
        move.SetCopyBody(False)
        position = move.GetPosition()
        # C3D хранит матрицу с переносом в последней строке: transpose column-vector matrix.
        if not position.InitByMatrix3D([matrix[j][i] for i in range(4) for j in range(4)]):
            raise RuntimeError("Position matrix rejected")
        updated = move.Update()
        ks.IKompasDocument3D(scratch.raw).RebuildDocument()
        found = ks.IFeature(scratch.part).GetResultBodies() or []
        stage("reposition", updated=updated, bodies=len(found), error=ks.IFeature(move).GetObjectError())
        if not updated or len(found) != 2 or ks.IFeature(move).GetObjectError():
            raise RuntimeError("Body reposition failed")
        measured = json.loads((directory / "surface-metrics.json").read_text())["prediction"]["bbox_mm"]
        expected_box = measured[0] + measured[1]
        boxes = [ks.IBody(b).GetGabarit(0, 0, 0, 0, 0, 0) for b in found]
        index = min(
            range(len(found)),
            key=lambda i: max(abs(a - b) for a, b in zip(boxes[i][1:], expected_box, strict=True)),
        )
        copied["prediction"], copied["reference"] = found[index], found[1 - index]
        box = boxes[index]
        error = max(abs(a - b) for a, b in zip(box[1:], expected_box, strict=True))
        stage("transform_verification", bbox_native=box, bbox_mesh=expected_box, maximum_difference_mm=error)
        if not box[0] or error > 0.02:
            raise RuntimeError("Native and mesh coordinate transforms disagree")
        scratch.save_as(directory / "boolean-inputs.m3d")
        boolean = container.GetBooleans().Add()
        boolean.SetBooleanType(lib.by_name("ksIntersect"))
        boolean.SetBodies([copied["prediction"], copied["reference"]])
        boolean.SetBaseObject(copied["prediction"])
        boolean.SetModifyObjects([copied["reference"]])
        boolean.SetSaveCopyBaseObject(False)
        boolean.SetSaveCopyModifyObjects(False)
        with app.quiet():
            updated = boolean.Update()
        ks.IKompasDocument3D(scratch.raw).RebuildDocument()
        bodies = ks.IFeature(scratch.part).GetResultBodies() or []
        stage(
            "intersection", updated=updated, error=ks.IFeature(boolean).GetObjectError(), bodies=len(bodies)
        )
        if not updated or ks.IFeature(boolean).GetObjectError() or not bodies:
            raise RuntimeError("Native intersection did not produce a valid solid")
        intersection = volume(scratch.part)
        vp, vr = native["prediction"]["volume_mm3"], native["reference"]["volume_mm3"]
        if not 0 < intersection <= min(vp, vr) + 1e-5:
            raise RuntimeError("Intersection volume outside bounds")
        union = vp + vr - intersection
        result.update(
            {
                "status": "ok",
                "intersection_mm3": intersection,
                "union_mm3": union,
                "iou": intersection / union,
                "dice": 2 * intersection / (vp + vr),
                "missing_volume_mm3": vr - intersection,
                "extra_volume_mm3": vp - intersection,
                "missing_fraction_reference": (vr - intersection) / vr,
                "extra_fraction_prediction": (vp - intersection) / vp,
                "extra_fraction_reference": (vp - intersection) / vr,
                "symmetric_difference_mm3": vp + vr - 2 * intersection,
                "symmetric_difference_fraction_union": (vp + vr - 2 * intersection) / union,
            }
        )
        scratch.save_as(directory / "intersection.m3d")
    except Exception as exc:
        result.update(
            {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
        )
        raise
    finally:
        atomic_json(output, result)
        for doc in reversed(opened):
            doc.close(save=False)
        if original is not None:
            original.activate()


if __name__ == "__main__":
    main()

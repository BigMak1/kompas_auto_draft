"""Выгрузка копий M3D через живой КОМПАС, без изменения исходных документов.

Запуск на Windows: python native_export.py <comparison-directory> <KOMPAS-Bin>.
STL — только поверхность для расстояний; объём и площадь читаются из native B-rep.
"""

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def atomic_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def main() -> None:
    directory, bin_dir = map(Path, sys.argv[1:3])
    sys.path.insert(0, str(REPO / "vendor" / "kompas_lib"))
    import kompas_lib as lib

    app = lib.Kompas.connect(start=False, bin_dir=str(bin_dir))
    original = app.active_document
    output = directory / "native-v2"
    output.mkdir(exist_ok=True)
    try:
        for label in ("prediction", "reference"):
            model = directory / "inputs" / f"{label}.m3d"
            if app.find_open(model) is not None:
                raise RuntimeError(f"Evaluator snapshot already open: {model}")
            doc = app.open(model, read_only=True, visible=False)
            try:
                part, ks = doc.part, app.ks
                mi = ks.IMassInertiaParam(part)
                mi.SetLengthUnits(lib.by_name("ksLUnMM"))
                mi.Calculate()
                feature = ks.IFeature(part)
                bodies = feature.GetResultBodies() or []
                em = ks.IEmbodimentsManager(part)
                info = {
                    "file_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                    "kompas_version": app.version,
                    "kompas_build": app.build,
                    "length_units": mi.GetLengthUnits(),
                    "mass_units": mi.GetMassUnits(),
                    "volume_mm3": mi.GetVolume(),
                    "area_mm2": mi.GetArea(),
                    "mass_native": mi.GetMass(),
                    "center_mm": [mi.GetXc(), mi.GetYc(), mi.GetZc()],
                    "inertia_native": [
                        mi.GetJx(),
                        mi.GetJy(),
                        mi.GetJz(),
                        mi.GetJxy(),
                        mi.GetJxz(),
                        mi.GetJyz(),
                    ],
                    "bbox_native": part.GetGabarit(False, False, 0, 0, 0, 0, 0, 0),
                    "bodies": len(bodies),
                    "solid": [ks.IBody(body).IsSolid() for body in bodies],
                    "faces": len(feature.GetModelObjects([lib.by_name("o3d_face")]) or []),
                    "part_error": feature.GetObjectError(),
                    "embodiment_count": em.GetEmbodimentCount(),
                    "embodiment_index": em.GetCurrentEmbodimentIndex(),
                    "document_comment": doc.raw.GetComment(),
                    "tree": [
                        {
                            "name": f.GetName(),
                            "type": ks.IModelObject(f).GetModelObjectType(),
                            "excluded": f.IsExcluded(),
                            "error": f.GetObjectError(),
                        }
                        for f in feature.GetSubFeatures(1, True, False) or []
                    ],
                    "exports": [],
                }
                atomic_json(output / f"{label}.json", info)
                for chord in (0.01, 0.003):
                    path = output / f"{label}-{chord:g}.stl"
                    converter = app.raw.GetConverter(lib.by_name("ksConverterToSTL"))
                    params = ks.IAdditionConvertParameters(
                        converter.GetConverterParameters(lib.by_name("ksCMFileSaveAs"))
                    )
                    params.SetFormat(lib.by_name("ksConverterToSTL"))
                    params.SetFormatBinary(True)
                    params.SetLengthUnits(lib.by_name("ksLUnMM"))
                    params.SetStepType(lib.by_name("ksSpaceStep") | lib.by_name("ksDeviationStep"))
                    params.SetStep(chord)
                    params.SetAngle(0.05)
                    params.SetMaxTesselationCellCount(2000000)
                    params.SetExportObjects(bodies)
                    with app.quiet():
                        success = ks.IKompasDocument3D(doc.raw).ConvertToAdditionFormat(str(path), params)
                    if not success or not path.is_file() or path.stat().st_size < 84:
                        raise RuntimeError(f"Failed STL export: {path}; result={success}")
                    content = path.read_bytes()
                    info["exports"].append(
                        {
                            "path": path.name,
                            "requested_chord_mm": chord,
                            "step_type": params.GetStepType(),
                            "step": params.GetStep(),
                            "sha256": hashlib.sha256(content).hexdigest(),
                            "bytes": len(content),
                            "actual_format": "binary"
                            if len(content) == 84 + 50 * int.from_bytes(content[80:84], "little")
                            else "ascii",
                            "format_binary_setting": params.IsFormatBinary(),
                            "angle_setting": params.GetAngle(),
                        }
                    )
                    atomic_json(output / f"{label}.json", info)
                print(json.dumps(info, ensure_ascii=False), flush=True)
                del feature, bodies, em, mi, part, params, converter
            finally:
                doc.close(save=False)
    finally:
        if original is not None:
            original.activate()


if __name__ == "__main__":
    main()

"""Доверенный исполнитель Windows. Не запускает переданный агентом Python."""

import argparse
import contextlib
import dataclasses
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any
from zipfile import ZipFile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from harness.expressions import resolve, variables  # noqa: E402
from harness.parametric import add_variables, bind_parameters, describe_variable  # noqa: E402
from harness.port_budget import PacedLibrary, windows_port_budget  # noqa: E402
from harness.recipe import validate  # noqa: E402

V25_DETAILED_FACE_LIMIT = 256


def atomic(path: Path, content: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def adapt_feature_diagnostics(feature_type: type[Any]) -> bool:
    """Подключить диагностику v26 и вернуть доступность предупреждений дерева."""

    def has_error(feature: Any) -> bool:
        return bool(feature.GetObjectError())

    def object_errors(feature: Any) -> list[int]:
        code = feature.GetObjectError()
        return [code] if code else []

    if not hasattr(feature_type, "IsHasError"):
        feature_type.IsHasError = has_error
    if not hasattr(feature_type, "GetObjectErrors"):
        feature_type.GetObjectErrors = object_errors
    if not hasattr(feature_type, "IsHasWarning"):
        # В v25 доступны ошибки операций и проверки геометрии, но нет предупреждений дерева.
        feature_type.IsHasWarning = lambda feature: False
        return False
    return True


def count_model_objects(ks: Any, feature: Any, object_type: int) -> int:
    """Посчитать объекты из GetModelObjects без создания обёртки для каждого."""
    api = ks.ksapi
    app = feature.kompasApp.GetReference()
    types = ks.ToInt32Vector([object_type])
    try:
        objects = api.IFeature_GetModelObjects(app, ks.GetReference(feature), types)
        if not objects:
            return 0
        try:
            return int(api.ObjectReferenceVector_Count(objects))
        finally:
            api.ObjectReferenceVector_Delete(app, objects)
    finally:
        api.Int32Vector_Delete(types)


def edge_order(edge: Any) -> tuple:
    """Индекс задаёт геометрия ребра, независимо от его направления и адреса."""

    def point(value: Any) -> tuple:
        return tuple(round(coordinate, 6) for coordinate in value) if value is not None else ()

    return (
        edge.kind,
        point(edge.mid),
        tuple(sorted((point(edge.start), point(edge.end)))),
        point(edge.center),
        round(edge.length, 6),
        edge.convexity or "",
    )


def inspect_runtime(bin_dir: Path) -> dict:
    """Только подключение и импорт: без открытия, создания и изменения документов."""
    sys.path.insert(0, str(bin_dir))
    sys.path.insert(0, str(REPO / "vendor" / "kompas_lib"))
    import kompas_lib as lib
    import ksapi

    app = ksapi.GetKompas()
    if app is None:
        raise RuntimeError("No running KOMPAS")
    names = ("ksapi.py", "constants.py", "constants3d.py", "ksAPICLink.dll")
    files = {name: (bin_dir / name).is_file() for name in names}
    symbols = {
        name: callable(getattr(lib, name))
        for name in ("PartBuilder", "line", "circle", "arc", "rect", "polygon", "render_model")
    }
    if not all(files.values()) or not all(symbols.values()):
        raise RuntimeError("Incomplete runtime")
    return {
        "status": "read_only_preflight_passed",
        "python": sys.version,
        "kompas_version": app.GetApplicationName(True),
        "kompas_build": app.GetBuildNumber(),
        "runtime_files": files,
        "library_symbols": symbols,
        "native_model_created": False,
        "recipe_version": 2,
        "parametric_capabilities": ["variables", "arithmetic", "sketch_constraints", "parameter_bindings"],
    }


def perform(recipe: dict, directory: Path, bin_dir: Path) -> dict:
    validate(recipe)
    for name in ("ksapi.py", "constants.py", "constants3d.py", "ksAPICLink.dll"):
        if not (bin_dir / name).is_file():
            raise RuntimeError(f"Missing runtime file: {name}")
    os.environ["KOMPAS_BIN_DIR"] = str(bin_dir)
    sys.path.insert(0, str(REPO / "vendor" / "kompas_lib"))
    import kompas_lib as lib

    ks = lib._loader.load(bin_dir)
    feature_warnings_available = adapt_feature_diagnostics(ks.IFeature)
    if not feature_warnings_available:
        ks.ksapi = PacedLibrary(ks.ksapi, windows_port_budget())
    app = lib.Kompas.connect(start=False, bin_dir=str(bin_dir))

    class WorkerPartBuilder(lib.PartBuilder):
        def _read_metrics(self) -> Any:
            if feature_warnings_available:
                return super()._read_metrics()
            part = self.part
            mass = self.ks.IMassInertiaParam(part)
            mass.Calculate()
            feature = self.ks.IFeature(part)
            bodies = feature.GetResultBodies() or []
            faces = count_model_objects(self.ks, feature, int(lib.by_name("o3d_face")))
            box = None
            if faces:
                bounds = part.GetGabarit(False, False, 0, 0, 0, 0, 0, 0)
                if bounds[0]:
                    box = lib.Box3(*bounds[1:7])
            return lib.Metrics(
                float(mass.GetVolume()) * 1000.0, float(part.GetMass()), faces, len(bodies), box
            )

        def check_quality(self, op: Any = None) -> tuple[list[str], list[str]]:
            # Тысячи вызовов по рёбрам и граням обрывают подключение KsAPI v25.
            if not feature_warnings_available and self.metrics().faces > V25_DETAILED_FACE_LIMIT:
                return [], ["Детальная проверка рёбер и граней пропущена в KsAPI v25."]
            return super().check_quality(op)

    original = app.active_document
    doc = app.new_part(tag="e2e:" + directory.name)
    result = None
    try:
        builder = WorkerPartBuilder(doc)
        refs: dict[str, Any] = {p: p for p in ("XY", "XZ", "YZ")}
        values = variables(recipe.get("variables", []))
        parametric: dict[str, Any] = {"steps": {}}
        with doc.edit(), doc.geometry_cache():
            add_variables(builder, recipe.get("variables", []), values)
            for step in recipe["steps"]:
                op, name = step["op"], step["id"]
                args = resolve({k: v for k, v in step.items() if k not in ("op", "id", "bindings")}, values)
                if "count" in args:
                    args["count"] = int(args["count"])
                if op == "sketch":
                    items = []
                    for item in args["items"]:
                        kwargs = {k: v for k, v in item.items() if k not in ("type", "args")}
                        shape_args = [item["args"]] if item["type"] == "polygon" else item["args"]
                        shape = getattr(lib, item["type"])(*shape_args, **kwargs)
                        items.extend(shape if isinstance(shape, list) else [shape])
                    args["items"] = items
                for key in ("base", "plane", "sketch"):
                    if key in args:
                        args[key] = refs[args[key]]
                if "edges" in args:
                    ordered = sorted(builder.edges(), key=edge_order)
                    args["edges"] = [ordered[i] for i in args["edges"]]
                    del ordered
                if "steps" in args:
                    args["steps"] = [refs[s] for s in args["steps"]]
                refs[name] = getattr(builder, op)(**args, name=name)
                parametric["steps"][name] = bind_parameters(builder, refs[name], step, values)
        parametric["variables"] = [describe_variable(v) for v in builder.vars.list()]
        parametric["unused_variables"] = builder.vars.unused() if values else []
        with doc.geometry_cache():
            metrics = dataclasses.asdict(builder.metrics())
            report = builder.report()
        if metrics["volume"] <= 0 or metrics["bodies"] != 1:
            raise RuntimeError("Expected exactly one nonempty solid")
        feature = app.ks.IFeature(doc.part)
        bodies = feature.GetResultBodies() or []
        if len(bodies) != 1 or not app.ks.IBody(bodies[0]).IsSolid():
            raise RuntimeError("Expected exactly one nonempty solid")
        model_temp = directory / "model.pending.m3d"
        doc.save_as(model_temp)
        if not model_temp.is_file() or model_temp.stat().st_size == 0:
            raise RuntimeError("KOMPAS did not write the model")
        with ZipFile(model_temp) as model:
            if "Contents" not in model.namelist() or model.testzip() is not None:
                raise RuntimeError("KOMPAS wrote an invalid M3D container")
        if not feature_warnings_available:
            report += "\nПредупреждения дерева недоступны через установленную обёртку KsAPI."
        result = {
            "status": "built",
            "kompas_version": app.version,
            "kompas_build": app.build,
            "metrics": metrics,
            "report": report,
            "parametric": parametric,
        }
        if feature_warnings_available:
            try:
                atomic(directory / "preview.png", lib.render_model(doc, max_px=1600))
            except Exception as exc:  # noqa: BLE001 -- артефакт сохраняем и при ошибке рендера
                result["preview_error"] = str(exc)
        if not feature_warnings_available and metrics["faces"] > V25_DETAILED_FACE_LIMIT:
            result["edges"] = []
            result["edges_truncated"] = True
        else:
            try:
                ordered = sorted(builder.edges(), key=edge_order)
                result["edges"] = [{"index": i, "description": str(e)} for i, e in enumerate(ordered)]
                del ordered
            except OSError as exc:
                result.update(edges=[], edges_truncated=True, edges_error=str(exc))
        del bodies, feature
    finally:
        try:
            doc.close(save=False)  # Только документ, созданный этим вызовом.
        except OSError as exc:
            if result is None:
                raise
            result["cleanup_error"] = str(exc)
        else:
            if original:
                try:
                    original.activate()
                except OSError as exc:
                    if result is None:
                        raise
                    result["restore_view_error"] = str(exc)
    model_path = directory / "model.m3d"
    data = model_temp.read_bytes()
    if "cleanup_error" in result:
        atomic(model_path, data)
    else:
        model_temp.replace(model_path)
    result["model_sha256"] = hashlib.sha256(data).hexdigest()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--bin-dir", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        result = inspect_runtime(args.bin_dir)
        atomic(
            args.directory / "native-preflight.json",
            json.dumps(result, ensure_ascii=False, indent=2).encode(),
        )
        print(json.dumps(result, ensure_ascii=False))
        return
    recipe = json.loads(sys.stdin.buffer.read(1_000_001))
    try:
        with contextlib.redirect_stdout(sys.stderr):
            result = perform(recipe, args.directory, args.bin_dir)
    except Exception as exc:  # noqa: BLE001 -- сохранить неудачный прогон на границе процесса
        traceback.print_exc()
        result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    atomic(args.directory / "result.json", json.dumps(result, ensure_ascii=False, indent=2).encode())
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

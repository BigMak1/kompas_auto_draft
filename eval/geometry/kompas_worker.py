"""Windows-вычислитель KsAPI. Запуск отдельным процессом: <job-directory> <Bin>.

Только стандартная библиотека и vendor/kompas_lib. Измерения на копиях, один процесс,
подключение к открытому КОМПАС. Источники API и живые проверки — README.md.
"""

import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]


def digest(data: Any) -> str:
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, data: Any) -> None:
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


@contextmanager
def exclusive_session() -> Iterator[None]:
    import msvcrt

    path = Path(tempfile.gettempdir()) / "kompas-geometry-evaluator.lock"
    with path.open("a+b") as stream:
        stream.write(b"0")
        stream.flush()
        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as error:
            raise RuntimeError("Другой вычислитель уже работает с КОМПАС") from error
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def connect(bin_dir: Path) -> tuple[Any, Any]:
    if os.name != "nt":
        raise RuntimeError("Вычислитель KsAPI требует Windows")
    for name in ("ksapi.py", "constants.py", "constants3d.py", "ksAPICLink.dll"):
        if not (bin_dir / name).is_file():
            raise FileNotFoundError(bin_dir / name)
    command = (
        "@(Get-Process kStudy,KOMPAS -ErrorAction SilentlyContinue | "
        "Select-Object -ExpandProperty Path) | ConvertTo-Json -Compress"
    )
    process = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
        capture_output=True,
        text=True,
        check=False,
    )
    paths = json.loads(process.stdout.strip() or "[]")
    if isinstance(paths, str):
        paths = [paths]
    if len(paths) != 1 or Path(paths[0]).parent.resolve() != bin_dir.resolve():
        raise RuntimeError(f"Нужен один КОМПАС из подтверждённого Bin; запущено: {paths}")
    sys.path.insert(0, str(REPO / "vendor/kompas_lib"))
    import kompas_lib as lib

    app = lib.Kompas.connect(start=False, bin_dir=str(bin_dir))
    if "25" not in app.version:
        raise RuntimeError(f"Протокол требует КОМПАС v25, подключено: {app.version}")
    return app, lib


def mass_parameters(app: Any, lib: Any, part: Any) -> Any:
    mi = app.ks.IMassInertiaParam(part)
    mi.SetLengthUnits(lib.by_name("ksLUnMM"))
    mi.SetMassUnits(lib.by_name("ksMUnGR"))
    mi.SetSourceData(False)
    mi.SetCopySourceData(False)
    mi.SetUseMassFromSource(False)
    mi.SetManualMassCentre(False)
    if not mi.Calculate():
        raise RuntimeError("Не удалось вычислить МЦХ")
    return mi


def input_path(job: Path, request: dict, name: str) -> Path:
    record = request["inputs"][name]
    path = (job / record["path"]).resolve()
    if not path.is_relative_to(job.resolve()) or sha256(path) != record["sha256"]:
        raise ValueError(f"Некорректная копия исходной модели: {name}")
    return path


@contextmanager
def open_copy(app: Any, path: Path) -> Iterator[Any]:
    if app.find_open(path) is not None:
        raise RuntimeError(f"Копия уже открыта; чужой документ не трогаем: {path}")
    doc = app.open(path, read_only=True, visible=False)
    try:
        yield doc
    finally:
        doc.close(save=False)


def export_model(app: Any, lib: Any, job: Path, request: dict) -> dict:
    source = input_path(job, request, "model")
    ks = app.ks
    with open_copy(app, source) as doc:
        feature = ks.IFeature(doc.part)
        bodies = feature.GetResultBodies() or []
        if not bodies:
            raise ValueError("В M3D нет тел для выгрузки")
        metadata = {
            "author": doc.raw.GetAuthor(),
            "created": doc.raw.GetCreationDate(),
            "changed": doc.raw.GetLastChangeDate(),
            "comment": doc.raw.GetComment(),
        }
        tree = [
            {
                "name": f.GetName(),
                "type": ks.IModelObject(f).GetModelObjectType(),
                "excluded": f.IsExcluded(),
                "error": f.GetObjectError(),
            }
            for f in feature.GetSubFeatures(1, True, False) or []
        ]
        mi = mass_parameters(app, lib, doc.part)
        axes = [mi.GetAxisX(0.0, 0.0, 0.0), mi.GetAxisY(0.0, 0.0, 0.0), mi.GetAxisZ(0.0, 0.0, 0.0)]
        moments = [mi.GetJx0(), mi.GetJy0(), mi.GetJz0()]
        version = f"{app.version}; build {app.build}"
        raw = {
            "version": version,
            "metadata": metadata,
            "tree": tree,
            "length_units": mi.GetLengthUnits(),
            "mass_units": mi.GetMassUnits(),
            "axes": axes,
            "principal_moments": moments,
            "central_moments": [mi.GetJx(), mi.GetJy(), mi.GetJz(), mi.GetJxy(), mi.GetJxz(), mi.GetJyz()],
            "bbox": doc.part.GetGabarit(False, False, 0, 0, 0, 0, 0, 0),
        }
        facts = {
            "source": "kompas",
            "source_version": version,
            "native_sha256": sha256(source),
            "mesh_units": "mm",
            "units": "mm",
            "body_count": len(bodies),
            "solid_valid": all(ks.IBody(b).IsSolid() for b in bodies)
            and not feature.GetObjectError()
            and not any(t["error"] for t in tree if not t["excluded"]),
            "volume_mm3": mi.GetVolume(),
            "area_mm2": mi.GetArea(),
            "mass_g": mi.GetMass(),
            "center_mm": [mi.GetXc(), mi.GetYc(), mi.GetZc()],
            "principal_axes": [[axes[column][row + 1] for column in range(3)] for row in range(3)]
            if all(axis[0] for axis in axes)
            else None,
            "principal_moments": moments,
            "inertia_unit": "g_mm2",
            "tree_fingerprint": digest(tree) if tree else None,
            "document_fingerprint": digest(metadata) if metadata["created"] > 0 else None,
            "topology": {},
            "dimensions_mm": {},
        }
        for label, constant in (("faces", "o3d_face"), ("edges", "o3d_edge"), ("vertices", "o3d_vertex")):
            facts["topology"][label] = sum(
                len(ks.IFeature(body).GetModelObjects([lib.by_name(constant)]) or []) for body in bodies
            )
        # Обёртка измеряет геометрию; исходные ответы сохраняем отдельно от DTO Facts.
        try:
            measures = lib.measure_model(doc)
            raw["measures"] = {}
            for kind in ("diameter", "radius", "linear", "thread", "angle"):
                items = measures.items(kind)
                raw["measures"][kind] = [asdict(item) for item in items]
                if kind != "angle":
                    facts["dimensions_mm"][kind] = [item.value for item in items for _ in range(item.same)]
        except Exception as error:
            raw["measurements_error"] = f"{type(error).__name__}: {error}"
        profile = request["profile"]
        converter = app.raw.GetConverter(lib.by_name("ksConverterToSTL"))
        params = ks.IAdditionConvertParameters(
            converter.GetConverterParameters(lib.by_name("ksCMFileSaveAs"))
        )
        params.SetFormat(lib.by_name("ksConverterToSTL"))
        params.SetFormatBinary(True)
        params.SetLengthUnits(lib.by_name("ksLUnMM"))
        params.SetStepType(lib.by_name("ksSpaceStep") | lib.by_name("ksDeviationStep"))
        params.SetStep(profile["chord_mm"])
        params.SetAngle(profile["angle"])
        params.SetMaxTesselationCellCount(profile["max_cells"])
        params.SetExportObjects(bodies)
        path = job / "model.pending.stl"
        with app.quiet():
            success = ks.IKompasDocument3D(doc.raw).ConvertToAdditionFormat(str(path), params)
        if not path.is_file() or path.stat().st_size < 84:
            raise RuntimeError(f"STL не создан; код экспортёра: {success}")
        path.replace(job / "model.stl")
        path = job / "model.stl"
        raw["export_returned"] = success
        facts["mesh_sha256"] = sha256(path)
        write_json(job / "raw-facts.json", raw)
        write_json(job / "kernel-facts.json", facts)
        write_json(
            job / "export-source.json",
            {
                "profile": profile,
                "implementation": request["implementation"],
                "native_sha256": facts["native_sha256"],
                "source_version": version,
                "files": {
                    name: sha256(job / name) for name in ("model.stl", "kernel-facts.json", "raw-facts.json")
                },
            },
        )
    return {"status": "ok", "source_version": version}


def intersect(app: Any, lib: Any, job: Path, request: dict) -> dict:
    from contextlib import ExitStack

    solid = request["solid_request"]
    if (
        digest({key: value for key, value in solid.items() if key != "request_sha256"})
        != solid["request_sha256"]
    ):
        raise ValueError("Хеш запроса пересечения не совпал")
    matrix = solid["prediction_to_reference"]
    ks = app.ks
    result = {
        "request_sha256": solid["request_sha256"],
        "backend": "kompas_v25",
        "backend_version": f"{app.version}; build {app.build}",
    }
    with ExitStack() as stack:
        documents = {}
        volumes = {}
        for role in ("prediction", "reference"):
            path = input_path(job, request, role)
            if sha256(path) != solid[f"{role}_native_sha256"]:
                raise ValueError("Запрос пересечения относится к другой M3D")
            documents[role] = stack.enter_context(open_copy(app, path))
            volumes[role] = mass_parameters(app, lib, documents[role].part).GetVolume()
        scratch = app.new_part(visible=False)
        stack.callback(scratch.close, save=False)
        container = ks.IModelContainer(scratch.part)
        scratch.activate()
        for role, doc in documents.items():
            bodies = ks.IFeature(doc.part).GetResultBodies() or []
            if len(bodies) != 1:
                raise ValueError("Первая версия пересечения поддерживает одно тело в каждой модели")
            copy = container.GetCopiesGeometry().Add()
            copy.SetContextObjects(False)
            copy.SetAutoUpdate(False)
            copy.SetWatchForSourceChange(False)
            if not copy.AddInitialObjectsFromExternalDocument(ks.IKompasDocument3D(doc.raw), None, bodies):
                raise RuntimeError("Не удалось передать тело в рабочий документ")
            if not copy.Update() or ks.IFeature(copy).GetObjectError():
                raise RuntimeError("Не удалось скопировать тело")
            if role == "prediction":
                body = (ks.IFeature(scratch.part).GetResultBodies() or [])[0]
                move = container.GetBodyRepositions().Add()
                move.SetRepositionBody(body)
                move.SetCopyBody(False)
                if not move.GetPosition().InitByMatrix3D([matrix[j][i] for i in range(4) for j in range(4)]):
                    raise RuntimeError("Матрица совмещения отклонена")
                if not move.Update() or ks.IFeature(move).GetObjectError():
                    raise RuntimeError("Перемещение тела не выполнено")
        bodies = ks.IFeature(scratch.part).GetResultBodies() or []
        if len(bodies) != 2:
            raise RuntimeError("Ожидалось два тела перед пересечением")
        total = mass_parameters(app, lib, scratch.part).GetVolume()
        if not math.isclose(total, sum(volumes.values()), rel_tol=1e-7, abs_tol=1e-5):
            raise RuntimeError("Объёмы скопированных тел не совпали с исходными")
        boxes = [ks.IBody(body).GetGabarit(0.0, 0.0, 0.0, 0.0, 0.0, 0.0) for body in bodies]
        if not all(box[0] for box in boxes):
            raise RuntimeError("Не удалось проверить габариты тел")
        expected = request["prediction_bounds_mm"]
        distances = [max(abs(a - b) for a, b in zip(box[1:], expected, strict=True)) for box in boxes]
        if min(distances) > request["bbox_tolerance_mm"]:
            raise RuntimeError("Преобразование тела не совпало с преобразованием сетки")
        write_json(job / "transform-check.json", {"boxes": boxes, "expected": expected, "errors": distances})
        # Разнесённые точные B-rep габариты доказывают пустое пересечение без Update.
        disjoint = any(
            boxes[0][i + 4] < boxes[1][i + 1] or boxes[1][i + 4] < boxes[0][i + 1] for i in range(3)
        )
        if disjoint:
            intersection = 0.0
        else:
            boolean = container.GetBooleans().Add()
            boolean.SetBooleanType(lib.by_name("ksIntersect"))
            boolean.SetBodies(bodies)
            boolean.SetBaseObject(bodies[0])
            boolean.SetModifyObjects([bodies[1]])
            boolean.SetSaveCopyBaseObject(False)
            boolean.SetSaveCopyModifyObjects(False)
            with app.quiet():
                updated = boolean.Update()
            error = ks.IFeature(boolean).GetObjectError()
            remaining = ks.IFeature(scratch.part).GetResultBodies() or []
            if not updated or error:
                raise RuntimeError(f"Ошибка булева пересечения: Update={updated}, error={error}")
            intersection = mass_parameters(app, lib, scratch.part).GetVolume() if remaining else 0.0
        if not math.isfinite(intersection) or not 0 <= intersection <= min(volumes.values()):
            raise RuntimeError(f"Объём пересечения вне допустимого диапазона: {intersection}")
        result.update(
            status="ok",
            prediction_volume_mm3=volumes["prediction"],
            reference_volume_mm3=volumes["reference"],
            intersection_volume_mm3=intersection,
        )
    return {"status": "ok", "result": result}


def main() -> int:
    job, bin_dir = map(Path, sys.argv[1:3])
    try:
        request = json.loads((job / "request.json").read_text(encoding="utf-8"))
        actual = {
            "worker_sha256": sha256(Path(__file__)),
            "vendor_sha256": sha256(REPO / "vendor/kompas_lib/kompas_lib.py"),
            "normalizer_sha256": sha256(Path(__file__).with_name("normalize.py")),
        }
        if request["implementation"] != actual:
            raise ValueError("Версии вычислителя на двух машинах не совпадают")
        with exclusive_session():
            app, lib = connect(bin_dir)
            original = app.active_document
            try:
                action = request["action"]
                if action == "health":
                    response = {"status": "ok", "version": app.version, "build": app.build}
                elif action == "export":
                    response = export_model(app, lib, job, request)
                elif action == "intersect":
                    response = intersect(app, lib, job, request)
                else:
                    raise ValueError(f"Неизвестное действие: {action}")
            finally:
                if original is not None:
                    original.activate()
    except Exception as error:
        write_json(
            job / "response.json",
            {
                "status": "error",
                "reason": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            },
        )
        return 2
    write_json(job / "response.json", response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

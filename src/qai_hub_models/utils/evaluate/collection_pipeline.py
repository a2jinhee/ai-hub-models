# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""End-to-end evaluate pipeline for collection models."""

from __future__ import annotations

import argparse
import importlib

import qai_hub as hub

from qai_hub_models import Precision
from qai_hub_models.utils.args import get_model_kwargs
from qai_hub_models.utils.base_app import CollectionAppEvaluateProtocol
from qai_hub_models.utils.evaluate.helpers import (
    _load_quant_cpu_onnx,
    evaluate_on_dataset,
)
from qai_hub_models.utils.inference import AsyncOnDeviceModel, compile_model_from_args
from qai_hub_models.utils.input_spec import InputSpec


def evaluate_model(
    model_id: str,
    args: argparse.Namespace,
    supports_quant_cpu: bool = False,
) -> None:
    """
    Evaluate a collection model's accuracy on a dataset.

    Parameters
    ----------
    model_id
        Model folder name (e.g. ``mediapipe_hand``).
    args
        Parsed argparse.Namespace from :func:`evaluate_parser`.
    supports_quant_cpu
        If True and precision != float, adds a "quant cpu" executor via
        :func:`_load_quant_cpu_onnx` for the QDQ ONNX accuracy check.
    """
    model_module = importlib.import_module(f"qai_hub_models.models.{model_id}")
    model_cls = model_module.Model
    app_cls = model_module.App

    model_kwargs = get_model_kwargs(model_cls, vars(args))

    eval_dataset_classes = model_cls.get_eval_dataset_classes()
    if len(eval_dataset_classes) == 0:
        # PSNR fallback: no datasets, just run export with inference enabled
        print(
            "Model does not have evaluation dataset specified. Evaluating PSNR on a single sample."
        )
        export_module = importlib.import_module(
            f"qai_hub_models.models.{model_id}.export"
        )
        export_model = export_module.export_model

        kwargs = {
            "device": args.device,
            "target_runtime": args.target_runtime,
            "skip_downloading": True,
            "skip_profiling": True,
            "compile_options": args.compile_options,
            "profile_options": args.profile_options,
            **model_kwargs,
        }
        num_calibration_samples = getattr(args, "num_calibration_samples", None)
        if num_calibration_samples is not None:
            kwargs["num_calibration_samples"] = num_calibration_samples

        export_model(model_id, **kwargs)
        return

    # Verify App implements the protocol (runtime check only - mypy can't verify)
    if not (isinstance(app_cls, type) and hasattr(app_cls, "from_components")):
        raise TypeError(
            f"App must implement CollectionAppEvaluateProtocol when eval_datasets is specified, got {app_cls}"
        )

    if args.use_dataset_cache:
        raise ValueError("Collection models do not support use_dataset_cache.")

    collection_model = model_cls.from_pretrained(**model_kwargs)
    num_components = len(collection_model.component_names)

    input_spec: InputSpec | None = None
    torch_model_list = list(collection_model.components.values())
    model_executors: dict[str, CollectionAppEvaluateProtocol] = {}
    on_device_model_list: list[AsyncOnDeviceModel] = []

    if not args.skip_torch_accuracy:
        model_executors["torch"] = app_cls.from_components(torch_model_list)
        input_spec = torch_model_list[0].get_input_spec()

    # On-device or quant-cpu path
    if not args.skip_device_accuracy or (
        supports_quant_cpu and getattr(args, "compute_quant_cpu_accuracy", False)
    ):
        if args.hub_model_id is not None:
            hub_model_id_list = args.hub_model_id.split(",")
            assert len(hub_model_id_list) == num_components, (
                f"Number of hub_model_ids ({len(hub_model_id_list)}) must equal "
                f"number of components ({num_components})"
            )
            compiled_model_list = [
                hub.get_model(model_id) for model_id in hub_model_id_list
            ]
        else:
            compiled_result = compile_model_from_args(
                model_id,
                args,
                model_kwargs,
            )
            assert isinstance(compiled_result, list)
            compiled_model_list = compiled_result

        for compiled_model in compiled_model_list:
            if compiled_model.get_producer() is None:
                raise ValueError(
                    "Compiled models must be compiled with AI Hub Workbench; they cannot be uploaded manually."
                )
            on_device_model_list.append(
                AsyncOnDeviceModel(
                    model=compiled_model,
                    input_names=None,
                    device=args.device,
                    inference_options=args.profile_options,
                )
            )

        if not args.skip_device_accuracy:
            model_executors["on-device"] = app_cls.from_components(on_device_model_list)

        if (
            supports_quant_cpu
            and getattr(args, "compute_quant_cpu_accuracy", False)
            and args.precision != Precision.float
        ):
            quant_cpu_model_list = [
                _load_quant_cpu_onnx(model) for model in compiled_model_list
            ]
            model_executors["quant cpu"] = app_cls.from_components(quant_cpu_model_list)

        input_spec = on_device_model_list[0].get_input_spec()

    if input_spec is None:
        raise ValueError("Cannot extract input spec.")

    evaluate_on_dataset(
        evaluator_func=collection_model.get_evaluator,
        dataset_cls=args.dataset_cls,
        model_executors=model_executors,
        input_spec=input_spec,
        samples_per_job=args.samples_per_job,
        num_samples=args.num_samples,
        seed=args.seed,
        use_cache=args.use_dataset_cache,
    )

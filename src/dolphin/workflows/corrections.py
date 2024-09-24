from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

from opera_utils import group_by_date

from dolphin import io, utils
from dolphin._log import log_runtime, setup_logging
from dolphin._types import PathOrStr
from dolphin.timeseries import ReferencePoint

from ._utils import parse_ionosphere_files
from .config import CorrectionOptions, DisplacementWorkflow

logger = logging.getLogger(__name__)


class CorrectionPaths(NamedTuple):
    """Named tuple of corrections workflow outputs."""

    tropospheric_corrections: list[Path] | None
    ionospheric_corrections: list[Path] | None


@log_runtime
def run(
    cfg: DisplacementWorkflow,
    correction_options: CorrectionOptions,
    unwrapped_paths: Sequence[PathOrStr],
    out_dir: Path = Path(),
    reference_point: ReferencePoint | None = None,
    log_file: PathOrStr | None = None,
    debug: bool = False,
) -> CorrectionOptions:
    """Run the corrections workflow on a stack of SLCs.

    Parameters
    ----------
    cfg : DisplacementWorkflow
        [`DisplacementWorkflow`][dolphin.workflows.config.DisplacementWorkflow] object
        for controlling the workflow.
    debug : bool, optional
        Enable debug logging, by default False.

    """
    setup_logging(logger_name="dolphin", debug=debug, filename=log_file)
    if len(correction_options.geometry_files) > 0:
        raise ValueError("No geometry files passed to run the corrections workflow")

    tropo_paths: list[Path] | None = None
    iono_paths: list[Path] | None = None

    grouped_iono_files = parse_ionosphere_files(
        correction_options.ionosphere_files, correction_options._iono_date_fmt
    )
    out_dir.mkdir(exist_ok=True)
    grouped_slc_files = group_by_date(cfg.cslc_file_list)

    # Prepare frame geometry files
    geometry_dir = out_dir / "geometry"
    geometry_dir.mkdir(exist_ok=True)
    assert unwrapped_paths is not None
    crs = io.get_raster_crs(unwrapped_paths[0])
    epsg = crs.to_epsg()
    out_bounds = io.get_raster_bounds(unwrapped_paths[0])
    frame_geometry_files = utils.prepare_geometry(
        geometry_dir=geometry_dir,
        geo_files=correction_options.geometry_files,
        matching_file=unwrapped_paths[0],
        dem_file=correction_options.dem_file,
        epsg=epsg,
        out_bounds=out_bounds,
        strides=cfg.output_options.strides,
    )

    # Troposphere
    if "height" not in frame_geometry_files:
        logger.warning(
            "DEM file is not given, skip estimating tropospheric corrections."
        )
    else:
        if correction_options.troposphere_files:
            from dolphin.atmosphere import estimate_tropospheric_delay

            assert unwrapped_paths is not None
            logger.info(
                "Calculating tropospheric corrections for %s files.",
                len(unwrapped_paths),
            )
            tropo_paths = estimate_tropospheric_delay(
                ifg_file_list=unwrapped_paths,
                troposphere_files=correction_options.troposphere_files,
                file_date_fmt=correction_options.tropo_date_fmt,
                slc_files=grouped_slc_files,
                geom_files=frame_geometry_files,
                reference_point=reference_point,
                output_dir=out_dir,
                tropo_model=correction_options.tropo_model,
                tropo_delay_type=correction_options.tropo_delay_type,
                epsg=epsg,
                bounds=out_bounds,
            )
        else:
            logger.info("No weather model, skip tropospheric correction.")

    # Ionosphere
    if grouped_iono_files:
        from dolphin.atmosphere import estimate_ionospheric_delay

        logger.info(
            "Calculating ionospheric corrections for %s files",
            len(unwrapped_paths),
        )
        assert unwrapped_paths is not None
        iono_paths = estimate_ionospheric_delay(
            ifg_file_list=unwrapped_paths,
            slc_files=grouped_slc_files,
            tec_files=grouped_iono_files,
            geom_files=frame_geometry_files,
            reference_point=reference_point,
            output_dir=out_dir,
            epsg=epsg,
            bounds=out_bounds,
        )
    else:
        logger.info("No TEC files, skip ionospheric correction.")

    return CorrectionPaths(
        ionospheric_corrections=iono_paths, tropospheric_corrections=tropo_paths
    )

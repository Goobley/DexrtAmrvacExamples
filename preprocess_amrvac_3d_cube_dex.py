import argparse
from pathlib import Path
from tqdm import tqdm
import yt
from yt.units import Mm, Unit
from yt_experiments.tiled_grid import YTTiledArbitraryGrid
import xarray as xr
import numpy as np
import lightweaver as lw
import promweaver as pw
import astropy.constants as const

from dexrt.config_schemas.dexrt import DexrtNonLteConfig, AtomicModelConfig, DexrtSystemConfig
from dexrt import write_config
from dexrt.sparse_data.morton import encode_morton_3, decode_morton_3

DTYPE = np.float32

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="preprocess_3d_cube",
        description="Prep a 3D AMRVAC model for dex synthesis"
    )
    parser.add_argument(
        "--path",
        dest="cube_path",
        help="Input file",
        metavar="FILE",
        type=Path
    )
    parser.add_argument(
        "--output",
        dest="output_path",
        help="Output file",
        metavar="FILE",
        type=Path
    )
    parser.add_argument(
        "--sparse",
        dest="sparse",
        help="Whether to write a sparse atmosphere file",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--sparse_block_size",
        dest="block_size",
        help="Block size to use for sparse atmosphere",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--sparse_temperature_threshold",
        dest="temperature_threshold",
        help="Upper temperature threshold for sparse atmosphere",
        type=float,
        default=250e3
    )
    parser.add_argument(
        "--scale",
        dest="scale",
        help="Rescale grid dimensions from finest amr resolution (e.g. 2 for double resolution)",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--pad_empty_cells",
        dest="pad_cells",
        help="Pad this many empty cells around the edge of the domain (top and bottom), e.g. 8 16 8. Anticipated to be a multiple of the block_size. x, y, z, dex order.",
        type=int,
        nargs='+',
        default=[0, 0, 0]
    )

    args = parser.parse_args()

    base_altitude = 10e6

    # Standard way of reading in the .dat files using yt after having specified the normalisation constants
    overrides = dict(length_unit=(1e8, 'cm'), temperature_unit=(1e6, 'K'), numberdensity_unit=(1e9,'cm**-3'))
    ds=yt.load(args.cube_path, units_override=overrides) #Need the e_is_internal flag here to use polytropic EoS as sim conserves only internal, not total energy. Needs a tweak to the base yt fields.py to accept additional keyword
    ds._e_is_internal = True


    def vturb_fn(T, epsilon=0.5, alpha=0.1, i=0, gamma=5/3, mH=1.6735575e-27):
        k = 1.380649e-23
        m = (1 + 4*alpha)/(1 + alpha + i) * mH
        return epsilon * np.sqrt(gamma * k * T / m)

    # Ascertain the dimensions of the specific simulation
    max_steps = (ds.domain_dimensions*ds.refine_by**ds.max_level) # Resolution of entire domain if at highest AMR level
    max_res_u = (ds.domain_width/max_steps) # Stepsize dx dy dz if at max AMR res
    max_res = (ds.domain_width/max_steps).to('m').value # Stepsize dx dy dz if at max AMR res

    # NOTE(cmo): My original run
    # xbounds = [-8, 8] * Mm
    # ybounds = [ds.domain_left_edge[1].value,30] * Mm
    # zbounds = [-512,512] * max_res[2] * Unit('m')
    # dims = [384, 720, 1024]


    # NOTE(cmo): Jack's run
    # xbounds = [-7.5,7.5] * Mm
    # ybounds = [2,30] * Mm
    # zbounds = [-20,20] * Mm
    # dims = [int(((b[1] - b[0]) / max_res).value) for b, max_res in zip([xbounds, ybounds, zbounds], max_res_u)]

    # NOTE(cmo): Adjusted original based on Jack's run
    xbounds = [-8, 8] * Mm
    ybounds = [2, 30] * Mm
    zbounds = [-512, 512] * max_res_u[2].to('Mm') # [-21.333333, 21.333333] * Mm
    dims = [int(((b[1] - b[0]) / max_res).value) for b, max_res in zip([xbounds, ybounds, zbounds], max_res_u)]

    # NOTE(cmo): Mini testing model
    # xbounds = [-8 * Mm + (192 - 32) * max_res_u[0], -8 * Mm + (192 + 32) * max_res_u[0]]
    # ybounds = [ds.domain_left_edge[1] + (250 - 64) * max_res_u[1], ds.domain_left_edge[1] + (250 + 64) * max_res_u[1]]
    # zbounds = [(368 - 128) * max_res_u[2], (368 + 128) * max_res_u[2]]
    # dims = [64, 128, 256]

    dims = [int(d * args.scale) for d in dims]
    for d in dims:
        if d / args.block_size != d // args.block_size:
            raise ValueError("Grid size is not a multiple of block size")

    grid_kwargs = {
        "left_edge": [xbounds[0], ybounds[0], zbounds[0]],
        "right_edge": [xbounds[1], ybounds[1], zbounds[1]],
        "dims": dims
    }
    amrvac_padding = [args.pad_cells[0], args.pad_cells[2], args.pad_cells[1]]
    dims_with_padding = [d + 2 * p for d, p in zip(dims, amrvac_padding)]
    print(f"dims with padding: {dims_with_padding} (amrvac order)")

    atmos_params = {}
    ds_attrs = {}
    if args.sparse:
        tag = YTTiledArbitraryGrid(
            grid_kwargs['left_edge'],
            grid_kwargs['right_edge'],
            grid_kwargs['dims'],
            args.block_size,
            ds=ds
        )
        num_blocks = [d // args.block_size for d in dims]
        lower_pad_blocks = [p // args.block_size for p in args.pad_cells]
        morton_order = []
        # NOTE(cmo): Here x, y, and z refer to dex, and are mapping from the
        # amrvac layout (stored as [x, y, z], but meaning [x, z, y] in dex)
        # Here we add the lower padding onto each coord. Those blocks are forced
        # to be empty (i.e. will not even be checked in the amrvac data).
        for z in range(num_blocks[1]):
            for y in range(num_blocks[2]):
                for x in range(num_blocks[0]):
                    morton_order.append(encode_morton_3(
                        np.int32(x + lower_pad_blocks[0]),
                        np.int32(y + lower_pad_blocks[1]),
                        np.int32(z + lower_pad_blocks[2])))
        morton_order = sorted(morton_order)

        temperature_blocks = []
        pressure_blocks = []
        vx_blocks = []
        vy_blocks = []
        vz_blocks = []

        assert args.temperature_threshold != 0.0

        active_blocks = []
        temperature_threshold = args.temperature_threshold
        for code in tqdm(morton_order):
            dex_x, dex_y, dex_z = decode_morton_3(code)
            # NOTE(cmo): Shift back by lower_pad_blocks
            amrvac_grid_coord = [
                dex_x - lower_pad_blocks[0],
                dex_z - lower_pad_blocks[2],
                dex_y - lower_pad_blocks[1],
            ]
            amrvac_grid_idx = np.ravel_multi_index(amrvac_grid_coord, num_blocks)

            block = tag.single_arbitrary_grid(amrvac_grid_idx)
            if np.any(block['temperature'].to('K').value < temperature_threshold):
                active_blocks.append(code)

                temperature_blocks.append(block['temperature'].to('K').value.squeeze().astype(DTYPE))
                pressure_blocks.append(block[('gas', 'thermal_pressure')].to('Pa').value.squeeze().astype(DTYPE))
                vx_blocks.append(block['velocity_1'].to('m/s').value.squeeze().astype(DTYPE))
                vy_blocks.append(block['velocity_2'].to('m/s').value.squeeze().astype(DTYPE))
                vz_blocks.append(block['velocity_3'].to('m/s').value.squeeze().astype(DTYPE))

        temperature = np.concatenate([np.ascontiguousarray(x.transpose(1, 2, 0)).reshape(-1) for x in temperature_blocks])
        pressure = np.concatenate([np.ascontiguousarray(x.transpose(1, 2, 0)).reshape(-1) for x in pressure_blocks])
        v_x = np.concatenate([np.ascontiguousarray(x.transpose(1, 2, 0)).reshape(-1) for x in vx_blocks])
        v_y = np.concatenate([np.ascontiguousarray(x.transpose(1, 2, 0)).reshape(-1) for x in vy_blocks])
        v_z = np.concatenate([np.ascontiguousarray(x.transpose(1, 2, 0)).reshape(-1) for x in vz_blocks])
        field_shape = ("cells",)
        active_blocks = np.array(active_blocks, dtype=np.uint32)

        atmos_params = atmos_params | dict(
            morton_tiles=(["num_active_tiles"], active_blocks),
        )
        ds_attrs = ds_attrs | dict(
            num_x=dims_with_padding[0],
            num_x_blocks=num_blocks[0] + 2 * lower_pad_blocks[0],
            num_y=dims_with_padding[2],
            num_y_blocks=num_blocks[2] + 2 * lower_pad_blocks[1],
            num_z=dims_with_padding[1],
            num_z_blocks=num_blocks[1] + 2 * lower_pad_blocks[2],
            block_size=args.block_size,
            program='dexrt (3d)', # masquerade
            output_format='sparse',
            sparse=1,
        )
    else:
        grid = ds.arbitrary_grid(**grid_kwargs)
        pressure = grid[('gas', 'thermal_pressure')].to('Pa').transpose(1, 2, 0).value.squeeze().astype(DTYPE)
        temperature = grid['temperature'].to('K').transpose(1, 2, 0).value.squeeze().astype(DTYPE)
        v_x = grid['velocity_1'].to('m/s').transpose(1, 2, 0).value.squeeze().astype(DTYPE)
        v_y = grid['velocity_2'].to('m/s').transpose(1, 2, 0).value.squeeze().astype(DTYPE)
        v_z = grid['velocity_3'].to('m/s').transpose(1, 2, 0).value.squeeze().astype(DTYPE)
        if max(args.pad_cells) > 0:
            padded_size = (dims_with_padding[1], dims_with_padding[2], dims_with_padding[0])
            def pad_arr(x):
                lb = [p if p != 0 else None for p in args.pad_cells]
                ub = [-p if p != 0 else None for p in args.pad_cells]
                new_x = np.zeros(padded_size, dtype=DTYPE)
                new_x[lb[2]:ub[2], lb[1]:ub[1], lb[0]:ub[0]] = x
                return new_x
            pressure = pad_arr(pressure)
            temperature = pad_arr(temperature)
            # NOTE(cmo): Disable these cells
            temperature[temperature == 0.0] = 2 * args.temperature_threshold
            v_x = pad_arr(v_x)
            v_y = pad_arr(v_y)
            v_z = pad_arr(v_z)
        field_shape = ("z", "y", "x")

    vturb = vturb_fn(temperature).astype(DTYPE)

    initial_ionisation_fraction = 0.8
    n_tot = pressure / (const.k_B.value * temperature)
    nh_tot = n_tot / (lw.DefaultAtomicAbundance.totalAbundance * (1.0 + initial_ionisation_fraction))
    nh_tot = nh_tot.astype(DTYPE)
    ne = lw.DefaultAtomicAbundance.totalAbundance * nh_tot * initial_ionisation_fraction
    ne = ne.astype(DTYPE)
    nh_tot[np.isnan(nh_tot)] = 0.0
    ne[np.isnan(ne)] = 0.0

    assert abs(max_res[0] - max_res[1]) < 1e-2, "Check aspect ratio"
    assert abs(max_res[0] - max_res[2]) < 1e-2, "Check aspect ratio"
    voxel_scale = np.array([max_res[0]], dtype=DTYPE)

    offset_z = np.array([base_altitude], dtype=DTYPE)
    offset_y = np.array([-0.5 * dims_with_padding[2] * voxel_scale.item()], dtype=DTYPE)
    offset_x = np.array([-0.5 * dims_with_padding[0] * voxel_scale.item()], dtype=DTYPE)

    atmos_params = atmos_params | dict(
        temperature=(field_shape, temperature),
        pressure=(field_shape, pressure),
        vx=(field_shape, v_x),
        vy=(field_shape, v_z),
        vz=(field_shape, v_y),
        nh_tot=(field_shape, nh_tot),
        ne=(field_shape, ne),
        vturb=(field_shape, vturb),
        voxel_scale=voxel_scale,
        offset_x=offset_x,
        offset_y=offset_y,
        offset_z=offset_z,
    )

    bc_ctx = pw.compute_falc_bc_ctx(active_atoms=["H", "Ca"])
    tabulated = pw.tabulate_bc(bc_ctx, mu_grid=np.linspace(0.05, 1.0, 20))
    I_with_zero = np.zeros((tabulated["I"].shape[0], tabulated["I"].shape[1] + 1))
    I_with_zero[:, 1:] = tabulated["I"][...]
    tabulated["I"] = I_with_zero
    tabulated["mu_grid"] = np.concatenate([[0], tabulated["mu_grid"]])

    bc_wavelength = tabulated["wavelength"].astype(DTYPE)
    bc_I = np.zeros(I_with_zero.shape, dtype=DTYPE)
    for la in range(tabulated["wavelength"].shape[0]):
        bc_I[la, :] = lw.convert_specific_intensity(
            tabulated["wavelength"][la],
            tabulated["I"][la, :],
            outUnits="kW / (m2 nm sr)"
        ).value

    bc_params = dict(
        prom_bc_I=(["prom_bc_wavelength", "prom_bc_mu"], bc_I),
        prom_bc_wavelength=(["prom_bc_wavelength"], bc_wavelength),
        prom_bc_mu_min=np.array([tabulated["mu_grid"][0]], dtype=DTYPE),
        prom_bc_mu_max=np.array([tabulated["mu_grid"][-1]], dtype=DTYPE),
    )

    ds = xr.Dataset(
        data_vars=atmos_params | bc_params,
        attrs=ds_attrs
    )
    ds.to_netcdf(args.output_path)


    config = DexrtNonLteConfig(
        atmos_path=str(args.output_path),
        output_path=str(args.output_path.parent / args.output_path.stem) + "_out.nc",
        atoms={
            "H": AtomicModelConfig(
                path="H_4.yaml"
            ),
            "Ca": AtomicModelConfig(
                path="CaII.yaml"
            ),
        },
        boundary_type="Promweaver",
        sparse_calculation=True,
        conserve_charge=True,
        conserve_pressure=True,
        system=DexrtSystemConfig(
            mem_pool_gb=8.0,
        ),
        rad_loss="None",
    )
    write_config(config, str(args.output_path.parent / args.output_path.stem) + ".yaml")

import argparse
from pathlib import Path
from tqdm import tqdm
import yt
from yt_experiments.tiled_grid import YTTiledArbitraryGrid
import xarray as xr
import numpy as np
import lightweaver as lw
import promweaver as pw
import astropy.constants as const

from dexrt.config_schemas.dexrt import DexrtNonLteConfig, AtomicModelConfig, DexrtSystemConfig
from dexrt import write_config
from dexrt.sparse_data.morton import encode_morton_2, decode_morton_2

DTYPE = np.float32
ZERO_VY = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="preprocess_2d_cube",
        description="Prep a 2D AMRVAC model for dex synthesis"
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
        default=16,
    )
    parser.add_argument(
        "--sparse_temperature_threshold",
        dest="temperature_threshold",
        help="Upper temperature threshold for sparse atmosphere",
        type=float,
        default=250e3,
    )
    parser.add_argument(
        "--scale",
        dest="scale",
        help="Rescale grid dimensions from finest amr resolution (e.g. 2 for double resolution)",
        type=float,
        default=1.0,
    )

    args = parser.parse_args()

    base_altitude = 10e6
    units = {
        "length_unit": (1e8, "cm"),
        "temperature_unit": (1e6, "K"),
        "numberdensity_unit": (1e8, "cm**-3")
    }
    unit_velocity =    11645084.295622544
    unit_temperature = 1000000.00000
    unit_pressure =    3.1754922399999996E-002
    # NOTE(cmo): The unit pressure nonsense is only needed because this file was
    # likely computed with _e_is_internal, but I didn't know that in the
    # original script.

    def temperature_(field, data):
        Te = data['Te'].value * data.ds.temperature_unit
        return Te

    def vturb_fn(T, epsilon=0.5, alpha=0.1, i=0, gamma=5/3, mH=1.6735575e-27):
        k = 1.380649e-23
        m = (1 + 4*alpha)/(1 + alpha + i) * mH
        return epsilon * np.sqrt(gamma * k * T / m)

    ds = yt.load(args.cube_path, units_override=units, unit_system="cgs")
    ds.add_field(("gas", "temperature"), function=temperature_, units="K", sampling_type="cell", force_override=True)

    x_left = -24
    x_right = 24
    y_bottom = 0
    y_top = 40

    # Resolution
    nyy = int(1280 * args.scale)
    nxx = int(1536 * args.scale)

    # Ascertain the dimensions of the specific simulation
    max_steps = (ds.domain_dimensions*ds.refine_by**ds.max_level) # Resolution of entire domain if at highest AMR level
    max_res = (ds.domain_width/max_steps).to('m').value # Stepsize dx dy dz if at max AMR res

    grid_kwargs = {
        "left_edge": [x_left, y_bottom, 0],
        "right_edge": [x_right, y_top, 1],
        "dims": [nxx, nyy, 1]
    }
    print(grid_kwargs)
    atmos_params = {}
    ds_attrs = {}
    if args.sparse:
        assert nxx % args.block_size == 0 and nyy % args.block_size == 0, "Resolution must be a multiple of the block_size"
        tag = YTTiledArbitraryGrid(
            grid_kwargs['left_edge'],
            grid_kwargs['right_edge'],
            grid_kwargs['dims'],
            [args.block_size, args.block_size, 1],
            ds=ds
        )
        num_blocks = [nxx // args.block_size, nyy // args.block_size]
        morton_order = []
        for z in range(num_blocks[1]):
            for x in range(num_blocks[0]):
                morton_order.append(encode_morton_2(np.int32(x), np.int32(z)))
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
            dex_x, dex_z = decode_morton_2(code)
            amrvac_grid_coord = [dex_x, dex_z]
            amrvac_grid_idx = np.ravel_multi_index(amrvac_grid_coord, num_blocks)

            block = tag.single_arbitrary_grid(amrvac_grid_idx)
            if np.any(block['temperature'].to('K').value < temperature_threshold):
                active_blocks.append(code)

                temperature_blocks.append(block['temperature'].to('K').value.squeeze().astype(DTYPE))
                # pressure_blocks.append(block[('gas', 'thermal_pressure')].to('Pa').value.squeeze().astype(DTYPE))
                pressure_blocks.append((block['e'] * unit_pressure / 10).value.squeeze().astype(DTYPE))
                vx_blocks.append(block['velocity_1'].to('m/s').value.squeeze().astype(DTYPE))
                vy_blocks.append(block['velocity_2'].to('m/s').value.squeeze().astype(DTYPE))
                vz_blocks.append(block['velocity_3'].to('m/s').value.squeeze().astype(DTYPE))

        temperature = np.concatenate([np.ascontiguousarray(x.T).reshape(-1) for x in temperature_blocks])
        pressure = np.concatenate([np.ascontiguousarray(x.T).reshape(-1) for x in pressure_blocks])
        v_x = np.concatenate([np.ascontiguousarray(x.T).reshape(-1) for x in vx_blocks])
        v_y = np.concatenate([np.ascontiguousarray(x.T).reshape(-1) for x in vy_blocks])
        v_z = np.concatenate([np.ascontiguousarray(x.T).reshape(-1) for x in vz_blocks])
        field_shape = ("cells",)
        active_blocks = np.array(active_blocks, dtype=np.uint32)

        atmos_params = atmos_params | dict(
            morton_tiles=(["num_active_tiles"], active_blocks),
        )
        ds_attrs = ds_attrs | dict(
            num_x=nxx,
            num_x_blocks=num_blocks[0],
            num_y=1,
            num_y_blocks=1,
            num_z=nyy,
            num_z_blocks=num_blocks[1],
            block_size=args.block_size,
            program='dexrt (2d)', # masquerade
            output_format='sparse',
            sparse=1,
        )
    else:
        grid = ds.arbitrary_grid(**grid_kwargs)
        # pressure = grid[('gas', 'thermal_pressure')].to('Pa').value.squeeze().astype(DTYPE)
        pressure = (grid['e'] * unit_pressure / 10).T.value.squeeze().astype(DTYPE)
        temperature = grid['temperature'].to('K').T.value.squeeze().astype(DTYPE)
        v_x = grid['velocity_1'].to('m/s').T.value.squeeze().astype(DTYPE)
        v_y = grid['velocity_2'].to('m/s').T.value.squeeze().astype(DTYPE)
        v_z = grid['velocity_3'].to('m/s').T.value.squeeze().astype(DTYPE)
        field_shape = ("z", "x")

    if ZERO_VY:
        # NOTE(cmo): Not a typo. v_z is amrvac z, which is dex y
        v_z[...] = 0.0

    vturb = vturb_fn(temperature).astype(DTYPE)

    initial_ionisation_fraction = 0.8
    n_tot = pressure / (const.k_B.value * temperature)
    nh_tot = n_tot / (lw.DefaultAtomicAbundance.totalAbundance * (1.0 + initial_ionisation_fraction))
    nh_tot = nh_tot.astype(DTYPE)
    ne = lw.DefaultAtomicAbundance.totalAbundance * nh_tot * initial_ionisation_fraction
    ne = ne.astype(DTYPE)

    assert abs(max_res[0] - max_res[1]) < 1e-2, "Check aspect ratio"
    voxel_scale = np.array([max_res[0]], dtype=DTYPE)

    offset_z = np.array([base_altitude], dtype=DTYPE)
    offset_x = np.array([-0.5 * nxx * voxel_scale.item()], dtype=DTYPE)

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
        offset_z=offset_z
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
